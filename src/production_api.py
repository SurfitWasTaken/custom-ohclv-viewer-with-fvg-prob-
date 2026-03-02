"""
Phase 5 — Component 2: Production API
FVG Probability Modeling Project

FastAPI-based REST service for real-time FVG probability prediction.

Endpoints:
  GET  /health    — liveness check
  POST /predict   — FVG prediction from recent candle data

Usage:
  uvicorn src.production_api:app --host 0.0.0.0 --port 5000
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fvg_detector import identify_fvg, scan_all_fvgs
from src.dataset_builder import is_fvg_mitigated
from src.feature_engineer import extract_all_features
from src.strategy_optimizer import compute_target, compute_stop
from src.model_monitor import ModelMonitor


# =====================================================================
# Pydantic schemas
# =====================================================================

class Candle(BaseModel):
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float = 0


class PredictRequest(BaseModel):
    candles_1h: List[Candle] = Field(
        ..., description="Recent 1H candles (≥200 required)")
    candles_4h: List[Candle] = Field(
        ..., description="Recent 4H candles (≥50 required)")
    candles_daily: List[Candle] = Field(
        ..., description="Recent daily candles (≥30 required)")
    symbol: str = Field(default='EURUSD', description="Currency pair")


class FVGInfo(BaseModel):
    type: str
    gap_low: float
    gap_high: float
    gap_mid: float
    gap_size: float


class PredictResponse(BaseModel):
    status: str
    symbol: str
    timestamp: str
    current_price: float
    upper_fvg: Optional[FVGInfo] = None
    lower_fvg: Optional[FVGInfo] = None
    prob_upper: Optional[float] = None
    prob_lower: Optional[float] = None
    bias: Optional[str] = None
    confidence: Optional[float] = None
    optimized_target: Optional[float] = None
    optimized_stop: Optional[float] = None
    risk_reward: Optional[float] = None
    prediction_id: Optional[str] = None
    latency_ms: Optional[float] = None


# =====================================================================
# RealTimeFVGPredictor
# =====================================================================

class RealTimeFVGPredictor:
    """
    Production-ready FVG probability predictor.

    Loads model artifacts at init, then scores new candle data on demand.
    """

    def __init__(self, model_dir: str = 'models'):
        self.model_dir = model_dir
        self.model = joblib.load(os.path.join(model_dir, 'xgboost_model.pkl'))

        with open(os.path.join(model_dir, 'feature_cols.json')) as f:
            self.feature_cols = json.load(f)

        # Load optimal config from Phase 5 optimizer (if available)
        opt_path = os.path.join(model_dir, 'phase5', 'optimal_config.json')
        if os.path.exists(opt_path):
            with open(opt_path) as f:
                self.optimal_config = json.load(f)
        else:
            self.optimal_config = {
                'target_mode': 'midpoint',
                'stop_mode': 'opposite',
            }

        self.monitor = ModelMonitor(
            log_path=os.path.join(model_dir, 'phase5', 'prediction_log.jsonl'))

    def predict(self,
                candles_1h: pd.DataFrame,
                candles_4h: pd.DataFrame,
                candles_daily: pd.DataFrame,
                symbol: str = 'EURUSD') -> dict:
        """
        Run prediction pipeline.

        Returns dict with prediction details or {'status': 'no_setup'}.
        """
        t0 = time.time()

        # Detect FVGs in the 1H data
        all_fvgs = scan_all_fvgs(candles_1h)
        current_idx = len(candles_1h) - 1
        current_price = candles_1h.iloc[current_idx]['close']
        current_time = candles_1h.index[current_idx]

        # Find active unmitigated FVGs
        active_bearish = [
            fvg for fvg in all_fvgs
            if fvg['type'] == 'bearish'
            and 2 <= (current_idx - fvg['formation_index']) <= 100
            and fvg['gap_low'] > current_price
            and not is_fvg_mitigated(candles_1h, fvg, current_idx)
        ]
        active_bullish = [
            fvg for fvg in all_fvgs
            if fvg['type'] == 'bullish'
            and 2 <= (current_idx - fvg['formation_index']) <= 100
            and fvg['gap_high'] < current_price
            and not is_fvg_mitigated(candles_1h, fvg, current_idx)
        ]

        if not active_bearish or not active_bullish:
            return {
                'status': 'no_setup',
                'symbol': symbol,
                'timestamp': str(current_time),
                'current_price': float(current_price),
                'latency_ms': round((time.time() - t0) * 1000, 1),
            }

        upper_fvg = min(active_bearish,
                        key=lambda x: x['gap_mid'] - current_price)
        lower_fvg = min(active_bullish,
                        key=lambda x: current_price - x['gap_mid'])

        # Extract features
        features = extract_all_features(
            candles_1h, candles_4h, candles_daily,
            upper_fvg, lower_fvg, current_idx)

        X = np.array([[features[c] for c in self.feature_cols]])
        prob_upper = float(self.model.predict_proba(X)[0, 1])
        prob_lower = 1 - prob_upper

        # Bias
        if prob_upper > 0.6:
            bias = 'LONG'
        elif prob_upper < 0.4:
            bias = 'SHORT'
        else:
            bias = 'NEUTRAL'

        confidence = max(prob_upper, prob_lower)

        # Compute optimized target / stop
        direction = 'LONG' if prob_upper > 0.5 else 'SHORT'
        atr = features.get('atr', 0.001)

        opt_target = compute_target(
            direction, float(current_price),
            upper_fvg, lower_fvg,
            mode=self.optimal_config.get('target_mode', 'midpoint'))

        opt_stop = compute_stop(
            direction, float(current_price),
            upper_fvg, lower_fvg, atr,
            mode=self.optimal_config.get('stop_mode', 'opposite'))

        risk = abs(float(current_price) - opt_stop)
        reward = abs(opt_target - float(current_price))
        rr = reward / risk if risk > 0 else 0

        # Log prediction
        record = self.monitor.log_prediction(
            timestamp=str(current_time),
            prob_upper=prob_upper,
            bias=bias,
            confidence=confidence,
            current_price=float(current_price),
            upper_fvg_mid=upper_fvg['gap_mid'],
            lower_fvg_mid=lower_fvg['gap_mid'],
            target=opt_target,
            stop_loss=opt_stop,
        )

        latency = round((time.time() - t0) * 1000, 1)

        return {
            'status': 'prediction_ready',
            'symbol': symbol,
            'timestamp': str(current_time),
            'current_price': float(current_price),
            'upper_fvg': {
                'type': upper_fvg['type'],
                'gap_low': float(upper_fvg['gap_low']),
                'gap_high': float(upper_fvg['gap_high']),
                'gap_mid': float(upper_fvg['gap_mid']),
                'gap_size': float(upper_fvg['gap_size']),
            },
            'lower_fvg': {
                'type': lower_fvg['type'],
                'gap_low': float(lower_fvg['gap_low']),
                'gap_high': float(lower_fvg['gap_high']),
                'gap_mid': float(lower_fvg['gap_mid']),
                'gap_size': float(lower_fvg['gap_size']),
            },
            'prob_upper': round(prob_upper, 6),
            'prob_lower': round(prob_lower, 6),
            'bias': bias,
            'confidence': round(confidence, 6),
            'optimized_target': round(opt_target, 6),
            'optimized_stop': round(opt_stop, 6),
            'risk_reward': round(rr, 3),
            'prediction_id': record['prediction_id'],
            'latency_ms': latency,
        }


# =====================================================================
# FastAPI application
# =====================================================================

def _candles_to_df(candles: List[Candle]) -> pd.DataFrame:
    """Convert a list of Candle Pydantic models to a pandas DataFrame."""
    records = [c.model_dump() for c in candles]
    df = pd.DataFrame(records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    df = df.set_index('timestamp').sort_index()
    return df


def create_app(model_dir: str = 'models') -> FastAPI:
    """Factory to create the FastAPI app with a predictor instance."""
    app = FastAPI(
        title='FVG Probability API',
        description='Real-time Fair Value Gap probability prediction',
        version='1.0.0',
    )

    predictor = RealTimeFVGPredictor(model_dir=model_dir)

    @app.get('/')
    def root():
        return {
            'message': 'FVG Probability API is running',
            'endpoints': {
                'health': '/health',
                'docs': '/docs',
                'redoc': '/redoc'
            }
        }

    @app.get('/health')
    def health():
        return {
            'status': 'healthy',
            'model_loaded': predictor.model is not None,
            'n_features': len(predictor.feature_cols),
            'optimal_config': predictor.optimal_config,
            'monitor': predictor.monitor.get_summary(),
        }

    @app.post('/predict', response_model=PredictResponse)
    def predict(req: PredictRequest):
        # Validate minimum candle counts
        if len(req.candles_1h) < 200:
            raise HTTPException(400, "Need ≥200 1H candles")
        if len(req.candles_4h) < 50:
            raise HTTPException(400, "Need ≥50 4H candles")
        if len(req.candles_daily) < 30:
            raise HTTPException(400, "Need ≥30 daily candles")

        c1h = _candles_to_df(req.candles_1h)
        c4h = _candles_to_df(req.candles_4h)
        c1d = _candles_to_df(req.candles_daily)

        try:
            result = predictor.predict(c1h, c4h, c1d, symbol=req.symbol)
        except Exception as e:
            raise HTTPException(500, f"Prediction failed: {e}")

        return JSONResponse(content=result)

    @app.post('/outcome')
    def record_outcome(prediction_id: str, outcome: str):
        """Record the actual outcome of a prediction."""
        if outcome not in ('upper', 'lower', 'neither'):
            raise HTTPException(400, "outcome must be upper|lower|neither")
        rec = predictor.monitor.log_outcome(prediction_id, outcome)
        if rec is None:
            raise HTTPException(404, f"prediction_id {prediction_id} not found")
        return {'status': 'recorded', 'correct': rec['correct']}

    return app


# Default app instance (for `uvicorn src.production_api:app`)
app = create_app()
