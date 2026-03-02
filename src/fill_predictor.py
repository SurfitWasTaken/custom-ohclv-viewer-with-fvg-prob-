"""
V2 Fill Predictor — Real-Time FVG Fill Probability

Lightweight prediction engine for live use.
Loads the trained multi-horizon XGBoost models and provides per-FVG
fill probabilities at multiple horizons.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fvg_detector import scan_all_fvgs
from src.fill_feature_engineer import extract_fvg_features, FEATURE_COLS
from src.dataset_builder import is_fvg_mitigated

HORIZONS = [1, 2, 5, 10, 20, 50, 100]


class FillPredictor:
    """
    Real-time FVG fill probability predictor.

    Usage:
        predictor = FillPredictor('models/')
        results = predictor.predict_all(candles_1h, candles_4h, candles_daily)
    """

    def __init__(self, model_dir: str = 'models'):
        model_path = os.path.join(model_dir, 'survival_model.pkl')
        cols_path = os.path.join(model_dir, 'feature_cols.json')

        # models is a dict: {horizon: XGBClassifier}
        self.models = joblib.load(model_path)

        with open(cols_path) as f:
            self.feature_cols = json.load(f)

        print(f"FillPredictor loaded: {len(self.models)} horizon models")

    def predict_single(self, candles_1h: pd.DataFrame,
                        fvg: Dict,
                        candles_4h: Optional[pd.DataFrame] = None,
                        candles_daily: Optional[pd.DataFrame] = None,
                        all_fvgs: Optional[List[Dict]] = None) -> Dict:
        """Predict fill probabilities for a single FVG."""
        feats = extract_fvg_features(
            candles_1h, fvg, fvg['formation_index'],
            candles_4h, candles_daily, all_fvgs)

        X = np.array([[feats[c] for c in self.feature_cols]], dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict each horizon and enforce monotonicity
        probs = {}
        prev = 0.0
        for h in sorted(HORIZONS):
            if h in self.models:
                p = float(self.models[h].predict_proba(X)[0, 1])
                p = max(p, prev)  # monotonic
                probs[h] = round(p, 4)
                prev = p

        # Urgency classification
        p10 = probs.get(10, 0)
        if p10 > 0.80:
            urgency = 'imminent'
        elif p10 > 0.40:
            urgency = 'moderate'
        else:
            urgency = 'low'

        return {
            'fvg_type': fvg['type'],
            'gap_low': round(float(fvg['gap_low']), 5),
            'gap_high': round(float(fvg['gap_high']), 5),
            'gap_mid': round(float(fvg['gap_mid']), 5),
            'gap_size': round(float(fvg['gap_size']), 5),
            'formation_idx': fvg['formation_index'],
            'formation_time': str(fvg['formation_time']),
            'fill_probabilities': probs,
            'urgency': urgency,
        }

    def predict_all(self, candles_1h: pd.DataFrame,
                     candles_4h: Optional[pd.DataFrame] = None,
                     candles_daily: Optional[pd.DataFrame] = None,
                     max_age: int = 100) -> Dict:
        """Predict fill probabilities for ALL active (unmitigated) FVGs."""
        all_fvgs = scan_all_fvgs(candles_1h)
        current_idx = len(candles_1h) - 1
        current_price = float(candles_1h.iloc[-1]['close'])

        active = []
        for fvg in all_fvgs:
            age = current_idx - fvg['formation_index']
            if age < 2 or age > max_age:
                continue
            if is_fvg_mitigated(candles_1h, fvg, current_idx + 1):
                continue

            try:
                pred = self.predict_single(
                    candles_1h, fvg, candles_4h, candles_daily, all_fvgs)
                pred['age_candles'] = age
                active.append(pred)
            except Exception:
                continue

        # Bias summary
        bullish_below = [f for f in active
                         if f['fvg_type'] == 'bullish' and f['gap_high'] < current_price]
        bearish_above = [f for f in active
                         if f['fvg_type'] == 'bearish' and f['gap_low'] > current_price]

        if bullish_below and bearish_above:
            net_bias = 'competing'
        elif bearish_above and not bullish_below:
            net_bias = 'bullish_pull'
        elif bullish_below and not bearish_above:
            net_bias = 'bearish_pull'
        else:
            net_bias = 'no_active_fvgs'

        return {
            'current_price': round(current_price, 5),
            'active_fvgs': sorted(active, key=lambda x: x['age_candles']),
            'bias_summary': {
                'bullish_below': len(bullish_below),
                'bearish_above': len(bearish_above),
                'net_bias': net_bias,
                'total_active': len(active),
            },
        }
