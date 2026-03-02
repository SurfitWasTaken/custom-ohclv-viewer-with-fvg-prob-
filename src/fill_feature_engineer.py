"""
V2 Feature Engineering — Single-FVG Fill Probability

Extracts features for an individual FVG (not a competing pair).
Designed for survival analysis: predict time-to-fill.

Feature categories:
  1. Gap properties       (size, vol, geometry)
  2. Distance from price  (raw + ATR-normalised)
  3. Momentum             (returns, trend direction)
  4. Volatility           (ATR, realised vol)
  5. HTF context          (4H/daily trend alignment)
  6. Market context       (session hour, nearby FVG density)
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, List


# ── helpers ─────────────────────────────────────────────────────────────

def _safe(val, default=0.0):
    """Replace NaN/Inf with default."""
    if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))):
        return default
    return float(val)


def _atr(candles: pd.DataFrame, idx: int, period: int = 14) -> float:
    """Average True Range at index."""
    start = max(0, idx - period)
    subset = candles.iloc[start:idx + 1]
    if len(subset) < 2:
        return 0.001
    high = subset['high'].values
    low = subset['low'].values
    close = subset['close'].values
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - prev_close),
                               np.abs(low - prev_close)))
    return float(np.mean(tr[-period:]))


def _returns(candles: pd.DataFrame, idx: int, period: int) -> float:
    """Log return over period candles ending at idx."""
    if idx < period:
        return 0.0
    c_now = candles.iloc[idx]['close']
    c_prev = candles.iloc[idx - period]['close']
    if c_prev == 0:
        return 0.0
    return float(np.log(c_now / c_prev))


def _trend(candles: pd.DataFrame, idx: int, period: int = 20) -> float:
    """Linear regression slope over last `period` candles, normalised by ATR."""
    start = max(0, idx - period + 1)
    closes = candles.iloc[start:idx + 1]['close'].values
    if len(closes) < 5:
        return 0.0
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    atr = _atr(candles, idx, 14)
    return float(slope / atr) if atr > 0 else 0.0


def _realised_vol(candles: pd.DataFrame, idx: int, period: int = 20) -> float:
    """Annualised realised volatility from log returns."""
    start = max(1, idx - period + 1)
    closes = candles.iloc[start:idx + 1]['close'].values
    if len(closes) < 5:
        return 0.0
    log_rets = np.diff(np.log(closes))
    return float(np.std(log_rets) * np.sqrt(252 * 24))  # hourly data


def _htf_trend_at(htf_candles: pd.DataFrame, timestamp,
                   period: int = 20) -> float:
    """Get HTF trend direction (-1 to +1) at or before the given timestamp."""
    if htf_candles is None or htf_candles.empty:
        return 0.0
    mask = htf_candles.index <= timestamp
    if mask.sum() < 5:
        return 0.0
    subset = htf_candles.loc[mask].iloc[-period:]
    closes = subset['close'].values
    if len(closes) < 5:
        return 0.0
    x = np.arange(len(closes))
    slope = np.polyfit(x, closes, 1)[0]
    std = np.std(closes)
    return float(np.clip(slope / std if std > 0 else 0, -1, 1))


# ── main feature extractor ─────────────────────────────────────────────

def extract_fvg_features(candles_1h: pd.DataFrame,
                          fvg: Dict,
                          formation_idx: int,
                          candles_4h: Optional[pd.DataFrame] = None,
                          candles_daily: Optional[pd.DataFrame] = None,
                          all_fvgs: Optional[List[Dict]] = None) -> Dict:
    """
    Extract features for a single FVG at formation time.

    Parameters
    ----------
    candles_1h : pd.DataFrame
        Hourly OHLCV.
    fvg : dict
        FVG from scan_all_fvgs().
    formation_idx : int
        Index in candles_1h where the FVG was formed.
    candles_4h, candles_daily : pd.DataFrame, optional
        Higher-timeframe data for trend features.
    all_fvgs : list of dict, optional
        All FVGs for nearby-density features.

    Returns
    -------
    dict of feature_name -> value
    """
    c = candles_1h
    idx = formation_idx
    atr = _atr(c, idx)
    current_price = float(c.iloc[idx]['close'])
    gap_size = float(fvg['gap_size'])
    gap_mid = float(fvg['gap_mid'])
    gap_high = float(fvg['gap_high'])
    gap_low = float(fvg['gap_low'])
    is_bullish = fvg['type'] == 'bullish'

    feats = {}

    # ── 1. Gap properties ──────────────────────────────────────────
    feats['gap_size'] = _safe(gap_size)
    feats['gap_size_atr'] = _safe(gap_size / atr if atr > 0 else 0)
    feats['gap_volume'] = _safe(float(fvg['candle_2']['volume']))

    # Impulse candle geometry
    c2 = fvg['candle_2']
    body = abs(c2['close'] - c2['open'])
    total_range = c2['high'] - c2['low']
    feats['impulse_ratio'] = _safe(body / total_range if total_range > 0 else 0)
    feats['body_to_atr'] = _safe(body / atr if atr > 0 else 0)

    # FVG type
    feats['is_bullish'] = 1.0 if is_bullish else 0.0

    # ── 2. Distance from price at formation ────────────────────────
    if is_bullish:
        dist = current_price - gap_high  # positive = price above gap
    else:
        dist = gap_low - current_price   # positive = price below gap

    feats['dist_to_gap'] = _safe(dist)
    feats['dist_to_gap_atr'] = _safe(dist / atr if atr > 0 else 0)
    feats['gap_above_price'] = 1.0 if gap_mid > current_price else 0.0

    # Gap position relative to recent range
    lookback = min(50, idx)
    if lookback > 0:
        recent_high = float(c.iloc[idx - lookback:idx + 1]['high'].max())
        recent_low = float(c.iloc[idx - lookback:idx + 1]['low'].min())
        rng = recent_high - recent_low
        feats['gap_pct_of_range'] = _safe(
            (gap_mid - recent_low) / rng if rng > 0 else 0.5)
    else:
        feats['gap_pct_of_range'] = 0.5

    # ── 3. Momentum ────────────────────────────────────────────────
    feats['ret_5'] = _safe(_returns(c, idx, 5))
    feats['ret_10'] = _safe(_returns(c, idx, 10))
    feats['ret_20'] = _safe(_returns(c, idx, 20))
    feats['trend_20'] = _safe(_trend(c, idx, 20))

    # Momentum toward/away from gap
    # Positive = price is moving TOWARD the gap
    ret_5 = feats['ret_5']
    if is_bullish:
        feats['momentum_toward_gap'] = _safe(-ret_5)  # bearish move → toward bullish gap
    else:
        feats['momentum_toward_gap'] = _safe(ret_5)   # bullish move → toward bearish gap

    # ── 4. Volatility ──────────────────────────────────────────────
    feats['atr_14'] = _safe(atr)
    feats['realized_vol'] = _safe(_realised_vol(c, idx))

    # Vol percentile vs lookback
    if idx >= 50:
        recent_atrs = [_atr(c, i) for i in range(idx - 49, idx + 1)]
        feats['vol_percentile'] = _safe(
            sum(1 for a in recent_atrs if a <= atr) / len(recent_atrs))
    else:
        feats['vol_percentile'] = 0.5

    # ── 5. HTF context ─────────────────────────────────────────────
    ts = c.index[idx]
    feats['trend_4h'] = _safe(_htf_trend_at(candles_4h, ts))
    feats['trend_daily'] = _safe(_htf_trend_at(candles_daily, ts))

    # Does HTF trend support fill?
    # Bullish FVG fills require bearish move → HTF bearish supports fill
    # Bearish FVG fills require bullish move → HTF bullish supports fill
    if is_bullish:
        feats['htf_supports_fill'] = _safe(-feats['trend_4h'])
    else:
        feats['htf_supports_fill'] = _safe(feats['trend_4h'])

    # ── 6. Market context ──────────────────────────────────────────
    if hasattr(ts, 'hour'):
        feats['session_hour'] = float(ts.hour)
    else:
        feats['session_hour'] = 0.0

    if hasattr(ts, 'weekday'):
        feats['day_of_week'] = float(ts.weekday())
    else:
        feats['day_of_week'] = 0.0

    # ── 7. Nearby FVG density ──────────────────────────────────────
    if all_fvgs is not None:
        n_same = 0
        n_opposite = 0
        for other in all_fvgs:
            age = idx - other['formation_index']
            if age < 2 or age > 50:
                continue
            if other['formation_index'] == fvg['formation_index']:
                continue
            if other['type'] == fvg['type']:
                n_same += 1
            else:
                n_opposite += 1
        feats['n_nearby_same'] = float(n_same)
        feats['n_nearby_opposite'] = float(n_opposite)
    else:
        feats['n_nearby_same'] = 0.0
        feats['n_nearby_opposite'] = 0.0

    return feats


# ── feature column list ─────────────────────────────────────────────────

FEATURE_COLS = [
    'gap_size', 'gap_size_atr', 'gap_volume',
    'impulse_ratio', 'body_to_atr',
    'is_bullish',
    'dist_to_gap', 'dist_to_gap_atr', 'gap_above_price', 'gap_pct_of_range',
    'ret_5', 'ret_10', 'ret_20', 'trend_20',
    'momentum_toward_gap',
    'atr_14', 'realized_vol', 'vol_percentile',
    'trend_4h', 'trend_daily', 'htf_supports_fill',
    'session_hour', 'day_of_week',
    'n_nearby_same', 'n_nearby_opposite',
]
