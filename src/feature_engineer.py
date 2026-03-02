"""
Phase 1: Feature Engineering Pipeline
FVG Probability Modeling Project

Computes market context features around competing FVG scenarios.
Five feature categories:
  1. Distance Features
  2. Momentum Features
  3. FVG Characteristics
  4. Volatility & Market Regime
  5. Higher Timeframe Context

All feature functions are designed to prevent lookahead bias:
  - Only data at or before current_idx is used
  - No future prices or future FVG information is accessed
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List


# ---------------------------------------------------------------------------
# 1. Distance Features (7 features)
# ---------------------------------------------------------------------------

def compute_distance_features(current_price: float, upper_fvg: Dict,
                               lower_fvg: Dict, atr: float) -> Dict:
    """
    Compute distance-based features between current price and competing FVGs.

    Parameters
    ----------
    current_price : float
        Current closing price.
    upper_fvg : dict
        FVG above current price with keys 'gap_mid', 'gap_low', 'gap_high'.
    lower_fvg : dict
        FVG below current price with keys 'gap_mid', 'gap_low', 'gap_high'.
    atr : float
        Average True Range for normalization.

    Returns
    -------
    dict with 7 distance features.
    """
    dist_to_upper_mid = upper_fvg['gap_mid'] - current_price
    dist_to_lower_mid = current_price - lower_fvg['gap_mid']

    return {
        # Absolute distances
        'dist_to_upper_mid': dist_to_upper_mid,
        'dist_to_lower_mid': dist_to_lower_mid,

        # ATR-normalized distances
        'dist_to_upper_atr': dist_to_upper_mid / atr if atr > 0 else 0.0,
        'dist_to_lower_atr': dist_to_lower_mid / atr if atr > 0 else 0.0,

        # Distance ratio (asymmetry) — >1 means upper is farther
        'distance_ratio': (dist_to_upper_mid / dist_to_lower_mid
                           if dist_to_lower_mid > 0 else 0.0),

        # Closest edge distances
        'dist_to_upper_edge': upper_fvg['gap_low'] - current_price,
        'dist_to_lower_edge': current_price - lower_fvg['gap_high'],
    }


# ---------------------------------------------------------------------------
# 2. Momentum Features (7 features)
# ---------------------------------------------------------------------------

def compute_momentum_features(candles: pd.DataFrame,
                               current_idx: int) -> Dict:
    """
    Compute momentum and trend features from OHLCV candles.

    Only uses data at or before current_idx (no lookahead).

    Parameters
    ----------
    candles : pd.DataFrame
        OHLCV data with datetime index.
    current_idx : int
        Position-based index of the current candle.

    Returns
    -------
    dict with 7 momentum features.
    """
    current_price = candles.iloc[current_idx]['close']

    # Short-term returns
    def _ret(lookback: int) -> float:
        if current_idx >= lookback:
            return current_price / candles.iloc[current_idx - lookback]['close'] - 1
        return 0.0

    return_5 = _ret(5)
    return_10 = _ret(10)
    return_20 = _ret(20)

    # Trend flags
    is_uptrend_10 = (1 if current_idx >= 10
                     and current_price > candles.iloc[current_idx - 10]['close']
                     else 0)
    is_uptrend_20 = (1 if current_idx >= 20
                     and current_price > candles.iloc[current_idx - 20]['close']
                     else 0)

    # Moving average
    window_start = max(0, current_idx - 19)  # 20-period lookback inclusive
    ma_20 = candles.iloc[window_start:current_idx + 1]['close'].mean()
    price_vs_ma20 = (current_price / ma_20 - 1) if current_idx >= 20 else 0.0

    return {
        'return_5': return_5,
        'return_10': return_10,
        'return_20': return_20,
        'is_uptrend_10': is_uptrend_10,
        'is_uptrend_20': is_uptrend_20,
        'ma_20': ma_20,
        'price_vs_ma20': price_vs_ma20,
    }


# ---------------------------------------------------------------------------
# 3. FVG Characteristics (10 features)
# ---------------------------------------------------------------------------

def compute_fvg_features(upper_fvg: Dict, lower_fvg: Dict,
                          current_idx: int, atr: float) -> Dict:
    """
    Compute FVG-specific characteristic features.

    Parameters
    ----------
    upper_fvg : dict
        FVG above current price, as returned by fvg_detector.identify_fvg().
    lower_fvg : dict
        FVG below current price.
    current_idx : int
        Current candle index in the 1H dataset.
    atr : float
        Average True Range for normalization.

    Returns
    -------
    dict with 10 FVG characteristic features.
    """
    return {
        # FVG sizes (raw and ATR-normalized)
        'upper_fvg_size': upper_fvg['gap_size'],
        'lower_fvg_size': lower_fvg['gap_size'],
        'upper_fvg_size_atr': upper_fvg['gap_size'] / atr if atr > 0 else 0.0,
        'lower_fvg_size_atr': lower_fvg['gap_size'] / atr if atr > 0 else 0.0,

        # FVG ages (candles since formation)
        'upper_fvg_age': current_idx - upper_fvg['formation_index'],
        'lower_fvg_age': current_idx - lower_fvg['formation_index'],

        # FVG formation volume (proxy for strength)
        'upper_fvg_volume': upper_fvg['candle_2']['volume'],
        'lower_fvg_volume': lower_fvg['candle_2']['volume'],

        # Impulse candle body size (directional conviction)
        'upper_fvg_impulse_size': abs(upper_fvg['candle_2']['close']
                                      - upper_fvg['candle_2']['open']),
        'lower_fvg_impulse_size': abs(lower_fvg['candle_2']['close']
                                      - lower_fvg['candle_2']['open']),
    }


# ---------------------------------------------------------------------------
# 4. Volatility & Market Regime (5 features)
# ---------------------------------------------------------------------------

def compute_volatility_features(candles: pd.DataFrame, current_idx: int,
                                 period: int = 20) -> Dict:
    """
    Compute volatility and market regime features.

    Uses only candles at or before current_idx (no lookahead).

    Parameters
    ----------
    candles : pd.DataFrame
        OHLCV data.
    current_idx : int
        Current candle position.
    period : int
        Lookback window for volatility calculation.

    Returns
    -------
    dict with 5 volatility features (empty dict if insufficient data).
    """
    if current_idx < period:
        return {
            'atr': 0.0,
            'realized_volatility': 0.0,
            'volatility_percentile': 0.0,
            'avg_candle_range': 0.0,
            'current_vs_avg_range': 0.0,
        }

    recent = candles.iloc[current_idx - period:current_idx + 1]

    # True Range
    high = recent['high']
    low = recent['low']
    close_prev = recent['close'].shift(1)

    tr = pd.concat([
        high - low,
        (high - close_prev).abs(),
        (low - close_prev).abs()
    ], axis=1).max(axis=1)

    atr = tr.mean()

    # Realized volatility (annualized for hourly data)
    returns = recent['close'].pct_change().dropna()
    realized_vol = returns.std() * np.sqrt(252 * 24) if len(returns) > 1 else 0.0

    # Average candle range
    candle_ranges = recent['high'] - recent['low']
    avg_range = candle_ranges.mean()

    current_range = (candles.iloc[current_idx]['high']
                     - candles.iloc[current_idx]['low'])
    current_vs_avg = current_range / avg_range if avg_range > 0 else 0.0

    return {
        'atr': atr,
        'realized_volatility': realized_vol,
        'volatility_percentile': 0.0,  # filled by build_feature_dataset
        'avg_candle_range': avg_range,
        'current_vs_avg_range': current_vs_avg,
    }


# ---------------------------------------------------------------------------
# 5. Higher Timeframe Context (3 features)
# ---------------------------------------------------------------------------

def compute_htf_features(candles_4h: pd.DataFrame,
                          candles_daily: pd.DataFrame,
                          current_time: pd.Timestamp) -> Dict:
    """
    Compute higher-timeframe trend context.

    Parameters
    ----------
    candles_4h : pd.DataFrame
        4-hour OHLCV data with datetime index.
    candles_daily : pd.DataFrame
        Daily OHLCV data with datetime index.
    current_time : pd.Timestamp
        Timestamp of the current 1H candle.

    Returns
    -------
    dict with 3 HTF features.
    """
    # Find corresponding 4H bar via forward-fill lookup
    idx_4h = candles_4h.index.get_indexer([current_time], method='ffill')[0]
    idx_daily = candles_daily.index.get_indexer([current_time], method='ffill')[0]

    # 4H trend (10-bar lookback)
    if idx_4h >= 10:
        trend_4h = (1 if candles_4h.iloc[idx_4h]['close']
                    > candles_4h.iloc[idx_4h - 10]['close'] else -1)
    else:
        trend_4h = 0

    # Daily trend (5-bar lookback)
    if idx_daily >= 5:
        trend_daily = (1 if candles_daily.iloc[idx_daily]['close']
                       > candles_daily.iloc[idx_daily - 5]['close'] else -1)
    else:
        trend_daily = 0

    return {
        'htf_trend_4h': trend_4h,
        'htf_trend_daily': trend_daily,
        'htf_alignment': 1 if trend_4h == trend_daily else 0,
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def extract_all_features(candles_1h: pd.DataFrame,
                          candles_4h: pd.DataFrame,
                          candles_daily: pd.DataFrame,
                          upper_fvg: Dict,
                          lower_fvg: Dict,
                          current_idx: int) -> Dict:
    """
    Extract all features for a single observation.

    Parameters
    ----------
    candles_1h : pd.DataFrame
        1-hour OHLCV data.
    candles_4h : pd.DataFrame
        4-hour OHLCV data.
    candles_daily : pd.DataFrame
        Daily OHLCV data.
    upper_fvg : dict
        FVG above current price (from fvg_detector).
    lower_fvg : dict
        FVG below current price (from fvg_detector).
    current_idx : int
        Position in the 1H dataset.

    Returns
    -------
    dict with all 32 features.
    """
    current_price = candles_1h.iloc[current_idx]['close']
    current_time = candles_1h.index[current_idx]

    # Volatility first — ATR needed for normalization
    vol_features = compute_volatility_features(candles_1h, current_idx)
    atr = vol_features['atr']

    features = {
        **compute_distance_features(current_price, upper_fvg, lower_fvg, atr),
        **compute_momentum_features(candles_1h, current_idx),
        **compute_fvg_features(upper_fvg, lower_fvg, current_idx, atr),
        **vol_features,
        **compute_htf_features(candles_4h, candles_daily, current_time),
    }

    return features


# ---------------------------------------------------------------------------
# Batch Feature Builder
# ---------------------------------------------------------------------------

def build_feature_dataset(candles_1h: pd.DataFrame,
                           candles_4h: pd.DataFrame,
                           candles_daily: pd.DataFrame,
                           fvg_list: List[Dict],
                           min_idx: int = 25) -> pd.DataFrame:
    """
    Build a DataFrame of features from a list of detected FVGs.

    For each FVG, we pair it with the nearest opposing FVG (one above, one
    below the current price at the evaluation point) and extract features.

    If no valid pair can be formed (e.g. no FVG exists on the other side),
    the FVG is skipped.

    Parameters
    ----------
    candles_1h : pd.DataFrame
        Hourly OHLCV with datetime index.
    candles_4h : pd.DataFrame
        4-Hour OHLCV with datetime index.
    candles_daily : pd.DataFrame
        Daily OHLCV with datetime index.
    fvg_list : list of dict
        All detected FVGs from fvg_detector.scan_all_fvgs().
    min_idx : int
        Skip FVGs formed before this index (insufficient lookback).

    Returns
    -------
    pd.DataFrame with one row per valid FVG observation and all features.
    """
    if not fvg_list:
        return pd.DataFrame()

    # Sort FVGs by formation index
    sorted_fvgs = sorted(fvg_list, key=lambda f: f['formation_index'])

    rows = []
    for i, fvg in enumerate(sorted_fvgs):
        eval_idx = fvg['formation_index'] + 2  # evaluate 2 candles after formation
        if eval_idx >= len(candles_1h) or eval_idx < min_idx:
            continue

        current_price = candles_1h.iloc[eval_idx]['close']

        # Find nearest FVG above and below current price from prior FVGs
        upper_fvg = None
        lower_fvg = None

        for j in range(i - 1, -1, -1):
            candidate = sorted_fvgs[j]
            if candidate['gap_mid'] > current_price and upper_fvg is None:
                upper_fvg = candidate
            elif candidate['gap_mid'] <= current_price and lower_fvg is None:
                lower_fvg = candidate
            if upper_fvg is not None and lower_fvg is not None:
                break

        # Also check the current FVG itself
        if fvg['gap_mid'] > current_price:
            upper_fvg = upper_fvg or fvg
            if lower_fvg is None:
                continue  # no opposing FVG
        else:
            lower_fvg = lower_fvg or fvg
            if upper_fvg is None:
                continue  # no opposing FVG

        try:
            features = extract_all_features(
                candles_1h, candles_4h, candles_daily,
                upper_fvg, lower_fvg, eval_idx
            )
            # Add metadata
            features['eval_time'] = candles_1h.index[eval_idx]
            features['eval_idx'] = eval_idx
            features['primary_fvg_type'] = fvg['type']
            features['primary_fvg_formation_time'] = fvg['formation_time']
            rows.append(features)
        except Exception:
            continue  # skip problematic observations

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Fill volatility_percentile via rolling rank
    if 'realized_volatility' in df.columns and len(df) > 1:
        df['volatility_percentile'] = (
            df['realized_volatility']
            .rank(pct=True)
        )

    return df
