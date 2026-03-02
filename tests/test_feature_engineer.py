"""
Unit Tests for Phase 1: Feature Engineering
FVG Probability Modeling Project

Tests cover:
  - All 5 feature category functions
  - Full pipeline orchestration
  - Lookahead bias prevention
  - Edge cases (insufficient history)
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineer import (
    compute_distance_features,
    compute_momentum_features,
    compute_fvg_features,
    compute_volatility_features,
    compute_htf_features,
    extract_all_features,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def make_candles(n: int = 50, start_price: float = 1.2000,
                 freq: str = 'h') -> pd.DataFrame:
    """Generate synthetic OHLCV data with realistic price movement."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq=freq, tz='UTC')
    closes = [start_price]
    for _ in range(n - 1):
        change = np.random.normal(0, 0.001)
        closes.append(closes[-1] + change)

    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(0, 0.0005, n))
    lows = closes - np.abs(np.random.normal(0, 0.0005, n))
    opens = closes + np.random.normal(0, 0.0003, n)
    volumes = np.random.randint(500, 5000, n)

    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
    }, index=dates)
    return df


def make_fvg(fvg_type: str = 'bullish', gap_mid: float = 1.2100,
             gap_size: float = 0.0015, formation_index: int = 10) -> dict:
    """Build a synthetic FVG dict matching fvg_detector output schema."""
    half = gap_size / 2
    return {
        'type': fvg_type,
        'gap_high': gap_mid + half,
        'gap_low': gap_mid - half,
        'gap_mid': gap_mid,
        'gap_size': gap_size,
        'formation_index': formation_index,
        'formation_time': pd.Timestamp('2023-01-01 10:00', tz='UTC'),
        'candle_1': {
            'open': 1.2085, 'high': 1.2090, 'low': 1.2080,
            'close': 1.2088, 'volume': 1500,
            'time': pd.Timestamp('2023-01-01 09:00', tz='UTC'),
        },
        'candle_2': {
            'open': 1.2090, 'high': 1.2115, 'low': 1.2088,
            'close': 1.2110, 'volume': 3000,
            'time': pd.Timestamp('2023-01-01 10:00', tz='UTC'),
        },
        'candle_3': {
            'open': 1.2112, 'high': 1.2120, 'low': 1.2108,
            'close': 1.2118, 'volume': 2000,
            'time': pd.Timestamp('2023-01-01 11:00', tz='UTC'),
        },
    }


# ── Distance Features ────────────────────────────────────────────────────

class TestDistanceFeatures:
    def test_basic_distances(self):
        upper = make_fvg('bearish', gap_mid=1.2100)
        lower = make_fvg('bullish', gap_mid=1.1900)
        price = 1.2000
        atr = 0.0020

        result = compute_distance_features(price, upper, lower, atr)

        assert result['dist_to_upper_mid'] == pytest.approx(0.0100, abs=1e-6)
        assert result['dist_to_lower_mid'] == pytest.approx(0.0100, abs=1e-6)
        assert result['distance_ratio'] == pytest.approx(1.0, abs=1e-4)
        assert result['dist_to_upper_atr'] == pytest.approx(5.0, abs=0.1)
        assert result['dist_to_lower_atr'] == pytest.approx(5.0, abs=0.1)

    def test_asymmetric_distances(self):
        upper = make_fvg('bearish', gap_mid=1.2200)
        lower = make_fvg('bullish', gap_mid=1.1900)
        price = 1.2000
        atr = 0.0020

        result = compute_distance_features(price, upper, lower, atr)

        assert result['dist_to_upper_mid'] > result['dist_to_lower_mid']
        assert result['distance_ratio'] > 1.0

    def test_zero_atr_safety(self):
        upper = make_fvg('bearish', gap_mid=1.2100)
        lower = make_fvg('bullish', gap_mid=1.1900)
        result = compute_distance_features(1.2000, upper, lower, atr=0.0)

        assert result['dist_to_upper_atr'] == 0.0
        assert result['dist_to_lower_atr'] == 0.0

    def test_edge_distances(self):
        upper = make_fvg('bearish', gap_mid=1.2100, gap_size=0.0020)
        lower = make_fvg('bullish', gap_mid=1.1900, gap_size=0.0020)
        price = 1.2000
        atr = 0.0020

        result = compute_distance_features(price, upper, lower, atr)

        # Upper edge = gap_low = gap_mid - half_size = 1.2100 - 0.001 = 1.2090
        assert result['dist_to_upper_edge'] == pytest.approx(0.0090, abs=1e-6)
        # Lower edge = gap_high = gap_mid + half_size = 1.1900 + 0.001 = 1.1910
        assert result['dist_to_lower_edge'] == pytest.approx(0.0090, abs=1e-6)

    def test_all_keys_present(self):
        upper = make_fvg('bearish', gap_mid=1.2100)
        lower = make_fvg('bullish', gap_mid=1.1900)
        result = compute_distance_features(1.2000, upper, lower, 0.002)

        expected_keys = {
            'dist_to_upper_mid', 'dist_to_lower_mid',
            'dist_to_upper_atr', 'dist_to_lower_atr',
            'distance_ratio', 'dist_to_upper_edge', 'dist_to_lower_edge',
        }
        assert set(result.keys()) == expected_keys


# ── Momentum Features ────────────────────────────────────────────────────

class TestMomentumFeatures:
    def test_returns_and_trends(self):
        candles = make_candles(50)
        result = compute_momentum_features(candles, 30)

        assert 'return_5' in result
        assert 'return_10' in result
        assert 'return_20' in result
        assert result['is_uptrend_10'] in (0, 1)
        assert result['is_uptrend_20'] in (0, 1)
        assert isinstance(result['ma_20'], float)

    def test_insufficient_history(self):
        candles = make_candles(50)
        result = compute_momentum_features(candles, 3)

        assert result['return_5'] == 0.0
        assert result['return_10'] == 0.0
        assert result['return_20'] == 0.0
        assert result['is_uptrend_10'] == 0
        assert result['is_uptrend_20'] == 0
        assert result['price_vs_ma20'] == 0.0

    def test_positive_return(self):
        candles = make_candles(50, start_price=1.0)
        # Artificially set a rising pattern
        candles.iloc[20, candles.columns.get_loc('close')] = 1.0000
        candles.iloc[25, candles.columns.get_loc('close')] = 1.0050

        result = compute_momentum_features(candles, 25)
        assert result['return_5'] > 0

    def test_all_keys_present(self):
        candles = make_candles(50)
        result = compute_momentum_features(candles, 30)

        expected_keys = {
            'return_5', 'return_10', 'return_20',
            'is_uptrend_10', 'is_uptrend_20',
            'ma_20', 'price_vs_ma20',
        }
        assert set(result.keys()) == expected_keys


# ── FVG Characteristics ──────────────────────────────────────────────────

class TestFvgFeatures:
    def test_basic_fvg_features(self):
        upper = make_fvg('bearish', gap_size=0.0020, formation_index=5)
        lower = make_fvg('bullish', gap_size=0.0010, formation_index=8)
        atr = 0.0015

        result = compute_fvg_features(upper, lower, current_idx=20, atr=atr)

        assert result['upper_fvg_size'] == 0.0020
        assert result['lower_fvg_size'] == 0.0010
        assert result['upper_fvg_age'] == 15
        assert result['lower_fvg_age'] == 12
        assert result['upper_fvg_size_atr'] == pytest.approx(0.0020 / 0.0015, rel=1e-4)

    def test_volumes(self):
        upper = make_fvg('bearish')
        lower = make_fvg('bullish')

        result = compute_fvg_features(upper, lower, 20, 0.002)

        assert result['upper_fvg_volume'] == 3000
        assert result['lower_fvg_volume'] == 3000

    def test_impulse_size(self):
        upper = make_fvg('bearish')
        lower = make_fvg('bullish')

        result = compute_fvg_features(upper, lower, 20, 0.002)

        # candle_2: open=1.2090, close=1.2110 → impulse = 0.0020
        assert result['upper_fvg_impulse_size'] == pytest.approx(0.0020, abs=1e-6)

    def test_all_keys_present(self):
        upper = make_fvg('bearish')
        lower = make_fvg('bullish')
        result = compute_fvg_features(upper, lower, 20, 0.002)

        expected_keys = {
            'upper_fvg_size', 'lower_fvg_size',
            'upper_fvg_size_atr', 'lower_fvg_size_atr',
            'upper_fvg_age', 'lower_fvg_age',
            'upper_fvg_volume', 'lower_fvg_volume',
            'upper_fvg_impulse_size', 'lower_fvg_impulse_size',
        }
        assert set(result.keys()) == expected_keys


# ── Volatility Features ──────────────────────────────────────────────────

class TestVolatilityFeatures:
    def test_sufficient_data(self):
        candles = make_candles(50)
        result = compute_volatility_features(candles, 30)

        assert result['atr'] > 0
        assert result['realized_volatility'] > 0
        assert result['avg_candle_range'] > 0
        assert result['current_vs_avg_range'] > 0

    def test_insufficient_data(self):
        candles = make_candles(50)
        result = compute_volatility_features(candles, 5, period=20)

        assert result['atr'] == 0.0
        assert result['realized_volatility'] == 0.0

    def test_all_keys_present(self):
        candles = make_candles(50)
        result = compute_volatility_features(candles, 30)

        expected_keys = {
            'atr', 'realized_volatility', 'volatility_percentile',
            'avg_candle_range', 'current_vs_avg_range',
        }
        assert set(result.keys()) == expected_keys


# ── Higher Timeframe Features ────────────────────────────────────────────

class TestHTFFeatures:
    def test_uptrend_alignment(self):
        candles_4h = make_candles(50, start_price=1.19, freq='4h')
        candles_daily = make_candles(30, start_price=1.19, freq='D')

        # Set a clear uptrend: later prices higher than earlier
        for i in range(len(candles_4h)):
            candles_4h.iloc[i, candles_4h.columns.get_loc('close')] = 1.19 + i * 0.001
        for i in range(len(candles_daily)):
            candles_daily.iloc[i, candles_daily.columns.get_loc('close')] = 1.19 + i * 0.003

        current_time = candles_4h.index[30]
        result = compute_htf_features(candles_4h, candles_daily, current_time)

        assert result['htf_trend_4h'] == 1
        assert result['htf_trend_daily'] == 1
        assert result['htf_alignment'] == 1

    def test_downtrend(self):
        candles_4h = make_candles(50, start_price=1.25, freq='4h')
        candles_daily = make_candles(30, start_price=1.25, freq='D')

        for i in range(len(candles_4h)):
            candles_4h.iloc[i, candles_4h.columns.get_loc('close')] = 1.25 - i * 0.001
        for i in range(len(candles_daily)):
            candles_daily.iloc[i, candles_daily.columns.get_loc('close')] = 1.25 - i * 0.003

        current_time = candles_4h.index[30]
        result = compute_htf_features(candles_4h, candles_daily, current_time)

        assert result['htf_trend_4h'] == -1
        assert result['htf_trend_daily'] == -1
        assert result['htf_alignment'] == 1  # both down = aligned

    def test_insufficient_history(self):
        candles_4h = make_candles(5, freq='4h')
        candles_daily = make_candles(3, freq='D')

        current_time = candles_4h.index[2]
        result = compute_htf_features(candles_4h, candles_daily, current_time)

        assert result['htf_trend_4h'] == 0
        assert result['htf_trend_daily'] == 0

    def test_all_keys_present(self):
        candles_4h = make_candles(50, freq='4h')
        candles_daily = make_candles(30, freq='D')
        current_time = candles_4h.index[30]

        result = compute_htf_features(candles_4h, candles_daily, current_time)

        expected_keys = {'htf_trend_4h', 'htf_trend_daily', 'htf_alignment'}
        assert set(result.keys()) == expected_keys


# ── Full Pipeline ─────────────────────────────────────────────────────────

class TestExtractAllFeatures:
    def test_returns_all_32_features(self):
        candles_1h = make_candles(50, freq='h')
        candles_4h = make_candles(50, freq='4h')
        candles_daily = make_candles(30, freq='D')

        upper = make_fvg('bearish', gap_mid=1.2100, formation_index=5)
        lower = make_fvg('bullish', gap_mid=1.1900, formation_index=8)

        result = extract_all_features(
            candles_1h, candles_4h, candles_daily,
            upper, lower, current_idx=30
        )

        # 7 distance + 7 momentum + 10 fvg + 5 volatility + 3 htf = 32
        assert len(result) == 32

    def test_all_values_numeric(self):
        candles_1h = make_candles(50, freq='h')
        candles_4h = make_candles(50, freq='4h')
        candles_daily = make_candles(30, freq='D')

        upper = make_fvg('bearish', gap_mid=1.2100, formation_index=5)
        lower = make_fvg('bullish', gap_mid=1.1900, formation_index=8)

        result = extract_all_features(
            candles_1h, candles_4h, candles_daily,
            upper, lower, current_idx=30
        )

        for key, val in result.items():
            assert isinstance(val, (int, float, np.integer, np.floating)), \
                f"Feature '{key}' has type {type(val)}, expected numeric"


# ── Lookahead Bias ────────────────────────────────────────────────────────

class TestNoLookaheadBias:
    def test_momentum_no_future_data(self):
        """Verify momentum features are identical regardless of future candles."""
        # Generate one long dataset, then slice to create a short version
        # This guarantees candles 0..30 are identical in both.
        candles_long = make_candles(100, freq='h')
        candles_short = candles_long.iloc[:31].copy()

        result_short = compute_momentum_features(candles_short, 30)
        result_long = compute_momentum_features(candles_long, 30)

        for key in result_short:
            assert result_short[key] == pytest.approx(result_long[key], abs=1e-10), \
                f"Lookahead bias detected in momentum feature '{key}'"

    def test_volatility_no_future_data(self):
        """Verify volatility features are identical regardless of future candles."""
        candles_long = make_candles(100, freq='h')
        candles_short = candles_long.iloc[:31].copy()

        result_short = compute_volatility_features(candles_short, 30)
        result_long = compute_volatility_features(candles_long, 30)

        for key in result_short:
            assert result_short[key] == pytest.approx(result_long[key], abs=1e-10), \
                f"Lookahead bias detected in volatility feature '{key}'"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
