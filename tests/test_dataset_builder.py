"""
Unit Tests for Phase 2: Dataset Builder
FVG Probability Modeling Project

Tests cover:
  - FVG mitigation check
  - Competing scenario scanning
  - Outcome labeling (upper, lower, neither, both_same_candle)
  - Dataset validation
"""

import pytest
import pandas as pd
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_builder import (
    is_fvg_mitigated,
    find_competing_fvg_scenarios,
    label_scenario_outcome,
    validate_dataset,
)


# ── Helpers ───────────────────────────────────────────────────────────────

def make_candles(n: int = 200, start_price: float = 1.2000,
                 freq: str = 'h') -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=n, freq=freq, tz='UTC')
    closes = [start_price]
    for _ in range(n - 1):
        closes.append(closes[-1] + np.random.normal(0, 0.001))
    closes = np.array(closes)
    highs = closes + np.abs(np.random.normal(0, 0.0005, n))
    lows = closes - np.abs(np.random.normal(0, 0.0005, n))
    opens = closes + np.random.normal(0, 0.0003, n)
    volumes = np.random.randint(500, 5000, n)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    return pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': volumes,
    }, index=dates)


def make_fvg(fvg_type: str, gap_mid: float, gap_size: float = 0.0015,
             formation_index: int = 10) -> dict:
    half = gap_size / 2
    return {
        'type': fvg_type,
        'gap_high': gap_mid + half,
        'gap_low': gap_mid - half,
        'gap_mid': gap_mid,
        'gap_size': gap_size,
        'formation_index': formation_index,
        'formation_time': pd.Timestamp('2023-01-01 10:00', tz='UTC'),
        'candle_1': {'open': 1.20, 'high': 1.201, 'low': 1.199,
                     'close': 1.200, 'volume': 1500,
                     'time': pd.Timestamp('2023-01-01 09:00', tz='UTC')},
        'candle_2': {'open': 1.201, 'high': 1.205, 'low': 1.200,
                     'close': 1.204, 'volume': 3000,
                     'time': pd.Timestamp('2023-01-01 10:00', tz='UTC')},
        'candle_3': {'open': 1.204, 'high': 1.206, 'low': 1.203,
                     'close': 1.205, 'volume': 2000,
                     'time': pd.Timestamp('2023-01-01 11:00', tz='UTC')},
    }


# ── Mitigation Tests ─────────────────────────────────────────────────────

class TestIsFvgMitigated:
    def test_bullish_not_mitigated(self):
        """Bullish FVG below; price never touches gap_high => unmitigated."""
        candles = make_candles(50)
        fvg = make_fvg('bullish', gap_mid=1.1800, formation_index=5)
        # Candle prices are near 1.2000, well above 1.1800
        assert is_fvg_mitigated(candles, fvg, 30) is False

    def test_bearish_not_mitigated(self):
        """Bearish FVG above; price never touches gap_low => unmitigated."""
        candles = make_candles(50)
        fvg = make_fvg('bearish', gap_mid=1.2500, formation_index=5)
        assert is_fvg_mitigated(candles, fvg, 30) is False

    def test_bullish_mitigated(self):
        """Bullish FVG where price drops into the gap => mitigated."""
        candles = make_candles(50)
        fvg = make_fvg('bullish', gap_mid=1.2000, gap_size=0.01,
                        formation_index=5)
        # gap_high = 1.2050.  Candle lows near 1.199x, so low <= gap_high
        assert is_fvg_mitigated(candles, fvg, 30) is True

    def test_only_checks_before_current(self):
        """Should not check candles at or after current_idx."""
        candles = make_candles(50)
        fvg = make_fvg('bearish', gap_mid=1.2500, formation_index=5)
        # Make candle at idx=25 pierce the gap
        candles.iloc[25, candles.columns.get_loc('high')] = 1.26
        # Check before the pierce — not mitigated
        assert is_fvg_mitigated(candles, fvg, 25) is False
        # Check after the pierce — mitigated
        assert is_fvg_mitigated(candles, fvg, 30) is True


# ── Scenario Scanner Tests ────────────────────────────────────────────────

class TestFindCompetingScenarios:
    def test_returns_list_of_dicts(self):
        candles = make_candles(300)
        from src.fvg_detector import scan_all_fvgs
        fvgs = scan_all_fvgs(candles)
        scenarios = find_competing_fvg_scenarios(candles, fvgs)
        assert isinstance(scenarios, list)
        if scenarios:
            assert 'upper_fvg' in scenarios[0]
            assert 'lower_fvg' in scenarios[0]
            assert 'current_idx' in scenarios[0]

    def test_upper_above_lower_below(self):
        """upper_fvg must be above, lower_fvg must be below current price."""
        candles = make_candles(300)
        from src.fvg_detector import scan_all_fvgs
        fvgs = scan_all_fvgs(candles)
        scenarios = find_competing_fvg_scenarios(candles, fvgs)
        for s in scenarios:
            assert s['upper_fvg']['gap_low'] > s['current_price'], \
                "Upper FVG gap_low must be above current price"
            assert s['lower_fvg']['gap_high'] < s['current_price'], \
                "Lower FVG gap_high must be below current price"


# ── Labeling Tests ────────────────────────────────────────────────────────

class TestLabelScenarioOutcome:
    def test_upper_hit_first(self):
        """When price rises to upper FVG first, outcome = 'upper'."""
        candles = make_candles(300)
        # Set up scenario: price at 1.20, upper at 1.201, lower at 1.18
        scenario = {
            'current_idx': 150,
            'current_price': candles.iloc[150]['close'],
            'upper_fvg': make_fvg('bearish', gap_mid=candles.iloc[150]['close'] + 0.0005,
                                  gap_size=0.001, formation_index=140),
            'lower_fvg': make_fvg('bullish', gap_mid=candles.iloc[150]['close'] - 0.05,
                                  gap_size=0.001, formation_index=140),
        }
        result = label_scenario_outcome(candles, scenario, max_candles_forward=100)
        # Upper is very close, lower is very far => upper should be hit
        assert result['outcome'] == 'upper'
        assert result['candles_to_hit'] >= 1

    def test_lower_hit_first(self):
        """When price drops to lower FVG first, outcome = 'lower'."""
        candles = make_candles(300)
        scenario = {
            'current_idx': 150,
            'current_price': candles.iloc[150]['close'],
            'upper_fvg': make_fvg('bearish', gap_mid=candles.iloc[150]['close'] + 0.05,
                                  gap_size=0.001, formation_index=140),
            'lower_fvg': make_fvg('bullish', gap_mid=candles.iloc[150]['close'] - 0.0005,
                                  gap_size=0.001, formation_index=140),
        }
        result = label_scenario_outcome(candles, scenario, max_candles_forward=100)
        assert result['outcome'] == 'lower'
        assert result['candles_to_hit'] >= 1

    def test_neither_hit(self):
        """Both FVGs very far => 'neither' within window."""
        candles = make_candles(300)
        scenario = {
            'current_idx': 150,
            'current_price': candles.iloc[150]['close'],
            'upper_fvg': make_fvg('bearish', gap_mid=candles.iloc[150]['close'] + 0.5,
                                  gap_size=0.001, formation_index=140),
            'lower_fvg': make_fvg('bullish', gap_mid=candles.iloc[150]['close'] - 0.5,
                                  gap_size=0.001, formation_index=140),
        }
        result = label_scenario_outcome(candles, scenario, max_candles_forward=10)
        assert result['outcome'] == 'neither'
        assert result['candles_to_hit'] is None

    def test_outcome_keys(self):
        candles = make_candles(300)
        scenario = {
            'current_idx': 150,
            'current_price': candles.iloc[150]['close'],
            'upper_fvg': make_fvg('bearish', gap_mid=1.25, formation_index=140),
            'lower_fvg': make_fvg('bullish', gap_mid=1.15, formation_index=140),
        }
        result = label_scenario_outcome(candles, scenario)
        assert set(result.keys()) == {'outcome', 'candles_to_hit', 'hit_candle_idx'}


# ── Validation Tests ──────────────────────────────────────────────────────

class TestValidateDataset:
    def test_clean_dataset_passes(self):
        """A clean dataset should produce PASS status."""
        np.random.seed(0)
        df = pd.DataFrame({
            'feature_a': np.random.randn(200),
            'feature_b': np.random.randn(200),
            'target': np.random.choice([0, 1], 200, p=[0.5, 0.5]),
            'candles_to_hit': np.random.randint(1, 50, 200),
            'scenario_idx': range(200),
        })
        report = validate_dataset(df)
        assert report['status'] == 'WARNING'  # < 1000 samples

    def test_missing_values_detected(self):
        df = pd.DataFrame({
            'feature_a': [1.0, np.nan, 3.0],
            'target': [0, 1, 0],
            'candles_to_hit': [5, 10, 15],
            'scenario_idx': [0, 1, 2],
        })
        report = validate_dataset(df)
        assert 'feature_a' in report['missing_columns']


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
