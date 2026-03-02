"""
Tests for Phase 4: Backtesting & Validation
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.backtester import (
    simulate_trade_outcome,
    sensitivity_analysis,
    permutation_test,
)


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_candles(prices: list) -> pd.DataFrame:
    """Create a minimal candle DataFrame from a list of close prices."""
    rows = []
    for i, p in enumerate(prices):
        rows.append({
            'open': p - 0.0001,
            'high': p + 0.0005,
            'low': p - 0.0005,
            'close': p,
            'volume': 100,
        })
    idx = pd.date_range('2023-01-01', periods=len(prices), freq='h')
    return pd.DataFrame(rows, index=idx)


# =====================================================================
# simulate_trade_outcome
# =====================================================================

class TestSimulateTradeOutcome:

    def test_long_win(self):
        """LONG trade hits target within window."""
        prices = [1.1000] * 5 + [1.1080] * 5  # price jumps up
        candles = _make_candles(prices)
        result = simulate_trade_outcome(
            candles, entry_idx=2,
            entry_price=1.1000, target=1.1050, stop_loss=1.0950,
            direction='LONG')
        assert result['result'] == 'WIN'
        assert result['pnl'] > 0

    def test_long_loss(self):
        """LONG trade hits stop loss."""
        prices = [1.1000] * 3 + [1.0900] * 5  # price drops
        candles = _make_candles(prices)
        result = simulate_trade_outcome(
            candles, entry_idx=1,
            entry_price=1.1000, target=1.1100, stop_loss=1.0950,
            direction='LONG')
        assert result['result'] == 'LOSS'
        assert result['pnl'] < 0

    def test_short_win(self):
        """SHORT trade hits target."""
        prices = [1.1000] * 3 + [1.0900] * 5  # price drops
        candles = _make_candles(prices)
        result = simulate_trade_outcome(
            candles, entry_idx=1,
            entry_price=1.1000, target=1.0950, stop_loss=1.1100,
            direction='SHORT')
        assert result['result'] == 'WIN'
        assert result['pnl'] > 0

    def test_short_loss(self):
        """SHORT trade hits stop loss."""
        prices = [1.1000] * 3 + [1.1150] * 5  # price jumps up
        candles = _make_candles(prices)
        result = simulate_trade_outcome(
            candles, entry_idx=1,
            entry_price=1.1000, target=1.0900, stop_loss=1.1100,
            direction='SHORT')
        assert result['result'] == 'LOSS'
        assert result['pnl'] < 0

    def test_timeout(self):
        """Trade neither hits target nor stop within window."""
        # Price stays flat, target and stop far away
        prices = [1.1000] * 20
        candles = _make_candles(prices)
        result = simulate_trade_outcome(
            candles, entry_idx=0,
            entry_price=1.1000, target=1.2000, stop_loss=1.0000,
            direction='LONG', max_hold_candles=10)
        assert result['result'] == 'TIMEOUT'

    def test_no_lookahead(self):
        """Ensure trade only inspects candles AFTER entry_idx."""
        # Target was hit on candle before entry → should NOT count
        prices = [1.1200, 1.1000, 1.1001, 1.1002, 1.1003]
        candles = _make_candles(prices)
        result = simulate_trade_outcome(
            candles, entry_idx=1,
            entry_price=1.1000, target=1.1100, stop_loss=1.0900,
            direction='LONG', max_hold_candles=3)
        # Only candles idx 2,3,4 are inspected; none reach 1.1100
        assert result['result'] == 'TIMEOUT'


# =====================================================================
# sensitivity_analysis
# =====================================================================

class TestSensitivityAnalysis:

    def test_basic(self):
        """Sensitivity analysis returns expected columns."""
        df = pd.DataFrame({
            'prob_confidence': [0.55, 0.60, 0.65, 0.70, 0.75,
                                0.80, 0.85, 0.55, 0.62, 0.78],
            'pnl': [0.001, -0.002, 0.003, 0.001, -0.001,
                    0.002, 0.004, -0.001, 0.001, 0.002],
            'result': ['WIN', 'LOSS', 'WIN', 'WIN', 'LOSS',
                       'WIN', 'WIN', 'LOSS', 'WIN', 'WIN'],
        })
        out = sensitivity_analysis(df, np.arange(0.50, 0.81, 0.10))
        assert 'threshold' in out.columns
        assert 'n_trades' in out.columns
        assert 'win_rate' in out.columns
        assert len(out) > 0

    def test_fewer_trades_at_higher_threshold(self):
        """Higher thresholds should yield fewer or equal trades."""
        df = pd.DataFrame({
            'prob_confidence': np.random.uniform(0.5, 0.95, 100),
            'pnl': np.random.normal(0, 0.001, 100),
            'result': np.random.choice(['WIN', 'LOSS'], 100),
        })
        out = sensitivity_analysis(df)
        # n_trades should be non-increasing
        for i in range(1, len(out)):
            assert out.iloc[i]['n_trades'] <= out.iloc[i - 1]['n_trades']


# =====================================================================
# permutation_test
# =====================================================================

class TestPermutationTest:

    def test_significant(self):
        """Strong positive PnL should yield low p-value."""
        df = pd.DataFrame({
            'pnl': [0.01] * 50 + [-0.001] * 10,
        })
        result = permutation_test(df, n_permutations=500)
        assert result['p_value'] < 0.1

    def test_random(self):
        """Zero-mean PnL should yield high p-value."""
        np.random.seed(42)
        df = pd.DataFrame({'pnl': np.random.normal(0, 0.001, 200)})
        result = permutation_test(df, n_permutations=500)
        # p-value should not be extremely low for random data
        assert result['p_value'] > 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
