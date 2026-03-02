"""
Tests for Phase 5: Strategy Optimizer + Production API + Model Monitor
"""

import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.strategy_optimizer import (
    compute_target,
    compute_stop,
    _simulate,
)
from src.model_monitor import ModelMonitor


# ── Strategy Optimizer ──────────────────────────────────────────────────

class TestComputeTarget:

    def test_long_edge(self):
        upper = {'gap_low': 1.10, 'gap_high': 1.12, 'gap_mid': 1.11}
        lower = {'gap_low': 1.05, 'gap_high': 1.07, 'gap_mid': 1.06}
        t = compute_target('LONG', 1.08, upper, lower, mode='edge')
        assert t == 1.10, "LONG edge should target upper gap_low"

    def test_long_partial(self):
        upper = {'gap_low': 1.10, 'gap_high': 1.12, 'gap_mid': 1.11}
        lower = {'gap_low': 1.05, 'gap_high': 1.07, 'gap_mid': 1.06}
        t = compute_target('LONG', 1.08, upper, lower, mode='partial')
        assert t == pytest.approx((1.10 + 1.11) / 2)

    def test_long_midpoint(self):
        upper = {'gap_low': 1.10, 'gap_high': 1.12, 'gap_mid': 1.11}
        lower = {'gap_low': 1.05, 'gap_high': 1.07, 'gap_mid': 1.06}
        t = compute_target('LONG', 1.08, upper, lower, mode='midpoint')
        assert t == 1.11

    def test_short_edge(self):
        upper = {'gap_low': 1.10, 'gap_high': 1.12, 'gap_mid': 1.11}
        lower = {'gap_low': 1.05, 'gap_high': 1.07, 'gap_mid': 1.06}
        t = compute_target('SHORT', 1.08, upper, lower, mode='edge')
        assert t == 1.07, "SHORT edge should target lower gap_high"

    def test_short_midpoint(self):
        upper = {'gap_low': 1.10, 'gap_high': 1.12, 'gap_mid': 1.11}
        lower = {'gap_low': 1.05, 'gap_high': 1.07, 'gap_mid': 1.06}
        t = compute_target('SHORT', 1.08, upper, lower, mode='midpoint')
        assert t == 1.06


class TestComputeStop:

    def test_long_atr(self):
        upper = {'gap_mid': 1.11}
        lower = {'gap_mid': 1.06}
        s = compute_stop('LONG', 1.08, upper, lower, atr=0.01,
                         mode='atr', atr_multiplier=1.5)
        assert s == pytest.approx(1.08 - 0.015)

    def test_short_atr(self):
        upper = {'gap_mid': 1.11}
        lower = {'gap_mid': 1.06}
        s = compute_stop('SHORT', 1.08, upper, lower, atr=0.01,
                         mode='atr', atr_multiplier=2.0)
        assert s == pytest.approx(1.08 + 0.02)

    def test_long_fixed(self):
        upper = {'gap_mid': 1.11}
        lower = {'gap_mid': 1.06}
        s = compute_stop('LONG', 1.08, upper, lower, atr=0.01,
                         mode='fixed', fixed_pips=0.003)
        assert s == pytest.approx(1.08 - 0.003)

    def test_long_opposite(self):
        upper = {'gap_mid': 1.11}
        lower = {'gap_mid': 1.06}
        s = compute_stop('LONG', 1.08, upper, lower, atr=0.01,
                         mode='opposite')
        assert s == 1.06

    def test_short_opposite(self):
        upper = {'gap_mid': 1.11}
        lower = {'gap_mid': 1.06}
        s = compute_stop('SHORT', 1.08, upper, lower, atr=0.01,
                         mode='opposite')
        assert s == 1.11


class TestSimulate:

    def _candles(self, prices):
        rows = [{'open': p, 'high': p + 0.0005, 'low': p - 0.0005,
                 'close': p, 'volume': 100} for p in prices]
        return pd.DataFrame(rows, index=pd.date_range(
            '2023-01-01', periods=len(prices), freq='h'))

    def test_long_win(self):
        c = self._candles([1.10] * 3 + [1.12] * 3)
        r = _simulate(c, 1, 1.10, 1.115, 1.09, 'LONG')
        assert r['result'] == 'WIN'

    def test_short_loss(self):
        c = self._candles([1.10] * 3 + [1.12] * 3)
        r = _simulate(c, 1, 1.10, 1.08, 1.115, 'SHORT')
        assert r['result'] == 'LOSS'


# ── Model Monitor ───────────────────────────────────────────────────────

class TestModelMonitor:

    def test_log_and_resolve(self, tmp_path):
        log = str(tmp_path / 'test.jsonl')
        mon = ModelMonitor(log_path=log, rolling_window=5)

        rec = mon.log_prediction(
            timestamp='2024-01-01T00:00:00',
            prob_upper=0.75,
            bias='LONG',
            confidence=0.75,
            current_price=1.10,
            upper_fvg_mid=1.12,
            lower_fvg_mid=1.06,
        )
        assert rec['prediction_id'].startswith('pred_')
        assert rec['outcome'] is None

        updated = mon.log_outcome(rec['prediction_id'], 'upper')
        assert updated['correct'] is True

    def test_summary(self, tmp_path):
        log = str(tmp_path / 'test2.jsonl')
        mon = ModelMonitor(log_path=log, rolling_window=3)

        for i in range(5):
            r = mon.log_prediction(
                timestamp=f'2024-01-{i+1:02d}T00:00:00',
                prob_upper=0.7 if i % 2 == 0 else 0.3,
                bias='LONG' if i % 2 == 0 else 'SHORT',
                confidence=0.7,
                current_price=1.10, upper_fvg_mid=1.12,
                lower_fvg_mid=1.06)
            mon.log_outcome(r['prediction_id'],
                            'upper' if i % 2 == 0 else 'lower')

        summary = mon.get_summary()
        assert summary['total_predictions'] == 5
        assert summary['resolved'] == 5
        assert summary['accuracy'] == 1.0  # all correct

    def test_persistence(self, tmp_path):
        log = str(tmp_path / 'persist.jsonl')
        mon = ModelMonitor(log_path=log)
        mon.log_prediction(
            timestamp='2024-01-01', prob_upper=0.6, bias='LONG',
            confidence=0.6, current_price=1.10,
            upper_fvg_mid=1.12, lower_fvg_mid=1.06)

        # Reload
        mon2 = ModelMonitor(log_path=log)
        assert mon2.get_summary()['total_predictions'] == 1


# ── Production API ──────────────────────────────────────────────────────

class TestProductionAPIHealth:

    def test_health_imports(self):
        """Verify that the app can be imported without crashing."""
        # This just tests the import path; full HTTP tests need httpx
        from src.production_api import create_app
        assert callable(create_app)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
