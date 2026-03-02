"""
Institutional Walk-Forward Backtest — FVG Fill Probability Model

Walk-forward evaluation with:
  - Expanding-window training (no future data leakage)
  - 100-candle purge gap between train/test (matches longest horizon)
  - Programmatic leakage audit
  - Bootstrap confidence intervals (1,000 resamples)
  - Economic simulation (threshold-based entry)
  - Self-contained HTML report

Usage:
    python3 src/backtest_model.py
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import timedelta

warnings.filterwarnings('ignore', category=FutureWarning)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fill_feature_engineer import FEATURE_COLS
from src.survival_model import (
    prepare_features, make_horizon_target, predict_fill_probabilities,
    HORIZONS,
)


# ════════════════════════════════════════════════════════════════════════
#  1. WALK-FORWARD FOLD GENERATOR
# ════════════════════════════════════════════════════════════════════════

PURGE_CANDLES = 100  # purge gap = longest prediction horizon


def generate_walk_forward_folds(
    df: pd.DataFrame,
    initial_train_months: int = 18,
    test_months: int = 2,
    step_months: int = 2,
) -> List[Dict]:
    """
    Generate expanding-window walk-forward folds with purge gap.

    Each fold:
      - Train: all data from start to train_end
      - Purge: 100 candles (~4 days) after train_end are excluded
      - Test:  test_months of data after purge
    """
    df = df.sort_values('formation_time').reset_index(drop=True)
    min_date = df['formation_time'].min()
    max_date = df['formation_time'].max()

    folds = []
    fold_num = 0

    # First train_end = min_date + initial_train_months
    train_end = min_date + pd.DateOffset(months=initial_train_months)

    while True:
        # Purge boundary: train_end + purge gap (100 hours for hourly data)
        purge_end = train_end + timedelta(hours=PURGE_CANDLES)

        # Test window
        test_start = purge_end
        test_end = test_start + pd.DateOffset(months=test_months)

        # Check we have enough data
        if test_start >= max_date:
            break

        # Clip test_end to data end
        if test_end > max_date:
            test_end = max_date

        # Build masks
        train_mask = df['formation_time'] < train_end
        purge_mask = (df['formation_time'] >= train_end) & \
                     (df['formation_time'] < purge_end)
        test_mask = (df['formation_time'] >= test_start) & \
                    (df['formation_time'] <= test_end)

        n_train = train_mask.sum()
        n_purge = purge_mask.sum()
        n_test = test_mask.sum()

        if n_test < 50:  # skip tiny folds
            break

        folds.append({
            'fold': fold_num,
            'train_end': str(train_end),
            'purge_end': str(purge_end),
            'test_start': str(test_start),
            'test_end': str(test_end),
            'train_idx': df.index[train_mask].tolist(),
            'test_idx': df.index[test_mask].tolist(),
            'n_train': n_train,
            'n_purge': n_purge,
            'n_test': n_test,
        })

        fold_num += 1
        train_end += pd.DateOffset(months=step_months)

    return folds


# ════════════════════════════════════════════════════════════════════════
#  2. LEAKAGE AUDIT
# ════════════════════════════════════════════════════════════════════════

def leakage_audit(df: pd.DataFrame, folds: List[Dict]) -> Dict:
    """Programmatic leakage checks. Returns audit results."""
    results = {'passed': True, 'checks': []}

    for fold in folds:
        train_df = df.iloc[fold['train_idx']]
        test_df = df.iloc[fold['test_idx']]

        # Check 1: No temporal overlap
        train_max_time = train_df['formation_time'].max()
        test_min_time = test_df['formation_time'].min()
        gap_hours = (test_min_time - train_max_time).total_seconds() / 3600

        check1 = gap_hours >= PURGE_CANDLES
        results['checks'].append({
            'fold': fold['fold'],
            'check': 'purge_gap',
            'passed': check1,
            'detail': f"Gap={gap_hours:.0f}h (required≥{PURGE_CANDLES}h)",
        })
        if not check1:
            results['passed'] = False

        # Check 2: No shared indices
        overlap = set(fold['train_idx']) & set(fold['test_idx'])
        check2 = len(overlap) == 0
        results['checks'].append({
            'fold': fold['fold'],
            'check': 'no_shared_indices',
            'passed': check2,
            'detail': f"Shared indices: {len(overlap)}",
        })
        if not check2:
            results['passed'] = False

        # Check 3: Formation indices ensure backward-looking features
        if 'formation_idx' in test_df.columns:
            min_formation_idx = test_df['formation_idx'].min()
            check3 = min_formation_idx >= 50  # MIN_FVG_IDX
            results['checks'].append({
                'fold': fold['fold'],
                'check': 'lookback_available',
                'passed': check3,
                'detail': f"Min formation_idx in test: {min_formation_idx}",
            })
            if not check3:
                results['passed'] = False

    return results


# ════════════════════════════════════════════════════════════════════════
#  3. PER-FOLD TRAINING + EVALUATION
# ════════════════════════════════════════════════════════════════════════

def train_and_evaluate_fold(
    df: pd.DataFrame,
    fold: Dict,
) -> Dict:
    """Train models and compute all metrics for a single fold."""
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, brier_score_loss

    train_df = df.iloc[fold['train_idx']]
    test_df = df.iloc[fold['test_idx']]

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)

    models = {}
    metrics = {}
    all_preds = {}

    for h in HORIZONS:
        y_train = make_horizon_target(train_df, h)
        y_test = make_horizon_target(test_df, h)

        # Skip degenerate horizons
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            metrics[h] = {'auc': None, 'brier': None, 'fill_rate': float(y_train.mean())}
            all_preds[h] = np.full(len(y_test), y_train.mean())
            continue

        model = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            verbosity=0,
        )
        model.fit(X_train, y_train, verbose=False)
        models[h] = model

        p_test = model.predict_proba(X_test)[:, 1]
        all_preds[h] = p_test

        try:
            auc = roc_auc_score(y_test, p_test)
        except ValueError:
            auc = None

        brier = brier_score_loss(y_test, p_test)

        metrics[h] = {
            'auc': round(float(auc), 4) if auc is not None else None,
            'brier': round(float(brier), 4),
            'fill_rate': round(float(y_test.mean()), 4),
            'pred_mean': round(float(p_test.mean()), 4),
            'actual_mean': round(float(y_test.mean()), 4),
            'abs_cal_error': round(abs(float(p_test.mean()) - float(y_test.mean())), 4),
        }

    # Feature importance (average across horizons)
    feat_imp = {}
    for feat_i, feat_name in enumerate(FEATURE_COLS):
        imps = [models[h].feature_importances_[feat_i]
                for h in HORIZONS if h in models]
        feat_imp[feat_name] = round(float(np.mean(imps)), 4) if imps else 0.0

    return {
        'fold': fold['fold'],
        'n_train': fold['n_train'],
        'n_test': fold['n_test'],
        'train_end': fold['train_end'],
        'test_start': fold['test_start'],
        'test_end': fold['test_end'],
        'metrics': metrics,
        'predictions': all_preds,
        'actuals': {h: make_horizon_target(test_df, h) for h in HORIZONS},
        'test_df': test_df,
        'feature_importance': feat_imp,
    }


# ════════════════════════════════════════════════════════════════════════
#  4. BOOTSTRAP CONFIDENCE INTERVALS
# ════════════════════════════════════════════════════════════════════════

def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_boot: int = 1000,
                 alpha: float = 0.05) -> Dict:
    """Bootstrap CI for a metric function."""
    rng = np.random.RandomState(42)
    scores = []
    n = len(y_true)

    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        try:
            s = metric_fn(y_true[idx], y_pred[idx])
            if s is not None and not np.isnan(s):
                scores.append(s)
        except Exception:
            continue

    if len(scores) < 10:
        return {'mean': None, 'ci_low': None, 'ci_high': None}

    scores = np.array(scores)
    return {
        'mean': round(float(np.mean(scores)), 4),
        'ci_low': round(float(np.percentile(scores, 100 * alpha / 2)), 4),
        'ci_high': round(float(np.percentile(scores, 100 * (1 - alpha / 2))), 4),
    }


# ════════════════════════════════════════════════════════════════════════
#  5. CALIBRATION ANALYSIS
# ════════════════════════════════════════════════════════════════════════

def binned_calibration(y_true: np.ndarray, y_pred: np.ndarray,
                       n_bins: int = 5) -> List[Dict]:
    """Compute binned calibration statistics."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = []

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge for last bin
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])

        n = mask.sum()
        if n == 0:
            bins.append({
                'bin': f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                'n': 0, 'pred_mean': 0, 'actual_mean': 0, 'error': 0,
            })
        else:
            pred_mean = float(y_pred[mask].mean())
            actual_mean = float(y_true[mask].mean())
            bins.append({
                'bin': f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}",
                'n': int(n),
                'pred_mean': round(pred_mean, 4),
                'actual_mean': round(actual_mean, 4),
                'error': round(abs(pred_mean - actual_mean), 4),
            })

    return bins


# ════════════════════════════════════════════════════════════════════════
#  6. ECONOMIC SIMULATION
# ════════════════════════════════════════════════════════════════════════

def economic_simulation(
    fold_results: List[Dict],
    entry_horizon: int = 10,
    threshold: float = 0.80,
    spread_pips: float = 1.5,
    sl_atr_mult: float = 1.5,
) -> Dict:
    """
    Realistic economic simulation using actual fill data.

    Strategy:
      - Signal: P(fill within `entry_horizon`) >= threshold
      - Entry: limit order at FVG boundary (gap_high for bullish, gap_low for bearish)
      - TP: gap midpoint → variable profit based on actual gap_size
      - SL: 1.5× ATR beyond boundary → variable loss based on volatility
      - Result: uses actual fill_depth and time_to_fill from data

    Also computes a NAIVE benchmark (all FVGs, no model) for comparison.
    """
    def _simulate_trades(test_df, preds, mask_fn, label):
        """Core trade simulation logic, reusable for model and naive."""
        trades = []
        for i in range(len(test_df)):
            if not mask_fn(i):
                continue

            row = test_df.iloc[i]
            gap_size = float(row.get('gap_size', 0))
            atr = float(row.get('atr_14', 0.001))
            filled = bool(row['filled'])
            fill_depth = float(row.get('fill_depth', 0))
            time_to_fill = int(row['time_to_fill'])

            # Skip tiny gaps (< 1 pip)
            if gap_size < 0.0001:
                continue

            # TP target: half the gap (gap midpoint)
            tp_pips_var = (gap_size * 0.5) * 10000  # convert to pips
            # SL: 1.5× ATR beyond entry
            sl_pips_var = (atr * sl_atr_mult) * 10000

            if filled and time_to_fill <= entry_horizon:
                # FVG was touched — but did it fill enough for our TP?
                actual_penetration_pips = (gap_size * fill_depth) * 10000

                if actual_penetration_pips >= tp_pips_var:
                    # Deep fill — TP hit
                    pnl = tp_pips_var - spread_pips
                else:
                    # Shallow fill — price touched boundary but reversed
                    # Partial profit or loss depending on depth
                    # Conservative: assume stopped out unless TP reached
                    pnl = -(sl_pips_var + spread_pips)
            else:
                # Not filled in time — SL hit (position expires worthless)
                pnl = -(sl_pips_var + spread_pips)

            trades.append({
                'fold': int(row.get('_fold', 0)),
                'pred': round(float(preds[i]), 4) if preds is not None else 0,
                'filled': filled and time_to_fill <= entry_horizon,
                'fill_depth': fill_depth,
                'pnl_pips': round(pnl, 2),
                'tp_target': round(tp_pips_var, 2),
                'sl_target': round(sl_pips_var, 2),
            })

        return trades

    def _compute_stats(trades, label):
        if not trades:
            return {'n_trades': 0, 'label': label}

        trades_df = pd.DataFrame(trades)
        cumulative_pnl = trades_df['pnl_pips'].cumsum()
        n_trades = len(trades_df)
        winners = trades_df[trades_df['pnl_pips'] > 0]
        losers = trades_df[trades_df['pnl_pips'] <= 0]

        win_rate = len(winners) / n_trades if n_trades > 0 else 0
        total_pnl = trades_df['pnl_pips'].sum()
        mean_pnl = trades_df['pnl_pips'].mean()
        std_pnl = trades_df['pnl_pips'].std()
        sharpe = (mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 0 else 0

        peak = cumulative_pnl.expanding().max()
        drawdown = cumulative_pnl - peak
        max_dd = float(drawdown.min())

        gross_profit = winners['pnl_pips'].sum() if len(winners) > 0 else 0
        gross_loss = abs(losers['pnl_pips'].sum()) if len(losers) > 0 else 0
        pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Per-fold breakdown
        try:
            per_fold = trades_df.groupby('fold').agg(
                n=('pnl_pips', 'count'),
                pnl=('pnl_pips', 'sum'),
                win_rate=('filled', 'mean'),
            ).reset_index().to_dict('records')
        except Exception:
            per_fold = []

        return {
            'label': label,
            'n_trades': n_trades,
            'win_rate': round(float(win_rate), 4),
            'total_pnl_pips': round(float(total_pnl), 2),
            'mean_pnl_pips': round(float(mean_pnl), 2),
            'median_pnl_pips': round(float(trades_df['pnl_pips'].median()), 2),
            'sharpe_ratio': round(float(sharpe), 4),
            'max_drawdown_pips': round(float(max_dd), 2),
            'profit_factor': round(float(pf), 4),
            'avg_win_pips': round(float(winners['pnl_pips'].mean()), 2) if len(winners) > 0 else 0,
            'avg_loss_pips': round(float(losers['pnl_pips'].mean()), 2) if len(losers) > 0 else 0,
            'entry_threshold': threshold,
            'entry_horizon': entry_horizon,
            'spread_pips': spread_pips,
            'sl_atr_mult': sl_atr_mult,
            'equity_curve': cumulative_pnl.tolist(),
            'per_fold_trades': per_fold,
        }

    # Combine all test data across folds
    all_test = []
    all_preds_h = []
    for result in fold_results:
        test_df = result['test_df'].copy()
        test_df['_fold'] = result['fold']
        preds = result['predictions'].get(entry_horizon)
        if preds is None:
            continue
        all_test.append(test_df)
        all_preds_h.extend(preds.tolist())

    if not all_test:
        return {'model': {'n_trades': 0}, 'naive': {'n_trades': 0}}

    combined_test = pd.concat(all_test, ignore_index=True)
    preds_arr = np.array(all_preds_h)

    # MODEL strategy: only trade when P(fill) >= threshold
    model_trades = _simulate_trades(
        combined_test, preds_arr,
        mask_fn=lambda i: preds_arr[i] >= threshold,
        label='model',
    )
    model_stats = _compute_stats(model_trades, 'Model-Selected')

    # NAIVE strategy: trade ALL FVGs (no model)
    naive_trades = _simulate_trades(
        combined_test, None,
        mask_fn=lambda i: True,
        label='naive',
    )
    naive_stats = _compute_stats(naive_trades, 'Naive (All FVGs)')

    return {
        'model': model_stats,
        'naive': naive_stats,
    }


# ════════════════════════════════════════════════════════════════════════
#  7. HTML REPORT GENERATOR
# ════════════════════════════════════════════════════════════════════════

def generate_html_report(
    fold_results: List[Dict],
    aggregated: Dict,
    bootstrap_results: Dict,
    calibration: Dict,
    economic: Dict,
    audit: Dict,
    folds_meta: List[Dict],
) -> str:
    """Generate self-contained HTML report."""

    # ── Fold summary table ──
    fold_rows = ''
    for r in fold_results:
        aucs = [r['metrics'][h]['auc'] for h in [2, 5, 10, 20] if r['metrics'][h]['auc'] is not None]
        avg_auc = np.mean(aucs) if aucs else 0
        briers = [r['metrics'][h]['brier'] for h in [2, 5, 10, 20] if r['metrics'][h]['brier'] is not None]
        avg_brier = np.mean(briers) if briers else 0
        fold_rows += f"""
        <tr>
            <td>{r['fold']}</td>
            <td>{r['n_train']:,}</td>
            <td>{r['n_test']:,}</td>
            <td>{r['train_end'][:10]}</td>
            <td>{r['test_start'][:10]} → {r['test_end'][:10]}</td>
            <td>{avg_auc:.4f}</td>
            <td>{avg_brier:.4f}</td>
        </tr>"""

    # ── Per-horizon metrics table ──
    horizon_rows = ''
    for h in HORIZONS:
        agg = aggregated.get(h, {})
        bs = bootstrap_results.get(h, {})
        auc_str = f"{agg.get('auc', 'N/A')}"
        if bs.get('auc_ci'):
            auc_str += f" [{bs['auc_ci']['ci_low']}, {bs['auc_ci']['ci_high']}]"
        brier_str = f"{agg.get('brier', 'N/A')}"
        if bs.get('brier_ci'):
            brier_str += f" [{bs['brier_ci']['ci_low']}, {bs['brier_ci']['ci_high']}]"
        cal_err = agg.get('abs_cal_error', 'N/A')
        horizon_rows += f"""
        <tr>
            <td>{h}</td>
            <td>{agg.get('fill_rate', 'N/A')}</td>
            <td>{auc_str}</td>
            <td>{brier_str}</td>
            <td>{agg.get('pred_mean', 'N/A')}</td>
            <td>{agg.get('actual_mean', 'N/A')}</td>
            <td>{cal_err}</td>
        </tr>"""

    # ── Calibration table ──
    cal_rows = ''
    for h in [2, 5, 10, 20, 50]:
        bins = calibration.get(h, [])
        for b in bins:
            cal_rows += f"""
            <tr>
                <td>{h}</td>
                <td>{b['bin']}</td>
                <td>{b['n']}</td>
                <td>{b['pred_mean']}</td>
                <td>{b['actual_mean']}</td>
                <td>{b['error']}</td>
            </tr>"""

    # ── Economic simulation (model vs naive) ──
    model_econ = economic.get('model', {})
    naive_econ = economic.get('naive', {})
    eq_data_model = model_econ.get('equity_curve', [])
    eq_data_naive = naive_econ.get('equity_curve', [])

    def _econ_card(e, title):
        if e.get('n_trades', 0) == 0:
            return f'<div class="metric-card"><h3>{title}</h3><p>No trades.</p></div>'
        return f"""
        <div class="metric-card">
            <h3>{title}</h3>
            <p>Entry: P(fill₁₀) ≥ {e.get('entry_threshold', 'N/A')} | SL: {e.get('sl_atr_mult', 'N/A')}× ATR | Spread: {e.get('spread_pips', 'N/A')} pips</p>
            <table>
                <tr><td>Trades</td><td><strong>{e['n_trades']}</strong></td></tr>
                <tr><td>Win Rate</td><td><strong>{e['win_rate']*100:.1f}%</strong></td></tr>
                <tr><td>Total P&L</td><td><strong>{e['total_pnl_pips']:.1f} pips</strong></td></tr>
                <tr><td>Mean P&L/trade</td><td><strong>{e['mean_pnl_pips']:.2f} pips</strong></td></tr>
                <tr><td>Avg Win</td><td><strong>{e.get('avg_win_pips', 0):.2f} pips</strong></td></tr>
                <tr><td>Avg Loss</td><td><strong>{e.get('avg_loss_pips', 0):.2f} pips</strong></td></tr>
                <tr><td>Sharpe Ratio</td><td><strong>{e['sharpe_ratio']:.4f}</strong></td></tr>
                <tr><td>Max Drawdown</td><td><strong>{e['max_drawdown_pips']:.1f} pips</strong></td></tr>
                <tr><td>Profit Factor</td><td><strong>{e['profit_factor']:.2f}</strong></td></tr>
            </table>
        </div>"""

    econ_summary = _econ_card(model_econ, 'Model-Selected Strategy')

    # ── Feature importance stability ──
    feat_tables = ''
    if fold_results:
        all_feats = {}
        for r in fold_results:
            for feat, imp in r['feature_importance'].items():
                if feat not in all_feats:
                    all_feats[feat] = []
                all_feats[feat].append(imp)

        feat_rows = ''
        sorted_feats = sorted(all_feats.items(), key=lambda x: -np.mean(x[1]))
        for feat, imps in sorted_feats[:15]:
            mean_imp = np.mean(imps)
            std_imp = np.std(imps)
            cv = std_imp / mean_imp if mean_imp > 0 else 0
            feat_rows += f"""
            <tr>
                <td>{feat}</td>
                <td>{mean_imp:.4f}</td>
                <td>±{std_imp:.4f}</td>
                <td>{cv:.2f}</td>
            </tr>"""

        feat_tables = f"""
        <div class="metric-card">
            <h3>Feature Importance Stability (Top 15)</h3>
            <table>
                <tr><th>Feature</th><th>Mean Imp.</th><th>Std</th><th>CV</th></tr>
                {feat_rows}
            </table>
        </div>"""

    # ── Audit result ──
    audit_status = '✅ ALL CHECKS PASSED' if audit['passed'] else '❌ LEAKAGE DETECTED'
    audit_color = '#22c55e' if audit['passed'] else '#ef4444'

    # ── Per-fold trades breakdown ──
    fold_trade_rows = ''
    if model_econ.get('per_fold_trades'):
        for ft in model_econ['per_fold_trades']:
            fold_trade_rows += f"""
            <tr>
                <td>Fold {ft['fold']}</td>
                <td>{ft['n']}</td>
                <td>{ft['pnl']:.1f}</td>
                <td>{ft['win_rate']*100:.1f}%</td>
            </tr>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FVG Fill Probability — Walk-Forward Backtest Report</title>
<style>
    :root {{
        --bg: #0f172a; --card: #1e293b; --border: #334155;
        --text: #e2e8f0; --dim: #94a3b8; --accent: #3b82f6;
        --green: #22c55e; --red: #ef4444; --amber: #f59e0b;
    }}
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background: var(--bg); color: var(--text);
        line-height: 1.6; padding: 2rem;
    }}
    h1 {{ color: #f8fafc; font-size: 1.8rem; margin-bottom: 0.3rem; }}
    h2 {{ color: var(--accent); font-size: 1.3rem; margin: 2rem 0 1rem; border-bottom: 1px solid var(--border); padding-bottom: 0.5rem; }}
    h3 {{ color: #cbd5e1; font-size: 1.1rem; margin-bottom: 0.8rem; }}
    .subtitle {{ color: var(--dim); font-size: 0.9rem; margin-bottom: 2rem; }}
    .metric-card {{
        background: var(--card); border: 1px solid var(--border);
        border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
    }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    th, td {{ padding: 0.5rem 0.8rem; text-align: left; border-bottom: 1px solid var(--border); }}
    th {{ color: var(--dim); font-weight: 600; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.5px; }}
    td {{ color: var(--text); }}
    .badge {{
        display: inline-block; padding: 0.3rem 0.8rem; border-radius: 6px;
        font-weight: 600; font-size: 0.85rem;
    }}
    .badge-pass {{ background: rgba(34,197,94,0.15); color: var(--green); }}
    .badge-fail {{ background: rgba(239,68,68,0.15); color: var(--red); }}
    canvas {{ max-width: 100%; height: 300px; margin-top: 1rem; }}
    .note {{ color: var(--dim); font-size: 0.8rem; font-style: italic; margin-top: 1rem; }}
</style>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
</head>
<body>

<h1>FVG Fill Probability — Walk-Forward Backtest</h1>
<p class="subtitle">
    Multi-Horizon XGBoost | {len(fold_results)} folds |
    Purge gap: {PURGE_CANDLES} candles |
    Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}
</p>

<!-- LEAKAGE AUDIT -->
<div class="metric-card">
    <h3>Leakage Audit</h3>
    <span class="badge {'badge-pass' if audit['passed'] else 'badge-fail'}">{audit_status}</span>
    <p class="note">
        Checks: purge gap ≥ {PURGE_CANDLES}h between train/test, no shared indices, sufficient lookback for features.
    </p>
</div>

<!-- FOLD SUMMARY -->
<h2>Walk-Forward Folds</h2>
<div class="metric-card">
    <table>
        <tr><th>Fold</th><th>Train N</th><th>Test N</th><th>Train End</th><th>Test Period</th><th>Avg AUC</th><th>Avg Brier</th></tr>
        {fold_rows}
    </table>
</div>

<!-- AGGREGATED METRICS -->
<h2>Aggregated OOS Metrics (with 95% Bootstrap CI)</h2>
<div class="metric-card">
    <table>
        <tr><th>Horizon</th><th>Fill Rate</th><th>AUC [95% CI]</th><th>Brier [95% CI]</th><th>Pred Mean</th><th>Actual Mean</th><th>|Cal Error|</th></tr>
        {horizon_rows}
    </table>
</div>

<!-- CALIBRATION -->
<h2>Binned Calibration</h2>
<div class="metric-card">
    <table>
        <tr><th>Horizon</th><th>Bin</th><th>N</th><th>Pred Mean</th><th>Actual Mean</th><th>|Error|</th></tr>
        {cal_rows}
    </table>
</div>

<!-- ECONOMIC SIMULATION -->
<h2>Economic Simulation (Realistic Variable TP/SL)</h2>
<div class="metric-card" style="margin-bottom:1rem;background:rgba(239,68,68,0.08);border-color:#7f1d1d;">
    <p style="color:#fca5a5;">⚠ <strong>Note:</strong> TP = half gap size (variable per FVG). SL = 1.5× ATR (variable per FVG).
    A fill only counts as a WIN if price penetrates ≥50% into the gap (reaching gap midpoint). Shallow fills (boundary touch only) count as losses.</p>
</div>
<div class="grid">
    {econ_summary}
    {_econ_card(naive_econ, 'Naive Benchmark (All FVGs, No Model)')}
</div>
<div class="metric-card">
    <h3>Model Per-Fold Breakdown</h3>
    <table>
        <tr><th>Fold</th><th>Trades</th><th>P&L (pips)</th><th>Win Rate</th></tr>
        {fold_trade_rows}
    </table>
</div>

<!-- EQUITY CURVE -->
{'<div class="metric-card"><h3>Equity Curve — Model vs Naive</h3><canvas id="eqChart"></canvas></div>' if eq_data_model or eq_data_naive else ''}

<!-- FEATURE IMPORTANCE -->
<h2>Feature Importance Stability</h2>
{feat_tables}

<p class="note">
    CV (Coefficient of Variation) measures feature importance stability across folds.
    CV &lt; 0.5 = stable, CV &gt; 1.0 = unstable.
</p>

<script>
const eqModel = {json.dumps(eq_data_model)};
const eqNaive = {json.dumps(eq_data_naive)};
const longer = eqModel.length > eqNaive.length ? eqModel : eqNaive;
if (longer.length > 0) {{
    new Chart(document.getElementById('eqChart'), {{
        type: 'line',
        data: {{
            labels: longer.map((_, i) => i + 1),
            datasets: [
                {{
                    label: 'Model-Selected',
                    data: eqModel,
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59,130,246,0.08)',
                    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2,
                }},
                {{
                    label: 'Naive (All FVGs)',
                    data: eqNaive,
                    borderColor: '#94a3b8',
                    backgroundColor: 'rgba(148,163,184,0.05)',
                    fill: false, tension: 0.3, pointRadius: 0, borderWidth: 1.5,
                    borderDash: [5, 5],
                }}
            ]
        }},
        options: {{
            responsive: true,
            plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
            scales: {{
                x: {{ title: {{ display: true, text: 'Trade #', color: '#94a3b8' }}, ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
                y: {{ title: {{ display: true, text: 'Pips', color: '#94a3b8' }}, ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
            }}
        }}
    }});
}}
</script>

</body>
</html>"""


# ════════════════════════════════════════════════════════════════════════
#  8. MAIN RUNNER
# ════════════════════════════════════════════════════════════════════════

def run_backtest(data_dir: str = 'data/processed',
                 output_dir: str = 'models') -> Dict:
    """Full walk-forward backtest pipeline."""
    from sklearn.metrics import roc_auc_score, brier_score_loss

    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  FVG FILL PROBABILITY — INSTITUTIONAL WALK-FORWARD BACKTEST")
    print("=" * 70)

    # Load combined dataset
    df = pd.read_parquet(os.path.join(data_dir, 'fill_dataset_combined.parquet'))
    df['formation_time'] = pd.to_datetime(df['formation_time'])
    print(f"\nDataset: {len(df):,} samples")
    print(f"Date range: {df['formation_time'].min()} → {df['formation_time'].max()}")

    # Generate folds
    print("\n── Generating walk-forward folds ──")
    folds = generate_walk_forward_folds(df)
    print(f"  Generated {len(folds)} folds")
    for f in folds:
        print(f"  Fold {f['fold']}: train={f['n_train']:,} | "
              f"purge={f['n_purge']} | test={f['n_test']:,} | "
              f"test: {f['test_start'][:10]} → {f['test_end'][:10]}")

    # Leakage audit
    print("\n── Leakage audit ──")
    audit = leakage_audit(df, folds)
    for check in audit['checks']:
        status = '✓' if check['passed'] else '✗'
        print(f"  {status} Fold {check['fold']} | {check['check']}: {check['detail']}")
    print(f"\n  {'✅ ALL PASSED' if audit['passed'] else '❌ LEAKAGE DETECTED'}")

    if not audit['passed']:
        print("\n⚠ ABORTING: Leakage detected. Fix data splits before proceeding.")
        return {}

    # Train and evaluate each fold
    print("\n── Training folds ──")
    fold_results = []
    t0 = time.time()

    for fold in folds:
        print(f"\n  Fold {fold['fold']}:")
        result = train_and_evaluate_fold(df, fold)
        fold_results.append(result)

        for h in [2, 5, 10, 20]:
            m = result['metrics'][h]
            auc_str = f"{m['auc']:.4f}" if m['auc'] is not None else 'N/A'
            print(f"    h={h:3d}: AUC={auc_str}  Brier={m.get('brier', 'N/A')}  "
                  f"pred={m.get('pred_mean', 'N/A')} vs actual={m.get('actual_mean', 'N/A')}")

    total_time = time.time() - t0
    print(f"\n  Total training time: {total_time:.1f}s")

    # Aggregate OOS predictions across all folds
    print("\n── Aggregating OOS metrics ──")
    aggregated = {}
    all_oos_preds = {h: [] for h in HORIZONS}
    all_oos_actuals = {h: [] for h in HORIZONS}

    for result in fold_results:
        for h in HORIZONS:
            if h in result['predictions'] and h in result['actuals']:
                all_oos_preds[h].extend(result['predictions'][h].tolist())
                all_oos_actuals[h].extend(result['actuals'][h].tolist())

    for h in HORIZONS:
        y_true = np.array(all_oos_actuals[h])
        y_pred = np.array(all_oos_preds[h])

        if len(y_true) == 0:
            continue

        try:
            auc = roc_auc_score(y_true, y_pred)
        except ValueError:
            auc = None

        brier = brier_score_loss(y_true, y_pred)

        aggregated[h] = {
            'auc': round(float(auc), 4) if auc is not None else None,
            'brier': round(float(brier), 4),
            'fill_rate': round(float(y_true.mean()), 4),
            'pred_mean': round(float(y_pred.mean()), 4),
            'actual_mean': round(float(y_true.mean()), 4),
            'abs_cal_error': round(abs(float(y_pred.mean()) - float(y_true.mean())), 4),
            'n_samples': len(y_true),
        }

        print(f"  h={h:3d}: AUC={aggregated[h]['auc']}  "
              f"Brier={aggregated[h]['brier']}  "
              f"CalErr={aggregated[h]['abs_cal_error']}  "
              f"N={aggregated[h]['n_samples']}")

    # Bootstrap CIs
    print("\n── Bootstrap confidence intervals (1,000 resamples) ──")
    bootstrap_results = {}
    for h in HORIZONS:
        y_true = np.array(all_oos_actuals[h])
        y_pred = np.array(all_oos_preds[h])
        if len(y_true) < 50:
            continue

        auc_ci = bootstrap_ci(y_true, y_pred, roc_auc_score)
        brier_ci = bootstrap_ci(y_true, y_pred, brier_score_loss)
        bootstrap_results[h] = {'auc_ci': auc_ci, 'brier_ci': brier_ci}

        print(f"  h={h:3d}: AUC={auc_ci['mean']} [{auc_ci['ci_low']}, {auc_ci['ci_high']}]  "
              f"Brier={brier_ci['mean']} [{brier_ci['ci_low']}, {brier_ci['ci_high']}]")

    # Calibration
    print("\n── Calibration analysis ──")
    calibration = {}
    for h in HORIZONS:
        y_true = np.array(all_oos_actuals[h])
        y_pred = np.array(all_oos_preds[h])
        if len(y_true) > 0:
            calibration[h] = binned_calibration(y_true, y_pred)

    # Economic simulation
    print("\n── Economic simulation (realistic variable TP/SL) ──")
    economic = economic_simulation(fold_results)
    for key, label in [('model', 'MODEL-SELECTED'), ('naive', 'NAIVE (ALL FVGs)')]:
        e = economic.get(key, {})
        if e.get('n_trades', 0) > 0:
            print(f"\n  {label}:")
            print(f"    Trades: {e['n_trades']}")
            print(f"    Win Rate: {e['win_rate']*100:.1f}%")
            print(f"    Total P&L: {e['total_pnl_pips']:.1f} pips")
            print(f"    Mean P&L/trade: {e['mean_pnl_pips']:.2f} pips")
            print(f"    Avg win: {e.get('avg_win_pips', 0):.2f} | Avg loss: {e.get('avg_loss_pips', 0):.2f}")
            print(f"    Sharpe: {e['sharpe_ratio']:.4f}")
            print(f"    Max DD: {e['max_drawdown_pips']:.1f} pips")
            print(f"    Profit Factor: {e['profit_factor']:.2f}")
        else:
            print(f"\n  {label}: No trades")

    # Generate report
    print("\n── Generating HTML report ──")
    html = generate_html_report(
        fold_results, aggregated, bootstrap_results,
        calibration, economic, audit,
        [{k: v for k, v in f.items() if k not in ('train_idx', 'test_idx')}
         for f in folds],
    )

    report_path = os.path.join(output_dir, 'backtest_report.html')
    with open(report_path, 'w') as f:
        f.write(html)
    print(f"  Report: {report_path}")

    # Save JSON results
    # Strip equity curves from JSON (too large)
    econ_json = {}
    for strat_key in ['model', 'naive']:
        e = economic.get(strat_key, {})
        econ_json[strat_key] = {k: v for k, v in e.items() if k != 'equity_curve'}

    json_results = {
        'n_folds': len(folds),
        'purge_candles': PURGE_CANDLES,
        'aggregated_metrics': {str(k): v for k, v in aggregated.items()},
        'bootstrap_ci': {str(k): v for k, v in bootstrap_results.items()},
        'economic_simulation': econ_json,
        'leakage_audit_passed': audit['passed'],
        'total_oos_samples': sum(f['n_test'] for f in folds),
        'training_time_seconds': round(total_time, 1),
    }
    json_path = os.path.join(output_dir, 'backtest_results.json')
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    print(f"  JSON: {json_path}")

    print(f"\n{'='*70}")
    print("  BACKTEST COMPLETE")
    print(f"{'='*70}")

    return json_results


if __name__ == '__main__':
    run_backtest()
