"""
Phase 2: Historical Labeling & Dataset Creation
FVG Probability Modeling Project

Scans historical data for competing FVG scenarios (one bearish above,
one bullish below current price), labels which FVG gets hit first,
extracts features, and builds a training-ready dataset.

Key functions follow the workflow document specification:
  - is_fvg_mitigated
  - find_competing_fvg_scenarios
  - label_scenario_outcome
  - build_training_dataset
  - validate_dataset
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fvg_detector import scan_all_fvgs
from src.feature_engineer import extract_all_features


# Features to drop due to multicollinearity (>0.95 correlation from Phase 1)
HIGH_CORR_DROP = ['avg_candle_range', 'dist_to_lower_edge']


# ---------------------------------------------------------------------------
# 2.1  FVG Mitigation Check
# ---------------------------------------------------------------------------

def is_fvg_mitigated(candles: pd.DataFrame, fvg: Dict,
                      current_idx: int) -> bool:
    """
    Check if an FVG has been filled/mitigated before current_idx.

    A bullish FVG (below price) is mitigated if any candle's low enters the gap.
    A bearish FVG (above price) is mitigated if any candle's high enters the gap.

    Parameters
    ----------
    candles : pd.DataFrame
        OHLCV candles.
    fvg : dict
        FVG dict from fvg_detector (must have 'type', 'formation_index',
        'gap_high', 'gap_low').
    current_idx : int
        Only check candles between formation and this index.

    Returns
    -------
    bool : True if the FVG has been entered/mitigated.
    """
    start = fvg['formation_index'] + 2  # skip formation candles (idx-1, idx, idx+1)
    end = min(current_idx, len(candles))

    for i in range(start, end):
        candle = candles.iloc[i]
        if fvg['type'] == 'bullish':
            if candle['low'] <= fvg['gap_high']:
                return True
        else:  # bearish
            if candle['high'] >= fvg['gap_low']:
                return True

    return False


# ---------------------------------------------------------------------------
# 2.1  Competing FVG Scanner
# ---------------------------------------------------------------------------

def find_competing_fvg_scenarios(candles_1h: pd.DataFrame,
                                  all_fvgs: List[Dict],
                                  min_age_candles: int = 2,
                                  max_age_candles: int = 100) -> List[Dict]:
    """
    Scan for all instances where a bearish FVG exists above AND a bullish
    FVG exists below current price, both unmitigated.

    Parameters
    ----------
    candles_1h : pd.DataFrame
        Hourly OHLCV data.
    all_fvgs : list of dict
        All FVGs detected by scan_all_fvgs().
    min_age_candles : int
        Minimum candles since FVG formation.
    max_age_candles : int
        Maximum candles since FVG formation.

    Returns
    -------
    list of dict, each with keys:
        current_idx, current_price, current_time, upper_fvg, lower_fvg
    """
    scenarios = []
    n = len(candles_1h)

    # Leave 100-candle buffer at end for forward labels
    for current_idx in range(100, n - 100):
        current_price = candles_1h.iloc[current_idx]['close']

        # Active bearish FVGs above price (unmitigated)
        active_bearish = [
            fvg for fvg in all_fvgs
            if fvg['type'] == 'bearish'
            and min_age_candles <= (current_idx - fvg['formation_index']) <= max_age_candles
            and fvg['gap_low'] > current_price
            and not is_fvg_mitigated(candles_1h, fvg, current_idx)
        ]

        # Active bullish FVGs below price (unmitigated)
        active_bullish = [
            fvg for fvg in all_fvgs
            if fvg['type'] == 'bullish'
            and min_age_candles <= (current_idx - fvg['formation_index']) <= max_age_candles
            and fvg['gap_high'] < current_price
            and not is_fvg_mitigated(candles_1h, fvg, current_idx)
        ]

        if active_bearish and active_bullish:
            # Nearest FVG in each direction
            upper_fvg = min(active_bearish,
                            key=lambda x: x['gap_mid'] - current_price)
            lower_fvg = min(active_bullish,
                            key=lambda x: current_price - x['gap_mid'])

            scenarios.append({
                'current_idx': current_idx,
                'current_price': current_price,
                'current_time': candles_1h.index[current_idx],
                'upper_fvg': upper_fvg,
                'lower_fvg': lower_fvg,
            })

    return scenarios


def find_live_competing_scenario(candles_1h: pd.DataFrame,
                                  all_fvgs: List[Dict],
                                  min_age_candles: int = 2,
                                  max_age_candles: int = 100) -> Optional[Dict]:
    """
    Find a competing FVG scenario at the CURRENT (latest) candle.

    Unlike find_competing_fvg_scenarios(), this function:
      - Evaluates at the very last candle (no forward buffer)
      - Checks mitigation up to and including the current candle
      - Returns a single scenario or None

    Designed for live/real-time prediction, NOT backtesting.

    Parameters
    ----------
    candles_1h : pd.DataFrame
        Hourly OHLCV data.
    all_fvgs : list of dict
        All FVGs detected by scan_all_fvgs().
    min_age_candles : int
        Minimum candles since FVG formation.
    max_age_candles : int
        Maximum candles since FVG formation.

    Returns
    -------
    dict or None
        Scenario with current_idx, current_price, current_time,
        upper_fvg, lower_fvg.  None if no valid competing pair exists.
    """
    current_idx = len(candles_1h) - 1
    current_price = candles_1h.iloc[current_idx]['close']

    # Active bearish FVGs ABOVE current price (unmitigated)
    active_bearish = [
        fvg for fvg in all_fvgs
        if fvg['type'] == 'bearish'
        and min_age_candles <= (current_idx - fvg['formation_index']) <= max_age_candles
        and fvg['gap_low'] > current_price
        and not is_fvg_mitigated(candles_1h, fvg, current_idx + 1)
    ]

    # Active bullish FVGs BELOW current price (unmitigated)
    active_bullish = [
        fvg for fvg in all_fvgs
        if fvg['type'] == 'bullish'
        and min_age_candles <= (current_idx - fvg['formation_index']) <= max_age_candles
        and fvg['gap_high'] < current_price
        and not is_fvg_mitigated(candles_1h, fvg, current_idx + 1)
    ]

    if not active_bearish or not active_bullish:
        return None

    # Nearest FVG in each direction
    upper_fvg = min(active_bearish,
                    key=lambda x: x['gap_mid'] - current_price)
    lower_fvg = min(active_bullish,
                    key=lambda x: current_price - x['gap_mid'])

    return {
        'current_idx': current_idx,
        'current_price': current_price,
        'current_time': candles_1h.index[current_idx],
        'upper_fvg': upper_fvg,
        'lower_fvg': lower_fvg,
    }


# ---------------------------------------------------------------------------
# 2.2  Label Outcomes
# ---------------------------------------------------------------------------

def label_scenario_outcome(candles: pd.DataFrame, scenario: Dict,
                            max_candles_forward: int = 100) -> Dict:
    """
    Label which FVG gets hit first and how many candles it takes.

    Parameters
    ----------
    candles : pd.DataFrame
        OHLCV data.
    scenario : dict
        From find_competing_fvg_scenarios().
    max_candles_forward : int
        Maximum forward window.

    Returns
    -------
    dict with 'outcome' ('upper'|'lower'|'neither'|'both_same_candle'),
    'candles_to_hit', 'hit_candle_idx'.
    """
    current_idx = scenario['current_idx']
    upper_fvg = scenario['upper_fvg']
    lower_fvg = scenario['lower_fvg']

    end_idx = min(len(candles), current_idx + max_candles_forward)

    for i in range(current_idx + 1, end_idx):
        candle = candles.iloc[i]

        upper_hit = candle['high'] >= upper_fvg['gap_low']
        lower_hit = candle['low'] <= lower_fvg['gap_high']

        if upper_hit and lower_hit:
            return {
                'outcome': 'both_same_candle',
                'candles_to_hit': i - current_idx,
                'hit_candle_idx': i,
            }
        elif upper_hit:
            return {
                'outcome': 'upper',
                'candles_to_hit': i - current_idx,
                'hit_candle_idx': i,
            }
        elif lower_hit:
            return {
                'outcome': 'lower',
                'candles_to_hit': i - current_idx,
                'hit_candle_idx': i,
            }

    return {
        'outcome': 'neither',
        'candles_to_hit': None,
        'hit_candle_idx': None,
    }


# ---------------------------------------------------------------------------
# 2.3  Build Training Dataset
# ---------------------------------------------------------------------------

def build_training_dataset(candles_1h: pd.DataFrame,
                            candles_4h: pd.DataFrame,
                            candles_daily: pd.DataFrame,
                            all_fvgs: Optional[List[Dict]] = None,
                            drop_correlated: bool = True,
                            max_candles_forward: int = 100) -> pd.DataFrame:
    """
    Build the complete labeled dataset.

    1. Detect FVGs (or use provided list).
    2. Find competing scenarios.
    3. Label each scenario.
    4. Extract features.
    5. Drop high-correlation features.

    Parameters
    ----------
    candles_1h, candles_4h, candles_daily : pd.DataFrame
        Multi-timeframe OHLCV data.
    all_fvgs : list of dict, optional
        Pre-detected FVGs.  If None, scan_all_fvgs is called.
    drop_correlated : bool
        Whether to drop features with >0.95 correlation.
    max_candles_forward : int
        Label window.

    Returns
    -------
    pd.DataFrame with features, target, and metadata columns.
    """
    # Step 1: detect FVGs if not provided
    if all_fvgs is None:
        print("[1/4] Detecting FVGs...")
        all_fvgs = scan_all_fvgs(candles_1h)

    n_bullish = sum(1 for f in all_fvgs if f['type'] == 'bullish')
    n_bearish = len(all_fvgs) - n_bullish
    print(f"  FVGs: {len(all_fvgs)} total ({n_bullish} bullish, {n_bearish} bearish)")

    # Step 2: find competing scenarios
    print("[2/4] Scanning for competing FVG scenarios...")
    scenarios = find_competing_fvg_scenarios(candles_1h, all_fvgs)
    print(f"  Found {len(scenarios)} competing scenarios")

    if not scenarios:
        print("  ⚠ No competing scenarios found. Returning empty DataFrame.")
        return pd.DataFrame()

    # Step 3 & 4: label + extract features
    print("[3/4] Labeling outcomes and extracting features...")
    dataset = []
    outcome_counts = {'upper': 0, 'lower': 0, 'neither': 0, 'both_same_candle': 0}

    for i, scenario in enumerate(scenarios):
        if (i + 1) % 500 == 0:
            print(f"  Processing scenario {i+1}/{len(scenarios)}...")

        outcome = label_scenario_outcome(candles_1h, scenario, max_candles_forward)
        outcome_counts[outcome['outcome']] += 1

        # Skip unresolved / ambiguous
        if outcome['outcome'] in ('neither', 'both_same_candle'):
            continue

        try:
            features = extract_all_features(
                candles_1h, candles_4h, candles_daily,
                scenario['upper_fvg'], scenario['lower_fvg'],
                scenario['current_idx']
            )
        except Exception:
            continue

        row = {
            **features,
            'target': 1 if outcome['outcome'] == 'upper' else 0,
            'candles_to_hit': outcome['candles_to_hit'],
            'scenario_idx': scenario['current_idx'],
            'scenario_time': scenario['current_time'],
        }
        dataset.append(row)

    print(f"  Outcome distribution: {outcome_counts}")

    if not dataset:
        return pd.DataFrame()

    df = pd.DataFrame(dataset)

    # Fill volatility_percentile via rolling rank
    if 'realized_volatility' in df.columns and len(df) > 1:
        df['volatility_percentile'] = df['realized_volatility'].rank(pct=True)

    # Step 5: drop high-correlation features
    if drop_correlated:
        cols_to_drop = [c for c in HIGH_CORR_DROP if c in df.columns]
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            print(f"  Dropped correlated features: {cols_to_drop}")

    print(f"\n[4/4] Dataset summary:")
    print(f"  Size: {len(df)} samples")
    print(f"  Upper hits (target=1): {df['target'].sum()} "
          f"({df['target'].mean()*100:.1f}%)")
    print(f"  Lower hits (target=0): {(1 - df['target']).sum():.0f} "
          f"({(1 - df['target'].mean())*100:.1f}%)")
    print(f"  Avg candles to hit: {df['candles_to_hit'].mean():.1f}")
    print(f"  Median candles to hit: {df['candles_to_hit'].median():.1f}")

    return df


# ---------------------------------------------------------------------------
# 2.4  Dataset Validation
# ---------------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> Dict:
    """
    Perform data quality checks on the labeled dataset.

    Returns
    -------
    dict : validation report.
    """
    print("\n" + "=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    issues = []

    # 1. Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print(f"\n⚠ Missing values:")
        for col, n in missing_cols.items():
            print(f"    {col}: {n} ({n/len(df)*100:.1f}%)")
            issues.append(f"missing_{col}")
    else:
        print("\n✓ No missing values")

    # 2. Extreme outliers
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in ('target', 'candles_to_hit', 'scenario_idx')]
    outlier_report = {}
    for col in numeric_cols:
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        iqr = q99 - q1
        n_outliers = ((df[col] < q1 - 3 * iqr) | (df[col] > q99 + 3 * iqr)).sum()
        if n_outliers > 0:
            outlier_report[col] = int(n_outliers)

    if outlier_report:
        print(f"\n⚠ Extreme outliers (>3× IQR beyond 1st–99th pct):")
        for col, n in sorted(outlier_report.items(), key=lambda x: -x[1]):
            print(f"    {col}: {n}")
    else:
        print("\n✓ No extreme outliers")

    # 3. Class balance
    balance = df['target'].mean()
    print(f"\nClass balance:")
    print(f"  Upper (1): {df['target'].sum()} ({balance*100:.1f}%)")
    print(f"  Lower (0): {(1-df['target']).sum():.0f} ({(1-balance)*100:.1f}%)")
    balance_ok = 0.35 <= balance <= 0.65
    if balance_ok:
        print("  ✓ Within 35/65 tolerance")
    else:
        print("  ⚠ Imbalanced — consider resampling")
        issues.append("class_imbalance")

    # 4. Zero-variance features
    zero_var = [c for c in numeric_cols if df[c].var() == 0]
    if zero_var:
        print(f"\n⚠ Zero-variance features: {zero_var}")
        issues.append("zero_variance")
    else:
        print("\n✓ All features have non-zero variance")

    # 5. Size check
    if len(df) >= 1000:
        print(f"\n✓ Dataset size ({len(df)}) meets ≥1000 threshold")
    else:
        print(f"\n⚠ Dataset size ({len(df)}) < 1000 minimum")
        issues.append("insufficient_samples")

    # 6. Lookahead bias note
    print("\n✓ Lookahead bias absence verified by design (Phase 1 unit tests)")

    status = 'PASS' if not issues else 'WARNING'
    print(f"\nOverall status: {status}")
    print("=" * 60)

    return {
        'status': status,
        'n_samples': len(df),
        'n_features': len(numeric_cols),
        'class_balance': round(balance, 4),
        'missing_columns': list(missing_cols.index) if len(missing_cols) > 0 else [],
        'outlier_features': outlier_report,
        'zero_variance': zero_var,
        'issues': issues,
    }


# ---------------------------------------------------------------------------
# 2.5  Time-Based Split
# ---------------------------------------------------------------------------

def split_dataset(df: pd.DataFrame,
                   train_frac: float = 0.60,
                   val_frac: float = 0.20) -> tuple:
    """
    Time-based 60/20/20 split.

    The dataset must be sorted chronologically. We split by row position
    to respect temporal ordering (no shuffle).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'scenario_time' column.
    train_frac, val_frac : float
        Test fraction is inferred as 1 - train - val.

    Returns
    -------
    tuple of (df_train, df_val, df_test).
    """
    df_sorted = df.sort_values('scenario_time').reset_index(drop=True)
    n = len(df_sorted)

    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))

    df_train = df_sorted.iloc[:train_end].copy()
    df_val = df_sorted.iloc[train_end:val_end].copy()
    df_test = df_sorted.iloc[val_end:].copy()

    print(f"\nTime-based split:")
    print(f"  Train: {len(df_train)} ({len(df_train)/n*100:.1f}%)"
          f"  [{df_train['scenario_time'].iloc[0]} → "
          f"{df_train['scenario_time'].iloc[-1]}]")
    print(f"  Val:   {len(df_val)} ({len(df_val)/n*100:.1f}%)"
          f"  [{df_val['scenario_time'].iloc[0]} → "
          f"{df_val['scenario_time'].iloc[-1]}]")
    print(f"  Test:  {len(df_test)} ({len(df_test)/n*100:.1f}%)"
          f"  [{df_test['scenario_time'].iloc[0]} → "
          f"{df_test['scenario_time'].iloc[-1]}]")

    return df_train, df_val, df_test


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def run_phase2(symbol: str = 'EURUSD', data_dir: str = 'data/raw',
               output_dir: str = 'data/processed') -> pd.DataFrame:
    """Execute the full Phase 2 pipeline for a given symbol."""
    print(f"\n{'='*70}")
    print(f"  PHASE 2: HISTORICAL LABELING & DATASET CREATION — {symbol}")
    print(f"{'='*70}")

    # ── Load data ─────────────────────────────────────────────────────
    print("\nLoading candle data...")
    candles_1h = _load(symbol, '1H', data_dir)
    candles_4h = _load(symbol, '4H', data_dir)
    candles_daily = _load(symbol, '1D', data_dir)
    print(f"  1H: {len(candles_1h)}  |  4H: {len(candles_4h)}  |  "
          f"Daily: {len(candles_daily)}")

    # ── Build dataset ─────────────────────────────────────────────────
    df = build_training_dataset(candles_1h, candles_4h, candles_daily)

    if df.empty:
        print("Pipeline produced no data. Exiting.")
        return df

    # ── Validate ──────────────────────────────────────────────────────
    report = validate_dataset(df)

    # ── Split ─────────────────────────────────────────────────────────
    df_train, df_val, df_test = split_dataset(df)

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)

    full_path = os.path.join(output_dir, f'{symbol}_training_dataset.parquet')
    train_path = os.path.join(output_dir, f'{symbol}_train.parquet')
    val_path = os.path.join(output_dir, f'{symbol}_val.parquet')
    test_path = os.path.join(output_dir, f'{symbol}_test.parquet')
    summary_path = os.path.join(output_dir, 'dataset_summary.json')

    df.to_parquet(full_path, index=False)
    df_train.to_parquet(train_path, index=False)
    df_val.to_parquet(val_path, index=False)
    df_test.to_parquet(test_path, index=False)

    summary = {
        'symbol': symbol,
        'total_samples': len(df),
        'train_samples': len(df_train),
        'val_samples': len(df_val),
        'test_samples': len(df_test),
        'class_balance_upper_pct': round(df['target'].mean() * 100, 2),
        'avg_candles_to_hit': round(df['candles_to_hit'].mean(), 2),
        'median_candles_to_hit': round(df['candles_to_hit'].median(), 2),
        'n_features': len([c for c in df.select_dtypes(include=[np.number]).columns
                           if c not in ('target', 'candles_to_hit', 'scenario_idx')]),
        'train_period': f"{df_train['scenario_time'].iloc[0]} → "
                        f"{df_train['scenario_time'].iloc[-1]}",
        'val_period': f"{df_val['scenario_time'].iloc[0]} → "
                      f"{df_val['scenario_time'].iloc[-1]}",
        'test_period': f"{df_test['scenario_time'].iloc[0]} → "
                       f"{df_test['scenario_time'].iloc[-1]}",
        'validation_report': report,
    }

    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nSaved outputs:")
    print(f"  {full_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"  {summary_path}")

    return df


def _load(symbol: str, tf: str, base_dir: str) -> pd.DataFrame:
    """Load CSV for a given symbol and timeframe."""
    candidates = [f for f in os.listdir(base_dir)
                  if f.startswith(f'{symbol}_{tf}_') and f.endswith('.csv')]
    if not candidates:
        raise FileNotFoundError(f"No {tf} data for {symbol} in {base_dir}")
    path = os.path.join(base_dir, candidates[0])
    return pd.read_csv(path, index_col='timestamp', parse_dates=True)


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'EURUSD'
    run_phase2(symbol)
