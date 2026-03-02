"""
V2 Dataset Builder — Single-FVG Fill-Time Labeling

For each detected FVG, creates a training record with:
  - Time-to-fill (candles until boundary is touched)
  - Censoring indicator (was fill observed?)
  - Fill depth and fully-filled flags
  - Feature vector from fill_feature_engineer

Designed for survival analysis (scikit-survival / lifelines).
"""

import os
import sys
import json
import glob
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fvg_detector import scan_all_fvgs
from src.fill_feature_engineer import extract_fvg_features, FEATURE_COLS


# ── Configuration ───────────────────────────────────────────────────────

MAX_FORWARD_CANDLES = 100   # observation window for each FVG
MIN_FVG_IDX = 50            # skip FVGs too close to start (need lookback)
PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']


# ── Labeling ────────────────────────────────────────────────────────────

def label_fvg_fill(candles: pd.DataFrame, fvg: Dict,
                   max_forward: int = MAX_FORWARD_CANDLES) -> Dict:
    """
    Label a single FVG with time-to-fill and fill characteristics.

    Parameters
    ----------
    candles : pd.DataFrame
        OHLCV data.
    fvg : dict
        FVG from scan_all_fvgs().
    max_forward : int
        Maximum candles to look forward.

    Returns
    -------
    dict with:
        time_to_fill : int   (candles until first touch, or max_forward if censored)
        filled        : bool  (True if fill observed within window)
        fill_depth    : float (0-1, how deep into gap; 0 if not filled)
        fully_filled  : bool  (price crossed entire gap)
    """
    idx = fvg['formation_index']
    start = idx + 2  # skip formation candles
    end = min(len(candles), idx + max_forward + 1)

    gap_high = fvg['gap_high']
    gap_low = fvg['gap_low']
    gap_size = gap_high - gap_low

    for i in range(start, end):
        candle = candles.iloc[i]

        if fvg['type'] == 'bullish':
            # Bullish FVG (below price): filled when low touches gap_high
            if candle['low'] <= gap_high:
                penetration = gap_high - max(candle['low'], gap_low)
                depth = penetration / gap_size if gap_size > 0 else 1.0
                return {
                    'time_to_fill': i - idx,
                    'filled': True,
                    'fill_depth': min(depth, 1.0),
                    'fully_filled': candle['low'] <= gap_low,
                }
        else:  # bearish
            # Bearish FVG (above price): filled when high touches gap_low
            if candle['high'] >= gap_low:
                penetration = min(candle['high'], gap_high) - gap_low
                depth = penetration / gap_size if gap_size > 0 else 1.0
                return {
                    'time_to_fill': i - idx,
                    'filled': True,
                    'fill_depth': min(depth, 1.0),
                    'fully_filled': candle['high'] >= gap_high,
                }

    # Right-censored: not filled within window
    return {
        'time_to_fill': max_forward,
        'filled': False,
        'fill_depth': 0.0,
        'fully_filled': False,
    }


# ── Dataset builder ─────────────────────────────────────────────────────

def build_fill_dataset(candles_1h: pd.DataFrame,
                       candles_4h: Optional[pd.DataFrame] = None,
                       candles_daily: Optional[pd.DataFrame] = None,
                       max_forward: int = MAX_FORWARD_CANDLES) -> pd.DataFrame:
    """
    Build survival-ready dataset for a single pair.

    Parameters
    ----------
    candles_1h : pd.DataFrame
    candles_4h, candles_daily : pd.DataFrame, optional
    max_forward : int

    Returns
    -------
    pd.DataFrame with columns: FEATURE_COLS + [time_to_fill, filled, fill_depth,
                                                fully_filled, fvg_type, formation_time]
    """
    print("  [1/3] Detecting FVGs...")
    all_fvgs = scan_all_fvgs(candles_1h)
    n_bull = sum(1 for f in all_fvgs if f['type'] == 'bullish')
    print(f"    Found {len(all_fvgs)} FVGs ({n_bull} bullish, {len(all_fvgs) - n_bull} bearish)")

    # Filter: skip FVGs too close to edges
    eligible = [f for f in all_fvgs
                if f['formation_index'] >= MIN_FVG_IDX
                and f['formation_index'] < len(candles_1h) - max_forward]
    print(f"    Eligible for labeling: {len(eligible)} (need {MIN_FVG_IDX} lookback + {max_forward} forward)")

    print("  [2/3] Labeling and extracting features...")
    rows = []
    fill_count = 0

    for i, fvg in enumerate(eligible):
        if (i + 1) % 500 == 0:
            print(f"    Processing FVG {i + 1}/{len(eligible)}...")

        # Label
        label = label_fvg_fill(candles_1h, fvg, max_forward)
        if label['filled']:
            fill_count += 1

        # Features
        try:
            feats = extract_fvg_features(
                candles_1h, fvg, fvg['formation_index'],
                candles_4h, candles_daily, all_fvgs)
        except Exception as e:
            continue

        row = {
            **feats,
            'time_to_fill': label['time_to_fill'],
            'filled': label['filled'],
            'fill_depth': label['fill_depth'],
            'fully_filled': label['fully_filled'],
            'fvg_type': fvg['type'],
            'formation_time': fvg['formation_time'],
            'formation_idx': fvg['formation_index'],
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    censor_pct = (1 - fill_count / len(eligible)) * 100 if eligible else 0
    print(f"\n  [3/3] Dataset summary:")
    print(f"    Samples:  {len(df)}")
    print(f"    Filled:   {fill_count} ({100 - censor_pct:.1f}%)")
    print(f"    Censored: {len(df) - fill_count} ({censor_pct:.1f}%)")
    if len(df) > 0 and fill_count > 0:
        filled_df = df[df['filled']]
        print(f"    Median time-to-fill: {filled_df['time_to_fill'].median():.1f} candles")
        print(f"    Mean time-to-fill:   {filled_df['time_to_fill'].mean():.1f} candles")
        print(f"    Fully filled:        {df['fully_filled'].sum()} ({df['fully_filled'].mean()*100:.1f}%)")

    return df


# ── Multi-pair runner ───────────────────────────────────────────────────

def _find_csv(symbol: str, tf: str, data_dir: str) -> Optional[str]:
    """Find CSV file for a given symbol and timeframe."""
    pattern = os.path.join(data_dir, f'{symbol}_{tf}_*.csv')
    files = glob.glob(pattern)
    return files[0] if files else None


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, index_col='timestamp', parse_dates=True)


def run_pipeline(data_dir: str = 'data/raw',
                 output_dir: str = 'data/processed',
                 pairs: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Build fill-time datasets for all pairs and combine.

    Returns the combined DataFrame.
    """
    if pairs is None:
        pairs = PAIRS

    os.makedirs(output_dir, exist_ok=True)
    all_dfs = []

    for pair in pairs:
        print(f"\n{'='*60}")
        print(f"  {pair}")
        print(f"{'='*60}")

        path_1h = _find_csv(pair, '1H', data_dir)
        path_4h = _find_csv(pair, '4H', data_dir)
        path_1d = _find_csv(pair, '1D', data_dir)

        if path_1h is None:
            print(f"  ⚠ No 1H data for {pair}, skipping.")
            continue

        c1h = _load_csv(path_1h)
        c4h = _load_csv(path_4h) if path_4h else None
        c1d = _load_csv(path_1d) if path_1d else None

        print(f"  Data: 1H={len(c1h)}", end="")
        if c4h is not None:
            print(f", 4H={len(c4h)}", end="")
        if c1d is not None:
            print(f", Daily={len(c1d)}", end="")
        print()

        df = build_fill_dataset(c1h, c4h, c1d)
        df['pair'] = pair
        all_dfs.append(df)

        # Save per-pair
        pair_path = os.path.join(output_dir, f'{pair}_fill_dataset.parquet')
        df.to_parquet(pair_path, index=False)
        print(f"  Saved: {pair_path}")

    if not all_dfs:
        print("\n⚠ No data produced.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)

    # Time-based split (60/20/20)
    combined = combined.sort_values('formation_time').reset_index(drop=True)
    n = len(combined)
    t1 = int(n * 0.6)
    t2 = int(n * 0.8)

    train = combined.iloc[:t1]
    val = combined.iloc[t1:t2]
    test = combined.iloc[t2:]

    # Save
    combined.to_parquet(os.path.join(output_dir, 'fill_dataset_combined.parquet'), index=False)
    train.to_parquet(os.path.join(output_dir, 'fill_train.parquet'), index=False)
    val.to_parquet(os.path.join(output_dir, 'fill_val.parquet'), index=False)
    test.to_parquet(os.path.join(output_dir, 'fill_test.parquet'), index=False)

    summary = {
        'total_samples': len(combined),
        'train_samples': len(train),
        'val_samples': len(val),
        'test_samples': len(test),
        'fill_rate_pct': round(combined['filled'].mean() * 100, 2),
        'censored_pct': round((1 - combined['filled'].mean()) * 100, 2),
        'median_ttf': round(combined[combined['filled']]['time_to_fill'].median(), 2),
        'pairs': pairs,
        'n_features': len(FEATURE_COLS),
        'feature_cols': FEATURE_COLS,
    }
    with open(os.path.join(output_dir, 'fill_dataset_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  COMBINED DATASET")
    print(f"{'='*60}")
    print(f"  Total:    {len(combined)}")
    print(f"  Train:    {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"  Fill rate: {combined['filled'].mean()*100:.1f}%")
    print(f"  Censored:  {(1-combined['filled'].mean())*100:.1f}%")
    print(f"  Saved to:  {output_dir}/")

    return combined


if __name__ == '__main__':
    run_pipeline()
