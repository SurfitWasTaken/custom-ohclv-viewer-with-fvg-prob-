# Phase 1: Feature Engineering Documentation

## Overview

This module implements the feature extraction pipeline for the FVG Probability Modeling project. Given OHLCV candle data (1H, 4H, Daily) and detected Fair Value Gaps, it computes 32 market-context features across 5 categories.

**Module**: [`src/feature_engineer.py`](src/feature_engineer.py)  
**Analysis**: [`src/feature_analysis.py`](src/feature_analysis.py)  
**Tests**: [`tests/test_feature_engineer.py`](tests/test_feature_engineer.py)

---

## Feature Categories

### 1. Distance Features (7 features)

Measure proximity between current price and competing FVGs.

| Feature | Formula | Typical Range |
|---------|---------|---------------|
| `dist_to_upper_mid` | `upper_fvg.gap_mid − price` | [0, 0.015] |
| `dist_to_lower_mid` | `price − lower_fvg.gap_mid` | [0, 0.022] |
| `dist_to_upper_atr` | `dist_to_upper_mid / ATR` | [0, 9] |
| `dist_to_lower_atr` | `dist_to_lower_mid / ATR` | [0, 9] |
| `distance_ratio` | `dist_upper / dist_lower` | [0, ∞) |
| `dist_to_upper_edge` | `upper_fvg.gap_low − price` | [-0.005, 0.015] |
| `dist_to_lower_edge` | `price − lower_fvg.gap_high` | [-0.004, 0.022] |

### 2. Momentum Features (7 features)

Capture short-term price direction and trend.

| Feature | Formula | Typical Range |
|---------|---------|---------------|
| `return_5` | `close[t] / close[t−5] − 1` | [-0.02, 0.02] |
| `return_10` | `close[t] / close[t−10] − 1` | [-0.02, 0.03] |
| `return_20` | `close[t] / close[t−20] − 1` | [-0.02, 0.03] |
| `is_uptrend_10` | `1 if close[t] > close[t−10]` | {0, 1} |
| `is_uptrend_20` | `1 if close[t] > close[t−20]` | {0, 1} |
| `ma_20` | `mean(close[t−19:t+1])` | [0.96, 1.22] |
| `price_vs_ma20` | `close / MA20 − 1` | [-0.015, 0.016] |

### 3. FVG Characteristics (10 features)

Describe the structural properties of competing FVGs.

| Feature | Formula | Typical Range |
|---------|---------|---------------|
| `upper_fvg_size` | `gap_high − gap_low` | [0, 0.013] |
| `lower_fvg_size` | `gap_high − gap_low` | [0, 0.016] |
| `upper_fvg_size_atr` | `gap_size / ATR` | [0, 9.3] |
| `lower_fvg_size_atr` | `gap_size / ATR` | [0, 7.1] |
| `upper_fvg_age` | `current_idx − formation_idx` | [2, 8685] |
| `lower_fvg_age` | `current_idx − formation_idx` | [2, 5234] |
| `upper_fvg_volume` | Impulse candle tick volume | [100, 35116] |
| `lower_fvg_volume` | Impulse candle tick volume | [167, 34544] |
| `upper_fvg_impulse_size` | `|close − open|` of candle_2 | [0, 0.014] |
| `lower_fvg_impulse_size` | `|close − open|` of candle_2 | [0, 0.017] |

### 4. Volatility & Market Regime (5 features)

Characterize current volatility environment.

| Feature | Formula | Typical Range |
|---------|---------|---------------|
| `atr` | Mean True Range (20-period) | [0.0006, 0.004] |
| `realized_volatility` | Annualized std of returns × √(252×24) | [0.02, 0.34] |
| `volatility_percentile` | Rank percentile of realized_vol | [0, 1] |
| `avg_candle_range` | Mean(high − low) over 20 periods | [0.0006, 0.004] |
| `current_vs_avg_range` | Current range / avg range | [0.07, 7.0] |

### 5. Higher Timeframe Context (3 features)

Capture trend direction on 4H and Daily timeframes.

| Feature | Formula | Typical Range |
|---------|---------|---------------|
| `htf_trend_4h` | `+1` (up), `−1` (down), `0` (insufficient data) | {−1, 0, 1} |
| `htf_trend_daily` | `+1` (up), `−1` (down), `0` (insufficient data) | {−1, 0, 1} |
| `htf_alignment` | `1` if 4H and Daily trends agree | {0, 1} |

---

## Lookahead Bias Prevention

All feature functions are guaranteed lookahead-free:
- Momentum features: only `candles.iloc[:current_idx+1]` is accessed
- Volatility: lookback window `[current_idx − period : current_idx + 1]`
- HTF features: `get_indexer(method='ffill')` maps to the most recent completed bar
- FVG features: `formation_index` is always ≤ `current_idx`

Unit tests verify this by computing features on a truncated dataset and comparing with a longer dataset — results are identical (see `TestNoLookaheadBias`).

---

## Correlation Analysis Results

**EURUSD, 3,434 observations:**

| Pair | Correlation | Action |
|------|-------------|--------|
| `dist_to_lower_mid` ↔ `dist_to_lower_edge` | 0.9608 | Drop `dist_to_lower_edge` |
| `atr` ↔ `avg_candle_range` | 0.9994 | Drop `avg_candle_range` |

**Recommendation**: Drop `avg_candle_range` (redundant with ATR) and `dist_to_lower_edge` in Phase 2 modeling. Retain the ATR-normalized variant for modeling.

---

## Usage

```python
from src.feature_engineer import extract_all_features, build_feature_dataset
from src.fvg_detector import scan_all_fvgs
import pandas as pd

# Load data
candles_1h = pd.read_csv('data/raw/EURUSD_1H_20210201_20240201.csv',
                         index_col='timestamp', parse_dates=True)
candles_4h = pd.read_csv('data/raw/EURUSD_4H_20210201_20240201.csv',
                         index_col='timestamp', parse_dates=True)
candles_daily = pd.read_csv('data/raw/EURUSD_1D_20190201_20240201.csv',
                            index_col='timestamp', parse_dates=True)

# Detect FVGs
fvg_list = scan_all_fvgs(candles_1h)

# Build feature dataset
df_features = build_feature_dataset(candles_1h, candles_4h, candles_daily, fvg_list)
print(df_features.shape)  # (n_observations, 36) — 32 features + 4 metadata cols
```
