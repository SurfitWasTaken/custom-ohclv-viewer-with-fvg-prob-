"""
Candle Aggregator — Custom Interval OHLCV Aggregation

Aggregates base-granularity candles (e.g., M1) into arbitrary target
intervals (e.g., 12-minute, 3-hour) following standard OHLCV rules:
  Open  = first sub-interval open
  High  = max of all sub-interval highs
  Low   = min of all sub-interval lows
  Close = last sub-interval close
  Volume = sum of all sub-interval volumes

Handles weekend gaps by not merging candles across large time gaps.
"""

import pandas as pd
import numpy as np
from typing import Optional


# Maximum gap (in minutes) between consecutive base candles before we
# consider it a session break (weekend / holiday).  Candles on opposite
# sides of a break are never merged into the same aggregated candle.
_MAX_GAP_MINUTES = 120


def aggregate_candles(df: pd.DataFrame,
                      interval_minutes: int,
                      max_gap_minutes: int = _MAX_GAP_MINUTES) -> pd.DataFrame:
    """
    Aggregate OHLCV candles into a custom interval.

    Parameters
    ----------
    df : pd.DataFrame
        Base candles with DatetimeIndex and columns: open, high, low, close, volume.
    interval_minutes : int
        Target candle size in minutes (e.g. 12 for 12-minute candles).
    max_gap_minutes : int
        If two consecutive base candles are more than this many minutes apart,
        treat it as a session break.  Default 120 (2 hours).

    Returns
    -------
    pd.DataFrame  with the same columns, aggregated to the target interval.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    if interval_minutes < 1:
        raise ValueError("interval_minutes must be >= 1")

    df = df.sort_index()

    # Compute time deltas between consecutive candles
    timestamps = df.index
    deltas = pd.Series(timestamps).diff().dt.total_seconds() / 60.0

    # Assign a session id — increments whenever there is a break
    session_ids = (deltas > max_gap_minutes).cumsum().values

    # Within each session, group candles into buckets of `interval_minutes`
    # using integer division on cumulative count within the session.
    bucket_ids = np.empty(len(df), dtype=int)
    current_session = -1
    counter = 0
    bucket = 0

    for i in range(len(df)):
        if session_ids[i] != current_session:
            current_session = session_ids[i]
            counter = 0
            bucket += 1          # new session always starts a new bucket
        else:
            counter += 1
            if counter % interval_minutes == 0:
                bucket += 1
        bucket_ids[i] = bucket

    df = df.copy()
    df['_bucket'] = bucket_ids

    agg = df.groupby('_bucket').agg(
        timestamp=('open', lambda x: x.index[0]),  # timestamp of first candle
        open=('open', 'first'),
        high=('high', 'max'),
        low=('low', 'min'),
        close=('close', 'last'),
        volume=('volume', 'sum'),
    )
    agg = agg.set_index('timestamp')
    agg.index.name = df.index.name or 'timestamp'

    return agg


# ── Convenience: time-based aggregation (for when base candles are
#    irregularly spaced, e.g. tick data converted to M1 with gaps) ──

def aggregate_candles_time(df: pd.DataFrame,
                           rule: str) -> pd.DataFrame:
    """
    Aggregate using pandas time-based resampling.

    Parameters
    ----------
    df : pd.DataFrame  — OHLCV with DatetimeIndex
    rule : str  — pandas offset alias, e.g. '12min', '2h', '1D'

    Returns
    -------
    pd.DataFrame  — resampled OHLCV
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna(subset=['open'])

    return resampled
