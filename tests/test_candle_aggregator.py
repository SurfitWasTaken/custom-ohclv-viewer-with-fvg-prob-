import pytest
import pandas as pd
import numpy as np

from src.candle_aggregator import aggregate_candles

@pytest.fixture
def m1_candles():
    dates = pd.date_range('2024-01-01 00:00', periods=60, freq='1min')
    df = pd.DataFrame({
        'open': np.arange(1, 61),
        'high': np.arange(1, 61) + 2,
        'low': np.arange(1, 61) - 1,
        'close': np.arange(1, 61) + 1,
        'volume': np.full(60, 100)
    }, index=dates)
    return df

def test_aggregate_15m(m1_candles):
    agg = aggregate_candles(m1_candles, interval_minutes=15)
    # 60 mins -> 4 periods
    assert len(agg) == 4
    
    # First candle (indices 0 to 14)
    # values are 1 to 15
    c1 = agg.iloc[0]
    assert c1['open'] == 1
    assert c1['high'] == 15 + 2 # max is at the 15th candle
    assert c1['low'] == 1 - 1   # min is at the 1st candle
    assert c1['close'] == 15 + 1 # close at the 15th candle
    assert c1['volume'] == 1500

def test_weekend_gap():
    # Create a Friday to Monday gap
    friday = pd.date_range('2024-01-05 23:00', periods=60, freq='1min')
    monday = pd.date_range('2024-01-08 00:00', periods=60, freq='1min')
    
    df1 = pd.DataFrame({'open': 1.0, 'high': 1.5, 'low': 0.5, 'close': 1.0, 'volume': 100}, index=friday)
    df2 = pd.DataFrame({'open': 2.0, 'high': 2.5, 'low': 1.5, 'close': 2.0, 'volume': 100}, index=monday)
    df = pd.concat([df1, df2])
    
    # Aggregate to 120m (2h). The total is 2 hours of data but separated by a weekend
    agg = aggregate_candles(df, interval_minutes=120)
    # Because of the break, it should NOT merge them into 1 candle
    assert len(agg) == 2
