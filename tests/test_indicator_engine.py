import pytest
import pandas as pd
import numpy as np

from src.indicator_engine import compute_ema, compute_vwap, compute_bollinger

@pytest.fixture
def sample_df():
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    return pd.DataFrame({
        'open': np.linspace(100, 200, 100),
        'high': np.linspace(105, 205, 100),
        'low': np.linspace(95, 195, 100),
        'close': np.linspace(102, 202, 100),
        'volume': np.random.randint(1000, 5000, 100)
    }, index=dates)

def test_compute_ema(sample_df):
    ema = compute_ema(sample_df, period=20)
    assert len(ema) == len(sample_df)
    assert not ema.isna().all()
    # Test empty
    assert compute_ema(pd.DataFrame()).empty

def test_compute_vwap(sample_df):
    vwap = compute_vwap(sample_df)
    assert len(vwap) == len(sample_df)
    assert not vwap.isna().all()

def test_compute_bollinger(sample_df):
    bb = compute_bollinger(sample_df, period=20)
    assert 'mid' in bb and 'upper' in bb and 'lower' in bb
    assert len(bb['mid']) == len(sample_df)
    # Upper should be >= mid
    valid = bb['mid'].dropna().index
    assert (bb['upper'][valid] >= bb['mid'][valid]).all()
    assert (bb['lower'][valid] <= bb['mid'][valid]).all()
