import pandas as pd
import numpy as np

def compute_ema(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Exponential Moving Average."""
    if df is None or df.empty or 'close' not in df:
        return pd.Series(dtype=float)
    return df['close'].ewm(span=period, adjust=False).mean()

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Calculate Volume Weighted Average Price."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    
    # Typical price
    tp = (df['high'] + df['low'] + df['close']) / 3
    
    # Cumulative typical price * volume
    cum_tp_vol = (tp * df['volume']).cumsum()
    # Cumulative volume
    cum_vol = df['volume'].cumsum()
    
    # Avoid division by zero
    cum_vol = cum_vol.replace(0, np.nan)
    return cum_tp_vol / cum_vol

def compute_bollinger(df: pd.DataFrame, period: int = 20, std_dev: float = 2.0) -> dict:
    """Calculate Bollinger Bands (Mid, Upper, Lower)."""
    if df is None or df.empty or 'close' not in df:
        return {'mid': pd.Series(dtype=float), 'upper': pd.Series(dtype=float), 'lower': pd.Series(dtype=float)}
        
    mid = df['close'].rolling(window=period).mean()
    std = df['close'].rolling(window=period).std()
    
    upper = mid + (std * std_dev)
    lower = mid - (std * std_dev)
    
    return {'mid': mid, 'upper': upper, 'lower': lower}
