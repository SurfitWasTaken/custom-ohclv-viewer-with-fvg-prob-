"""
FVG (Fair Value Gap) Detection Module
Implements ICT-style Fair Value Gap identification for forex data

Based on workflow Phase 0 specifications
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional


def identify_fvg(candles: pd.DataFrame, index: int) -> Optional[Dict]:
    """
    Identify Fair Value Gap at given candle index.
    
    A Fair Value Gap (FVG) is formed by three consecutive candles where:
    - Bullish FVG: candle_1.high < candle_3.low (gap between them)
    - Bearish FVG: candle_1.low > candle_3.high (gap between them)
    
    The middle candle (candle_2) is the "impulse" candle that creates the gap.
    
    Parameters:
    -----------
    candles : pd.DataFrame with columns [open, high, low, close, volume]
              and datetime index
    index : int, position of middle candle (requires index-1 and index+1)
    
    Returns:
    --------
    dict or None: {
        'type': 'bullish' or 'bearish',
        'gap_high': float,
        'gap_low': float,
        'gap_mid': float,
        'gap_size': float,
        'candle_1': dict,
        'candle_2': dict,
        'candle_3': dict,
        'formation_index': int,
        'formation_time': datetime
    }
    """
    # Boundary check
    if index < 1 or index >= len(candles) - 1:
        return None
    
    c1 = candles.iloc[index - 1]  # First candle
    c2 = candles.iloc[index]      # Middle (impulse) candle
    c3 = candles.iloc[index + 1]  # Third candle
    
    # Bullish FVG: c1.high < c3.low (gap between them)
    # The middle candle's body doesn't overlap with gap
    if c1['high'] < c3['low']:
        gap_low = c1['high']
        gap_high = c3['low']
        
        return {
            'type': 'bullish',
            'gap_high': gap_high,
            'gap_low': gap_low,
            'gap_mid': (gap_high + gap_low) / 2,
            'gap_size': gap_high - gap_low,
            'candle_1': {
                'open': float(c1['open']),
                'high': float(c1['high']),
                'low': float(c1['low']),
                'close': float(c1['close']),
                'volume': int(c1['volume']),
                'time': c1.name
            },
            'candle_2': {
                'open': float(c2['open']),
                'high': float(c2['high']),
                'low': float(c2['low']),
                'close': float(c2['close']),
                'volume': int(c2['volume']),
                'time': c2.name
            },
            'candle_3': {
                'open': float(c3['open']),
                'high': float(c3['high']),
                'low': float(c3['low']),
                'close': float(c3['close']),
                'volume': int(c3['volume']),
                'time': c3.name
            },
            'formation_index': index,
            'formation_time': candles.index[index]
        }
    
    # Bearish FVG: c1.low > c3.high (gap between them)
    elif c1['low'] > c3['high']:
        gap_high = c1['low']
        gap_low = c3['high']
        
        return {
            'type': 'bearish',
            'gap_high': gap_high,
            'gap_low': gap_low,
            'gap_mid': (gap_high + gap_low) / 2,
            'gap_size': gap_high - gap_low,
            'candle_1': {
                'open': float(c1['open']),
                'high': float(c1['high']),
                'low': float(c1['low']),
                'close': float(c1['close']),
                'volume': int(c1['volume']),
                'time': c1.name
            },
            'candle_2': {
                'open': float(c2['open']),
                'high': float(c2['high']),
                'low': float(c2['low']),
                'close': float(c2['close']),
                'volume': int(c2['volume']),
                'time': c2.name
            },
            'candle_3': {
                'open': float(c3['open']),
                'high': float(c3['high']),
                'low': float(c3['low']),
                'close': float(c3['close']),
                'volume': int(c3['volume']),
                'time': c3.name
            },
            'formation_index': index,
            'formation_time': candles.index[index]
        }
    
    return None


def scan_all_fvgs(candles: pd.DataFrame, min_gap_size: Optional[float] = None) -> List[Dict]:
    """
    Scan entire dataset for all FVGs.
    
    Parameters:
    -----------
    candles : pd.DataFrame with OHLCV data
    min_gap_size : float, optional minimum gap size filter (in price units)
    
    Returns:
    --------
    list of dict, all detected FVGs
    """
    fvgs = []
    
    for i in range(1, len(candles) - 1):
        fvg = identify_fvg(candles, i)
        
        if fvg is not None:
            # Apply minimum gap size filter if specified
            if min_gap_size is None or fvg['gap_size'] >= min_gap_size:
                fvgs.append(fvg)
    
    return fvgs


def test_fvg_reversion(candles: pd.DataFrame, fvg: Dict, 
                       max_candles_forward: int = 100) -> Dict:
    """
    Test if price returns to FVG and measure statistics.
    
    IMPORTANT: FVG is formed by 3 candles at indices [idx-1, idx, idx+1].
    We start checking from idx+2 to skip the formation candles.
    
    Parameters:
    -----------
    candles : pd.DataFrame
    fvg : dict, FVG from identify_fvg()
    max_candles_forward : int, how many candles to look ahead
    
    Returns:
    --------
    dict: {
        'touched': bool,
        'candles_to_touch': int or None,
        'touched_level': 'gap_high' | 'gap_low' | 'gap_mid' | None,
        'max_penetration': float,  # how deep into gap
        'fully_filled': bool  # did price cross entire gap
    }
    """
    idx = fvg['formation_index']
    
    # Start from idx+2 to skip all 3 formation candles
    # Formation candles are at: idx-1 (candle 1), idx (candle 2), idx+1 (candle 3)
    start_idx = idx + 2
    end_idx = min(len(candles), idx + max_candles_forward)
    
    for i in range(start_idx, end_idx):
        candle = candles.iloc[i]
        
        # Check if candle touches or enters gap
        if fvg['type'] == 'bullish':
            # Gap is below, price should come down to touch it
            if candle['low'] <= fvg['gap_high']:
                penetration = min(candle['low'], fvg['gap_high']) - fvg['gap_low']
                fully_filled = candle['low'] <= fvg['gap_low']
                
                return {
                    'touched': True,
                    'candles_to_touch': i - idx,
                    'touched_level': 'gap_high',
                    'max_penetration': penetration,
                    'fully_filled': fully_filled
                }
        
        else:  # bearish
            # Gap is above, price should come up to touch it
            if candle['high'] >= fvg['gap_low']:
                penetration = fvg['gap_high'] - max(candle['high'], fvg['gap_low'])
                fully_filled = candle['high'] >= fvg['gap_high']
                
                return {
                    'touched': True,
                    'candles_to_touch': i - idx,
                    'touched_level': 'gap_low',
                    'max_penetration': penetration,
                    'fully_filled': fully_filled
                }
    
    return {
        'touched': False,
        'candles_to_touch': None,
        'touched_level': None,
        'max_penetration': 0,
        'fully_filled': False
    }


def compute_fvg_statistics(candles: pd.DataFrame, fvgs: List[Dict], 
                           max_candles_forward: int = 100) -> pd.DataFrame:
    """
    Aggregate statistics across all FVGs.
    
    Parameters:
    -----------
    candles : pd.DataFrame
    fvgs : list of dict, FVGs from scan_all_fvgs()
    max_candles_forward : int, lookforward window
    
    Returns:
    --------
    pd.DataFrame with FVG data and reversion statistics
    """
    results = []
    
    for fvg in fvgs:
        reversion_stats = test_fvg_reversion(candles, fvg, max_candles_forward)
        
        # Flatten FVG dict for DataFrame
        result = {
            'formation_time': fvg['formation_time'],
            'formation_index': fvg['formation_index'],
            'type': fvg['type'],
            'gap_high': fvg['gap_high'],
            'gap_low': fvg['gap_low'],
            'gap_mid': fvg['gap_mid'],
            'gap_size': fvg['gap_size'],
            **reversion_stats
        }
        
        results.append(result)
    
    df_results = pd.DataFrame(results)
    
    # Print summary statistics
    if len(df_results) > 0:
        print("=" * 60)
        print("FVG REVERSION STATISTICS")
        print("=" * 60)
        print(f"Total FVGs detected: {len(df_results)}")
        print(f"  Bullish: {(df_results['type'] == 'bullish').sum()}")
        print(f"  Bearish: {(df_results['type'] == 'bearish').sum()}")
        print()
        print(f"FVGs touched: {df_results['touched'].sum()} ({df_results['touched'].mean()*100:.1f}%)")
        
        if df_results['touched'].sum() > 0:
            touched = df_results[df_results['touched']]
            print(f"Average candles to touch: {touched['candles_to_touch'].mean():.1f}")
            print(f"Median candles to touch: {touched['candles_to_touch'].median():.1f}")
            print(f"Min candles to touch: {touched['candles_to_touch'].min()}")
            print(f"Max candles to touch: {touched['candles_to_touch'].max()}")
        
        print()
        print(f"FVGs fully filled: {df_results['fully_filled'].sum()} ({df_results['fully_filled'].mean()*100:.1f}%)")
        print()
        print(f"Average gap size: {df_results['gap_size'].mean():.5f}")
        print(f"Median gap size: {df_results['gap_size'].median():.5f}")
        print("=" * 60)
    
    return df_results


# Test script
if __name__ == '__main__':
    import sys
    
    # Test with EURUSD 1H data
    print("Testing FVG detection on EURUSD 1H data...")
    
    df = pd.read_csv('data/raw/EURUSD_1H_20210201_20240201.csv',
                     index_col='timestamp', parse_dates=True)
    
    print(f"Loaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print()
    
    # Scan for FVGs
    print("Scanning for FVGs...")
    fvgs = scan_all_fvgs(df)
    
    print(f"Found {len(fvgs)} FVGs")
    
    if len(fvgs) > 0:
        # Show first few
        print("\nFirst 5 FVGs:")
        for i, fvg in enumerate(fvgs[:5]):
            print(f"{i+1}. {fvg['type'].upper()} FVG at {fvg['formation_time']}")
            print(f"   Gap: {fvg['gap_low']:.5f} - {fvg['gap_high']:.5f} (size: {fvg['gap_size']:.5f})")
        
        # Compute statistics
        print("\nComputing reversion statistics...")
        stats_df = compute_fvg_statistics(df, fvgs)
        
        # Save results
        stats_df.to_csv('data/processed/EURUSD_1H_fvgs.csv', index=False)
        print(f"\nResults saved to data/processed/EURUSD_1H_fvgs.csv")
