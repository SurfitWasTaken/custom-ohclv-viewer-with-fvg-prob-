"""
FVG Visualization Module
Creates candlestick charts with FVG zones highlighted for visual validation

Based on workflow Phase 0 specifications
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplfinance as mpf
from typing import Dict, Optional
import os
import random


def visualize_fvg(candles: pd.DataFrame, fvg: Dict, 
                  window_candles: int = 50, 
                  save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot candlestick chart with FVG highlighted.
    
    Parameters:
    -----------
    candles : pd.DataFrame with datetime index and OHLCV columns
    fvg : dict from identify_fvg()
    window_candles : int, number of candles to show around FVG
    save_path : str, optional path to save figure
    
    Returns:
    --------
    matplotlib Figure object
    """
    # Get window around FVG
    idx = fvg['formation_index']
    start_idx = max(0, idx - window_candles//2)
    end_idx = min(len(candles), idx + window_candles//2)
    
    plot_data = candles.iloc[start_idx:end_idx].copy()
    
    # Create custom style
    mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', 
                                wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', 
                            y_on_right=False)
    
    # Add FVG rectangle
    fvg_color = 'blue' if fvg['type'] == 'bullish' else 'red'
    fvg_alpha = 0.2
    
    # Create rectangle for the gap
    fig, axes = mpf.plot(plot_data, type='candle', style=s, 
                         returnfig=True, volume=True,
                         title=f"{fvg['type'].upper()} FVG - {fvg['formation_time']}")
    
    ax = axes[0]
    
    # Add FVG zone
    rect = patches.Rectangle(
        (0, fvg['gap_low']), 
        len(plot_data), 
        fvg['gap_size'],
        linewidth=2, 
        edgecolor=fvg_color, 
        facecolor=fvg_color, 
        alpha=fvg_alpha,
        label=f"{fvg['type'].upper()} FVG"
    )
    ax.add_patch(rect)
    
    # Add midline
    ax.axhline(y=fvg['gap_mid'], color=fvg_color, 
               linestyle='--', linewidth=1, alpha=0.7, 
               label='FVG Midpoint')
    
    # Highlight the three formation candles
    formation_candle_idx = idx - start_idx
    if 0 <= formation_candle_idx < len(plot_data):
        ax.axvline(x=formation_candle_idx-1, color='yellow', 
                   linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=formation_candle_idx, color='yellow', 
                   linestyle='-', linewidth=2, alpha=0.7, 
                   label='Impulse Candle')
        ax.axvline(x=formation_candle_idx+1, color='yellow', 
                   linestyle=':', linewidth=1, alpha=0.5)
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def visualize_competing_fvgs(candles: pd.DataFrame, upper_fvg: Dict, 
                             lower_fvg: Dict, current_price_idx: int,
                             window_candles: int = 100,
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize setup with competing FVGs above and below current price.
    
    Parameters:
    -----------
    candles : pd.DataFrame
    upper_fvg : dict, bearish FVG above price
    lower_fvg : dict, bullish FVG below price
    current_price_idx : int, current candle position
    window_candles : int, total candles to display
    save_path : str, optional path to save figure
    
    Returns:
    --------
    matplotlib Figure object
    """
    start_idx = max(0, current_price_idx - window_candles//2)
    end_idx = min(len(candles), current_price_idx + window_candles)
    
    plot_data = candles.iloc[start_idx:end_idx].copy()
    
    mc = mpf.make_marketcolors(up='g', down='r', edge='inherit', 
                                wick='inherit', volume='in')
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle=':', 
                            y_on_right=False)
    
    fig, axes = mpf.plot(plot_data, type='candle', style=s, 
                         returnfig=True, volume=True,
                         title='Competing FVGs Setup')
    
    ax = axes[0]
    
    # Upper bearish FVG (red)
    rect_upper = patches.Rectangle(
        (0, upper_fvg['gap_low']), 
        len(plot_data), 
        upper_fvg['gap_size'],
        linewidth=2, 
        edgecolor='red', 
        facecolor='red', 
        alpha=0.2,
        label='Bearish FVG (Above)'
    )
    ax.add_patch(rect_upper)
    ax.axhline(y=upper_fvg['gap_mid'], color='red', 
               linestyle='--', linewidth=1, alpha=0.7)
    
    # Lower bullish FVG (blue)
    rect_lower = patches.Rectangle(
        (0, lower_fvg['gap_low']), 
        len(plot_data), 
        lower_fvg['gap_size'],
        linewidth=2, 
        edgecolor='blue', 
        facecolor='blue', 
        alpha=0.2,
        label='Bullish FVG (Below)'
    )
    ax.add_patch(rect_lower)
    ax.axhline(y=lower_fvg['gap_mid'], color='blue', 
               linestyle='--', linewidth=1, alpha=0.7)
    
    # Current price line
    current_price = candles.iloc[current_price_idx]['close']
    current_candle_plot_idx = current_price_idx - start_idx
    ax.axvline(x=current_candle_plot_idx, color='black', 
               linestyle='-', linewidth=2, alpha=0.8, 
               label='Current Price')
    ax.axhline(y=current_price, color='black', 
               linestyle=':', linewidth=1, alpha=0.5)
    
    ax.legend(loc='upper left')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def generate_sample_visualizations(candles: pd.DataFrame, fvgs: list,
                                   num_samples: int = 50,
                                   output_dir: str = 'data/visualizations'):
    """
    Generate sample FVG visualizations for manual review.
    
    Parameters:
    -----------
    candles : pd.DataFrame
    fvgs : list of dict, FVGs from scan_all_fvgs()
    num_samples : int, number of random samples to generate
    output_dir : str, directory to save charts
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Randomly sample FVGs
    if len(fvgs) > num_samples:
        sampled_fvgs = random.sample(fvgs, num_samples)
    else:
        sampled_fvgs = fvgs
    
    print(f"Generating {len(sampled_fvgs)} sample visualizations...")
    
    for i, fvg in enumerate(sampled_fvgs):
        # Create filename
        fvg_type = fvg['type']
        timestamp = fvg['formation_time'].strftime('%Y%m%d_%H%M')
        filename = f"fvg_{i+1:03d}_{fvg_type}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Generate chart
        try:
            fig = visualize_fvg(candles, fvg, window_candles=50, save_path=filepath)
            plt.close(fig)  # Close to free memory
        except Exception as e:
            print(f"Error generating chart {i+1}: {e}")
    
    print(f"✓ Generated {len(sampled_fvgs)} charts in {output_dir}/")


# Test script
if __name__ == '__main__':
    import sys
    sys.path.append('/Users/kallif/Documents/Dope/Quant/fvg-probability/src')
    from fvg_detector import scan_all_fvgs
    
    print("Testing FVG visualization on EURUSD 1H data...")
    
    # Load data
    df = pd.read_csv('data/raw/EURUSD_1H_20210201_20240201.csv',
                     index_col='timestamp', parse_dates=True)
    
    # Scan for FVGs
    print("Scanning for FVGs...")
    fvgs = scan_all_fvgs(df)
    print(f"Found {len(fvgs)} FVGs")
    
    if len(fvgs) > 0:
        # Generate sample visualizations
        print("\nGenerating sample visualizations...")
        generate_sample_visualizations(df, fvgs, num_samples=50)
        
        print("\n✓ Visualization test complete")
        print("Review charts in data/visualizations/ for manual validation")
