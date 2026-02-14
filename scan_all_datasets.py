"""
Batch FVG Scanner
Scans all 12 datasets for FVGs and generates comprehensive statistics

Processes:
- 4 currency pairs (EURUSD, GBPUSD, USDJPY, AUDUSD)
- 3 timeframes (1H, 4H, 1D)
- Generates individual CSV files with FVG data
- Creates summary report with aggregate statistics
"""

import sys
sys.path.append('/Users/kallif/Documents/Dope/Quant/fvg-probability/src')

import pandas as pd
import json
from datetime import datetime
import os
from fvg_detector import scan_all_fvgs, compute_fvg_statistics


def scan_dataset(symbol, timeframe, data_dir='data/raw', output_dir='data/processed'):
    """
    Scan a single dataset for FVGs.
    
    Parameters:
    -----------
    symbol : str, e.g. 'EURUSD'
    timeframe : str, e.g. '1H'
    data_dir : str, directory containing raw data
    output_dir : str, directory to save processed FVG data
    
    Returns:
    --------
    dict, summary statistics for this dataset
    """
    # Find the data file
    import glob
    pattern = f"{data_dir}/{symbol}_{timeframe}_*.csv"
    files = glob.glob(pattern)
    
    if not files:
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'failed',
            'error': 'Data file not found'
        }
    
    filepath = files[0]
    
    try:
        # Load data
        print(f"\nProcessing: {symbol} {timeframe}")
        print(f"  Loading: {filepath}")
        
        df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
        print(f"  Candles: {len(df)}")
        
        # Scan for FVGs
        print(f"  Scanning for FVGs...")
        fvgs = scan_all_fvgs(df)
        print(f"  Found: {len(fvgs)} FVGs")
        
        if len(fvgs) == 0:
            return {
                'symbol': symbol,
                'timeframe': timeframe,
                'status': 'success',
                'total_candles': len(df),
                'total_fvgs': 0,
                'fvg_rate': 0.0
            }
        
        # Compute statistics
        print(f"  Computing statistics...")
        stats_df = compute_fvg_statistics(df, fvgs, max_candles_forward=100)
        
        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/{symbol}_{timeframe}_fvgs.csv"
        stats_df.to_csv(output_file, index=False)
        print(f"  ✓ Saved: {output_file}")
        
        # Return summary
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'success',
            'total_candles': len(df),
            'total_fvgs': len(fvgs),
            'fvg_rate': len(fvgs) / len(df),
            'bullish_fvgs': (stats_df['type'] == 'bullish').sum(),
            'bearish_fvgs': (stats_df['type'] == 'bearish').sum(),
            'touched_rate': stats_df['touched'].mean(),
            'avg_candles_to_touch': stats_df[stats_df['touched']]['candles_to_touch'].mean() if stats_df['touched'].sum() > 0 else None,
            'fully_filled_rate': stats_df['fully_filled'].mean(),
            'avg_gap_size': stats_df['gap_size'].mean(),
            'median_gap_size': stats_df['gap_size'].median(),
            'output_file': output_file
        }
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """
    Scan all 12 datasets and generate summary report.
    """
    print("=" * 80)
    print("PHASE 0: FVG BATCH SCANNER")
    print("=" * 80)
    
    # Configuration
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    timeframes = ['1H', '4H', '1D']
    
    results = []
    
    # Process each dataset
    for symbol in symbols:
        for timeframe in timeframes:
            result = scan_dataset(symbol, timeframe)
            results.append(result)
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"\nDatasets processed: {len(results)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    
    if successful:
        print("\nFVG Detection Summary:")
        total_fvgs = sum(r['total_fvgs'] for r in successful)
        total_candles = sum(r['total_candles'] for r in successful)
        
        print(f"  Total FVGs detected: {total_fvgs:,}")
        print(f"  Total candles: {total_candles:,}")
        print(f"  Overall FVG rate: {total_fvgs/total_candles*100:.2f}%")
        
        # Per timeframe summary
        print("\nBy Timeframe:")
        for tf in timeframes:
            tf_results = [r for r in successful if r['timeframe'] == tf]
            if tf_results:
                tf_fvgs = sum(r['total_fvgs'] for r in tf_results)
                tf_candles = sum(r['total_candles'] for r in tf_results)
                tf_touched = sum(r['touched_rate'] * r['total_fvgs'] for r in tf_results) / tf_fvgs if tf_fvgs > 0 else 0
                
                print(f"  {tf}: {tf_fvgs:,} FVGs ({tf_fvgs/tf_candles*100:.2f}%), {tf_touched*100:.1f}% touched")
        
        # Per symbol summary
        print("\nBy Symbol:")
        for sym in symbols:
            sym_results = [r for r in successful if r['symbol'] == sym]
            if sym_results:
                sym_fvgs = sum(r['total_fvgs'] for r in sym_results)
                sym_candles = sum(r['total_candles'] for r in sym_results)
                
                print(f"  {sym}: {sym_fvgs:,} FVGs ({sym_fvgs/sym_candles*100:.2f}%)")
    
    # Save detailed report
    report = {
        'scan_date': datetime.now().isoformat(),
        'datasets_processed': len(results),
        'successful': len(successful),
        'failed': len(failed),
        'results': results
    }
    
    report_file = 'data/processed/fvg_summary_report.json'
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n✓ Detailed report saved: {report_file}")
    print("\n" + "=" * 80)
    print("PHASE 0 BATCH SCAN COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
