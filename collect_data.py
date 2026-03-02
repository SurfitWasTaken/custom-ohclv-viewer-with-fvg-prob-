"""
Main data collection script for Phase -1
Collects 3 years of historical data for all required instruments and timeframes
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from oanda_collector import OANDADataCollector
from data_validator import DataValidator
from datetime import datetime
import json
import os


def main():
    """
    Execute Phase -1 data collection according to workflow specifications.
    
    Instruments: EURUSD, GBPUSD, USDJPY, AUDUSD
    Timeframes: 1H, 4H, Daily
    History: 3 years (2021-2024)
    """
    
    # Configuration
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    timeframes = ['1H', '4H', '1D']
    
    # Date range: 3 years for hourly/4H, 5 years for daily
    start_date_intraday = datetime(2021, 2, 1)  # Start Feb to avoid holiday gaps
    start_date_daily = datetime(2019, 2, 1)     # 5 years for daily
    end_date = datetime(2024, 2, 1)
    
    output_dir = './data/raw'
    
    print("=" * 80)
    print("PHASE -1: DATA ACQUISITION")
    print("=" * 80)
    print(f"Instruments: {', '.join(symbols)}")
    print(f"Timeframes: {', '.join(timeframes)}")
    print(f"Date Range (Intraday): {start_date_intraday} to {end_date}")
    print(f"Date Range (Daily): {start_date_daily} to {end_date}")
    print(f"Output Directory: {output_dir}")
    print("=" * 80)
    print()
    
    # Initialize collector
    collector = OANDADataCollector(environment="practice")
    
    # Collect data for each symbol and timeframe
    all_results = {}
    validation_reports = {}
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\n{'='*60}")
            print(f"Collecting: {symbol} {timeframe}")
            print(f"{'='*60}")
            
            try:
                # Use appropriate date range
                start = start_date_daily if timeframe == '1D' else start_date_intraday
                
                # Fetch data
                df = collector.get_historical_data(
                    symbol=symbol,
                    timeframe_str=timeframe,
                    start_date=start,
                    end_date=end_date
                )
                
                # Save to CSV
                filename = f"{symbol}_{timeframe}_{start.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath)
                
                print(f"✓ Saved to: {filename}")
                print(f"  Rows: {len(df)}")
                print(f"  Date range: {df.index[0]} to {df.index[-1]}")
                
                # Validate data
                print(f"  Validating data quality...")
                validator = DataValidator(df, expected_frequency=timeframe)
                report = validator.validate_all()
                
                validation_reports[f"{symbol}_{timeframe}"] = report
                
                # Print validation summary
                status_emoji = "✓" if report['status'] == 'PASS' else ("⚠" if report['status'] == 'WARNING' else "✗")
                print(f"  {status_emoji} Validation: {report['status']}")
                
                if report['issues']:
                    print(f"  Issues found:")
                    for issue in report['issues']:
                        print(f"    - {issue['type']} ({issue['severity']})")
                
                all_results[f"{symbol}_{timeframe}"] = {
                    'status': 'success',
                    'rows': len(df),
                    'filepath': filepath,
                    'start': str(df.index[0]),
                    'end': str(df.index[-1]),
                    'validation': report['status']
                }
                
            except Exception as e:
                print(f"✗ Failed: {e}")
                all_results[f"{symbol}_{timeframe}"] = {
                    'status': 'failed',
                    'error': str(e)
                }
    
    # Save collection summary
    print(f"\n{'='*80}")
    print("COLLECTION SUMMARY")
    print(f"{'='*80}")
    
    summary = {
        'collection_date': datetime.now().isoformat(),
        'symbols': symbols,
        'timeframes': timeframes,
        'results': all_results,
        'validation_reports': validation_reports
    }
    
    summary_file = os.path.join(output_dir, 'collection_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Print final statistics
    successful = sum(1 for r in all_results.values() if r['status'] == 'success')
    failed = sum(1 for r in all_results.values() if r['status'] == 'failed')
    
    print(f"\nTotal datasets: {len(all_results)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    
    # Validation summary
    passed = sum(1 for r in validation_reports.values() if r['status'] == 'PASS')
    warnings = sum(1 for r in validation_reports.values() if r['status'] == 'WARNING')
    failed_validation = sum(1 for r in validation_reports.values() if r['status'] == 'FAIL')
    
    print(f"\nValidation Results:")
    print(f"✓ PASS: {passed}")
    print(f"⚠ WARNING: {warnings}")
    print(f"✗ FAIL: {failed_validation}")
    
    print(f"\n{'='*80}")
    print("PHASE -1 COMPLETE")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
