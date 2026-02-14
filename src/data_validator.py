"""
Data Validation Pipeline for FVG Probability Modeling
Adapted from workflow to validate OANDA OHLCV data quality
"""

import pandas as pd
from datetime import timedelta
import json


class DataValidator:
    """
    Validate OHLCV data quality.
    
    Checks for:
    - Missing values
    - OHLC consistency (high >= low, etc.)
    - Duplicate timestamps
    - Time gaps (excluding weekends)
    - Statistical outliers
    """
    
    def __init__(self, df, expected_frequency='1H'):
        """
        Parameters:
        -----------
        df : pd.DataFrame, OHLCV data with timestamp index
        expected_frequency : str, '1H', '4H', or '1D'
        """
        self.df = df
        self.expected_frequency = expected_frequency
        self.issues = []
    
    def validate_all(self):
        """Run all validation checks"""
        self.check_missing_values()
        self.check_ohlc_consistency()
        self.check_duplicates()
        self.check_gaps()
        self.check_outliers()
        
        return self.get_report()
    
    def check_missing_values(self):
        """Check for NaN values"""
        missing = self.df.isnull().sum()
        if missing.any():
            self.issues.append({
                'type': 'missing_values',
                'severity': 'high',
                'details': missing[missing > 0].to_dict()
            })
    
    def check_ohlc_consistency(self):
        """Check OHLC relationships (high >= low, etc.)"""
        invalid_high_low = (self.df['high'] < self.df['low']).sum()
        invalid_high_open = (self.df['high'] < self.df['open']).sum()
        invalid_high_close = (self.df['high'] < self.df['close']).sum()
        invalid_low_open = (self.df['low'] > self.df['open']).sum()
        invalid_low_close = (self.df['low'] > self.df['close']).sum()
        
        total_invalid = (invalid_high_low + invalid_high_open + 
                        invalid_high_close + invalid_low_open + invalid_low_close)
        
        if total_invalid > 0:
            self.issues.append({
                'type': 'ohlc_inconsistency',
                'severity': 'critical',
                'details': {
                    'high_below_low': int(invalid_high_low),
                    'high_below_open': int(invalid_high_open),
                    'high_below_close': int(invalid_high_close),
                    'low_above_open': int(invalid_low_open),
                    'low_above_close': int(invalid_low_close),
                }
            })
    
    def check_duplicates(self):
        """Check for duplicate timestamps"""
        duplicates = self.df.index.duplicated().sum()
        if duplicates > 0:
            self.issues.append({
                'type': 'duplicate_timestamps',
                'severity': 'high',
                'count': int(duplicates)
            })
    
    def check_gaps(self):
        """Check for missing candles (gaps in time series)"""
        if self.expected_frequency == '1H':
            expected_delta = timedelta(hours=1)
        elif self.expected_frequency == '4H':
            expected_delta = timedelta(hours=4)
        elif self.expected_frequency == '1D':
            expected_delta = timedelta(days=1)
        else:
            return  # Skip for unsupported frequencies
        
        time_diffs = self.df.index.to_series().diff()
        
        # Allow for weekend gaps in forex (Friday close to Sunday open)
        # ~48-52 hours is normal for hourly data
        gaps = time_diffs[time_diffs > expected_delta * 1.5]
        
        # Filter out weekend gaps (> 48 hours but < 72 hours is likely weekend)
        if self.expected_frequency in ['1H', '4H']:
            non_weekend_gaps = gaps[(gaps < timedelta(hours=48)) | (gaps > timedelta(hours=72))]
        else:
            non_weekend_gaps = gaps[gaps < timedelta(days=7)]  # For daily, allow weekend gaps
        
        if len(non_weekend_gaps) > 0:
            self.issues.append({
                'type': 'time_gaps',
                'severity': 'medium',
                'count': len(non_weekend_gaps),
                'max_gap': str(non_weekend_gaps.max()),
                'examples': [str(x) for x in non_weekend_gaps.head(5).index.tolist()]
            })
    
    def check_outliers(self, z_threshold=5):
        """Check for statistical outliers in returns"""
        returns = self.df['close'].pct_change()
        z_scores = (returns - returns.mean()) / returns.std()
        outliers = (abs(z_scores) > z_threshold).sum()
        
        if outliers > 0:
            self.issues.append({
                'type': 'outliers',
                'severity': 'low',
                'count': int(outliers),
                'threshold': z_threshold,
                'max_return': f"{returns.abs().max():.2%}",
            })
    
    def get_report(self):
        """Generate validation report"""
        if not self.issues:
            return {
                'status': 'PASS',
                'total_candles': len(self.df),
                'date_range': f"{self.df.index[0]} to {self.df.index[-1]}",
                'issues': []
            }
        
        # Categorize by severity
        critical = [i for i in self.issues if i.get('severity') == 'critical']
        high = [i for i in self.issues if i.get('severity') == 'high']
        medium = [i for i in self.issues if i.get('severity') == 'medium']
        low = [i for i in self.issues if i.get('severity') == 'low']
        
        status = 'FAIL' if critical or high else 'WARNING'
        
        return {
            'status': status,
            'total_candles': len(self.df),
            'date_range': f"{self.df.index[0]} to {self.df.index[-1]}",
            'issues': self.issues,
            'summary': {
                'critical': len(critical),
                'high': len(high),
                'medium': len(medium),
                'low': len(low),
            }
        }


def validate_dataset(filepath, expected_frequency='1H'):
    """
    Convenience function to validate a CSV dataset.
    
    Parameters:
    -----------
    filepath : str, path to CSV file
    expected_frequency : str, '1H', '4H', or '1D'
    
    Returns:
    --------
    dict, validation report
    """
    df = pd.read_csv(filepath, index_col='timestamp', parse_dates=True)
    validator = DataValidator(df, expected_frequency)
    report = validator.validate_all()
    
    return report


# Test script
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python data_validator.py <csv_file> [frequency]")
        sys.exit(1)
    
    filepath = sys.argv[1]
    frequency = sys.argv[2] if len(sys.argv) > 2 else '1H'
    
    print(f"Validating {filepath}...")
    report = validate_dataset(filepath, frequency)
    
    print(json.dumps(report, indent=2, default=str))
