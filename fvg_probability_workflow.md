# Fair Value Gap Probability Modeling: Complete Workflow

**Project Objective**: Develop a quantitative model to predict which competing Fair Value Gap (FVG) will be tested first and estimate the expected number of candles until contact.

**Timeframe**: 8-10 weeks  
**Team**: Quant Engineers  
**Deliverable**: Production-ready probability model for FVG reversion bias

---

## Phase -1: Data Acquisition & Infrastructure (Week 0)

### Objective
Establish comprehensive data collection infrastructure and acquire high-quality historical OHLCV data across multiple instruments and timeframes to ensure model robustness and generalizability.

### Why Multiple Data Sources Matter
- **Robustness**: Model trained on single pair may overfit to that pair's specific characteristics
- **Generalization**: Different currency pairs have different volatility regimes, spreads, and behaviors
- **Validation**: Out-of-sample testing on unseen pairs provides true measure of model performance
- **Production readiness**: Model must work across portfolio of instruments, not just EUR/USD
- **Broker variation**: Even the same pair can have slight differences across brokers (spread, liquidity provider). While not critical for FVG detection, validating across multiple data sources (if available) adds confidence
- **Market regime diversity**: 3-year dataset should capture different market conditions (trending, ranging, volatile, calm)

### Data Requirements Specification

#### 1.1 Instruments (Currency Pairs)

**Primary Pairs** (Major liquidity, start here):
```python
PRIMARY_PAIRS = [
    'EURUSD',  # Most liquid, tightest spreads
    'GBPUSD',  # High volatility, good for testing edge cases
    'USDJPY',  # Different market hours dominance (Asia)
    'AUDUSD',  # Commodity currency, different drivers
]
```

**Secondary Pairs** (Add for robustness after initial validation):
```python
SECONDARY_PAIRS = [
    'USDCHF',  # Safe haven dynamics
    'NZDUSD',  # Smaller market, different liquidity
    'EUGJPY',  # Cross pair, different correlation structure
    'GBPJPY',  # High volatility cross
]
```

**Rationale**: Start with 4 major pairs (diverse enough for generalization, manageable for initial development). Expand to 8+ pairs once pipeline is proven.

#### 1.2 Timeframes

Required timeframes for each instrument:

| Timeframe | Purpose | Min History | Priority |
|-----------|---------|-------------|----------|
| 1 Hour | Primary analysis (FVG detection) | 3 years | Critical |
| 4 Hour | Higher timeframe context | 3 years | Critical |
| 1 Day | Trend/regime identification | 5 years | Critical |
| 15 Min | Optional: finer FVG testing | 1 year | Low |

**Data Volume Estimate**:
- 1H: ~26,280 candles per pair (3 years)
- 4H: ~6,570 candles per pair
- Daily: ~1,825 candles per pair (5 years)
- **Total per pair**: ~34,675 candles across timeframes
- **Total for 4 pairs**: ~138,700 candles

#### 1.3 Data Fields

Required fields for each candle:
```python
REQUIRED_FIELDS = {
    'timestamp': 'datetime64[ns]',  # UTC timezone
    'open': 'float64',
    'high': 'float64',
    'low': 'float64',
    'close': 'float64',
    'volume': 'float64',  # Tick volume or actual volume
}

OPTIONAL_FIELDS = {
    'bid_open': 'float64',    # For spread analysis
    'bid_close': 'float64',
    'ask_open': 'float64',
    'ask_close': 'float64',
    'spread': 'float64',
}
```

#### 1.4 Data Quality Requirements

- **Completeness**: < 0.1% missing candles (gaps must be documented)
- **Accuracy**: Tick-level data preferred, aggregated data acceptable
- **Consistency**: No backward adjustments or splits (forex doesn't have splits, but verify)
- **Timezone**: All data in UTC for consistency
- **No lookahead**: Ensure candle timestamps represent close time, not open time

### Data Source Options

#### Option 1: Free/Low-Cost Sources

**Yahoo Finance** (via yfinance)
```python
import yfinance as yf

# Limitations: 
# - May not have all forex pairs
# - Data quality varies
# - Limited historical depth
# - No guaranteed uptime

ticker = yf.Ticker("EURUSD=X")
data = ticker.history(period="3y", interval="1h")
```

**Pros**: Free, easy to start  
**Cons**: Unreliable for production, limited pairs, gaps in data  
**Verdict**: OK for proof-of-concept only

**Alpha Vantage** (Free tier: 5 calls/min, 500 calls/day)
```python
import requests

# Free API key required
API_KEY = "your_key"
url = f"https://www.alphavantage.co/query?function=FX_INTRADAY&from_symbol=EUR&to_symbol=USD&interval=60min&apikey={API_KEY}"
```

**Pros**: Decent data quality, official API  
**Cons**: Severe rate limits on free tier, limited history (2 years)  
**Verdict**: Better for development, but need paid tier for production

#### Option 2: Broker APIs (Recommended for Development)

**MetaTrader 5** (MT5)
```python
import MetaTrader5 as mt5

# Connect to MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# Get historical data
rates = mt5.copy_rates_range("EURUSD", mt5.TIMEFRAME_H1, 
                              datetime(2021, 1, 1), 
                              datetime(2024, 1, 1))

df = pd.DataFrame(rates)
df['time'] = pd.to_datetime(df['time'], unit='s')
```

**Pros**: 
- Free with broker account (many brokers offer MT5)
- High-quality tick data
- All major and minor pairs
- 10+ years history typically available
- Real-time updates for production

**Cons**: 
- Requires broker account
- Data may vary slightly between brokers
- Weekend gaps (markets closed)

**Verdict**: **RECOMMENDED for initial development**

**Interactive Brokers (IBKR) API**
```python
from ib_insync import IB, Forex

ib = IB()
ib.connect('127.0.0.1', 7497, clientId=1)

contract = Forex('EURUSD')
bars = ib.reqHistoricalData(
    contract,
    endDateTime='',
    durationStr='3 Y',
    barSizeSetting='1 hour',
    whatToShow='MIDPOINT',
    useRTH=False
)
```

**Pros**: Professional-grade data, very reliable  
**Cons**: Requires IBKR account, API complexity  
**Verdict**: Excellent for production, steeper learning curve

#### Option 3: Professional Data Vendors (Production Grade)

**Polygon.io** (Forex data: $199/month)
```python
from polygon import RESTClient

client = RESTClient(api_key="YOUR_API_KEY")

# Get forex data
aggs = client.get_aggs(
    ticker="C:EURUSD",
    multiplier=1,
    timespan="hour",
    from_="2021-01-01",
    to="2024-01-01"
)
```

**Pros**: Clean API, good documentation, reliable  
**Cons**: Paid service  
**Verdict**: Good mid-tier option

**Refinitiv (formerly Reuters)** / **Bloomberg**
- Enterprise-grade, extremely expensive ($20k+/year)
- Only consider for large institutional deployment
- Verdict: Overkill for this project

#### Option 4: Historical Data Downloads

**Dukascopy** (Swiss bank, free historical data)
- Website: https://www.dukascopy.com/swiss/english/marketwatch/historical/
- Format: Tick data, can be aggregated to any timeframe
- Quality: Excellent (institutional grade)
- **Limitation**: Manual download, no API on free tier

```python
# After downloading Dukascopy tick data
# Use library to convert to OHLCV

from dukascopy_data import DukascopyData

dd = DukascopyData()
df = dd.get_data(
    instrument="EURUSD",
    start_date="2021-01-01",
    end_date="2024-01-01",
    timeframe="H1"
)
```

**Verdict**: **RECOMMENDED for historical data** - highest quality free data available

### Recommended Data Strategy

**Phase 1 (Weeks 0-2): Development**
1. **Primary source**: MetaTrader 5 via broker (free, easy setup)
2. **Backup/validation**: Dukascopy historical downloads
3. **Instruments**: Start with EURUSD, GBPUSD, USDJPY, AUDUSD
4. **Timeframes**: 1H, 4H, Daily
5. **History**: 3 years (2021-2024)

**Phase 2 (Production): Production Deployment**
1. **Primary source**: Interactive Brokers API or Polygon.io (paid, reliable)
2. **Instruments**: Expand to 8+ pairs
3. **Real-time updates**: Websocket feeds for live candles

### Data Collection Implementation

#### 3.1 MT5 Data Collector
```python
import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time

class MT5DataCollector:
    """
    Collect historical OHLCV data from MetaTrader 5.
    """
    
    def __init__(self, account=None, password=None, server=None):
        """
        Initialize MT5 connection.
        
        Parameters:
        -----------
        account : int, MT5 account number (optional for demo)
        password : str, MT5 password
        server : str, MT5 server name
        """
        if not mt5.initialize():
            raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
        
        if account and password and server:
            if not mt5.login(account, password, server):
                raise Exception(f"MT5 login failed: {mt5.last_error()}")
        
        print(f"MT5 initialized: {mt5.terminal_info()}")
    
    def get_historical_data(self, symbol, timeframe_str, start_date, end_date):
        """
        Fetch historical OHLCV data.
        
        Parameters:
        -----------
        symbol : str, e.g. 'EURUSD'
        timeframe_str : str, '1H', '4H', '1D'
        start_date : datetime
        end_date : datetime
        
        Returns:
        --------
        pd.DataFrame with OHLCV data
        """
        # Map timeframe strings to MT5 constants
        timeframe_map = {
            '1H': mt5.TIMEFRAME_H1,
            '4H': mt5.TIMEFRAME_H4,
            '1D': mt5.TIMEFRAME_D1,
            '15M': mt5.TIMEFRAME_M15,
        }
        
        timeframe = timeframe_map.get(timeframe_str)
        if not timeframe:
            raise ValueError(f"Unsupported timeframe: {timeframe_str}")
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise ValueError(f"Symbol {symbol} not found")
        
        # Enable symbol if not visible
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                raise ValueError(f"Failed to enable symbol {symbol}")
        
        # Fetch data
        rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            raise ValueError(f"No data returned for {symbol} {timeframe_str}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to standard format
        df = df.rename(columns={
            'tick_volume': 'volume',  # MT5 uses tick volume
        })
        
        # Select and reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.set_index('timestamp')
        
        print(f"Fetched {len(df)} candles for {symbol} {timeframe_str}")
        
        return df
    
    def collect_all_instruments(self, symbols, timeframes, start_date, end_date, 
                                  output_dir='./data/raw'):
        """
        Collect data for multiple symbols and timeframes.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        results = {}
        
        for symbol in symbols:
            for timeframe in timeframes:
                try:
                    df = self.get_historical_data(symbol, timeframe, start_date, end_date)
                    
                    # Save to CSV
                    filename = f"{symbol}_{timeframe}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
                    filepath = os.path.join(output_dir, filename)
                    df.to_csv(filepath)
                    
                    results[f"{symbol}_{timeframe}"] = {
                        'status': 'success',
                        'rows': len(df),
                        'filepath': filepath,
                        'start': df.index[0],
                        'end': df.index[-1],
                    }
                    
                    print(f"✓ Saved {symbol} {timeframe}: {len(df)} candles")
                    
                    # Rate limiting (be nice to broker servers)
                    time.sleep(1)
                    
                except Exception as e:
                    results[f"{symbol}_{timeframe}"] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    print(f"✗ Failed {symbol} {timeframe}: {e}")
        
        return results
    
    def close(self):
        """Shutdown MT5 connection"""
        mt5.shutdown()


# Usage example
if __name__ == '__main__':
    collector = MT5DataCollector()
    
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    timeframes = ['1H', '4H', '1D']
    start = datetime(2021, 1, 1)
    end = datetime(2024, 1, 1)
    
    results = collector.collect_all_instruments(symbols, timeframes, start, end)
    
    # Print summary
    print("\n=== DATA COLLECTION SUMMARY ===")
    for key, info in results.items():
        if info['status'] == 'success':
            print(f"{key}: {info['rows']} rows ({info['start']} to {info['end']})")
        else:
            print(f"{key}: FAILED - {info['error']}")
    
    collector.close()
```

#### 3.2 Data Validation Pipeline
```python
class DataValidator:
    """
    Validate OHLCV data quality.
    """
    
    def __init__(self, df):
        self.df = df
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
                    'high_below_low': invalid_high_low,
                    'high_below_open': invalid_high_open,
                    'high_below_close': invalid_high_close,
                    'low_above_open': invalid_low_open,
                    'low_above_close': invalid_low_close,
                }
            })
    
    def check_duplicates(self):
        """Check for duplicate timestamps"""
        duplicates = self.df.index.duplicated().sum()
        if duplicates > 0:
            self.issues.append({
                'type': 'duplicate_timestamps',
                'severity': 'high',
                'count': duplicates
            })
    
    def check_gaps(self, expected_frequency='1H'):
        """Check for missing candles (gaps in time series)"""
        if expected_frequency == '1H':
            expected_delta = timedelta(hours=1)
        elif expected_frequency == '4H':
            expected_delta = timedelta(hours=4)
        elif expected_frequency == '1D':
            expected_delta = timedelta(days=1)
        else:
            return  # Skip for unsupported frequencies
        
        time_diffs = self.df.index.to_series().diff()
        
        # Allow for weekend gaps in forex (Friday close to Sunday open)
        # ~48-52 hours is normal
        gaps = time_diffs[time_diffs > expected_delta * 1.5]
        
        # Filter out weekend gaps
        non_weekend_gaps = gaps[gaps < timedelta(hours=60)]
        
        if len(non_weekend_gaps) > 0:
            self.issues.append({
                'type': 'time_gaps',
                'severity': 'medium',
                'count': len(non_weekend_gaps),
                'max_gap': str(non_weekend_gaps.max()),
                'examples': non_weekend_gaps.head(5).index.tolist()
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
                'count': outliers,
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
        critical = [i for i in self.issues if i['severity'] == 'critical']
        high = [i for i in self.issues if i['severity'] == 'high']
        medium = [i for i in self.issues if i['severity'] == 'medium']
        low = [i for i in self.issues if i['severity'] == 'low']
        
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


# Usage
df = pd.read_csv('data/raw/EURUSD_1H_20210101_20240101.csv', 
                 index_col='timestamp', parse_dates=True)
validator = DataValidator(df)
report = validator.validate_all()

print(json.dumps(report, indent=2, default=str))
```

#### 3.3 Database Schema

**TimescaleDB** (PostgreSQL extension for time-series data) - RECOMMENDED

```sql
-- Create extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- OHLCV table
CREATE TABLE ohlcv (
    timestamp TIMESTAMPTZ NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    data_source TEXT,  -- 'MT5', 'IBKR', 'Polygon', etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (timestamp, symbol, timeframe)
);

-- Convert to hypertable (TimescaleDB)
SELECT create_hypertable('ohlcv', 'timestamp');

-- Create indexes
CREATE INDEX idx_ohlcv_symbol_timeframe ON ohlcv (symbol, timeframe, timestamp DESC);

-- Metadata table
CREATE TABLE data_quality_logs (
    id SERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    check_timestamp TIMESTAMPTZ DEFAULT NOW(),
    status TEXT,  -- 'PASS', 'WARNING', 'FAIL'
    issues JSONB,
    total_candles INTEGER,
    date_range_start TIMESTAMPTZ,
    date_range_end TIMESTAMPTZ
);
```

**Python interface to database**:
```python
import psycopg2
from sqlalchemy import create_engine
import pandas as pd

class OHLCVDatabase:
    """
    Interface to TimescaleDB for OHLCV storage.
    """
    
    def __init__(self, connection_string):
        """
        connection_string: 'postgresql://user:password@localhost:5432/fvg_data'
        """
        self.engine = create_engine(connection_string)
    
    def insert_candles(self, df, symbol, timeframe, data_source='MT5'):
        """
        Insert OHLCV data into database.
        """
        df_copy = df.copy()
        df_copy['symbol'] = symbol
        df_copy['timeframe'] = timeframe
        df_copy['data_source'] = data_source
        
        df_copy.to_sql('ohlcv', self.engine, if_exists='append', index=True)
        
        print(f"Inserted {len(df_copy)} candles for {symbol} {timeframe}")
    
    def get_candles(self, symbol, timeframe, start_date=None, end_date=None):
        """
        Retrieve OHLCV data from database.
        """
        query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
        """
        
        if start_date:
            query += f" AND timestamp >= '{start_date}'"
        if end_date:
            query += f" AND timestamp <= '{end_date}'"
        
        query += " ORDER BY timestamp ASC"
        
        df = pd.read_sql(query, self.engine, index_col='timestamp', parse_dates=['timestamp'])
        
        return df
    
    def log_data_quality(self, symbol, timeframe, validation_report):
        """
        Log data quality check results.
        """
        import json
        
        query = """
        INSERT INTO data_quality_logs 
        (symbol, timeframe, status, issues, total_candles, date_range_start, date_range_end)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        with self.engine.connect() as conn:
            conn.execute(query, (
                symbol,
                timeframe,
                validation_report['status'],
                json.dumps(validation_report['issues']),
                validation_report['total_candles'],
                validation_report['date_range'].split(' to ')[0],
                validation_report['date_range'].split(' to ')[1],
            ))
```

### Data Versioning & Reproducibility

**Critical for Research Integrity**: Every model iteration must be traceable to exact data versions.

#### Version Control Strategy

```python
import hashlib
import json
from datetime import datetime

class DataVersionManager:
    """
    Track data provenance and enable reproducibility.
    """
    
    def __init__(self, version_file='data_versions.json'):
        self.version_file = version_file
        self.versions = self.load_versions()
    
    def load_versions(self):
        """Load existing version history"""
        try:
            with open(self.version_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}
    
    def save_versions(self):
        """Save version history"""
        with open(self.version_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def compute_hash(self, filepath):
        """Compute SHA256 hash of data file"""
        sha256_hash = hashlib.sha256()
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def register_dataset(self, filepath, symbol, timeframe, metadata=None):
        """
        Register a dataset with version tracking.
        
        Parameters:
        -----------
        filepath : str, path to data file
        symbol : str, e.g. 'EURUSD'
        timeframe : str, e.g. '1H'
        metadata : dict, optional additional info
        """
        file_hash = self.compute_hash(filepath)
        
        # Check if this exact version already exists
        version_key = f"{symbol}_{timeframe}"
        
        if version_key not in self.versions:
            self.versions[version_key] = []
        
        # Check if hash already exists (duplicate)
        existing_hashes = [v['hash'] for v in self.versions[version_key]]
        if file_hash in existing_hashes:
            print(f"Dataset {version_key} with this hash already registered")
            return
        
        # Create version entry
        version_entry = {
            'version': len(self.versions[version_key]) + 1,
            'hash': file_hash,
            'filepath': filepath,
            'registered_at': datetime.now().isoformat(),
            'metadata': metadata or {},
        }
        
        self.versions[version_key].append(version_entry)
        self.save_versions()
        
        print(f"Registered {version_key} v{version_entry['version']}: {file_hash[:8]}...")
    
    def get_latest_version(self, symbol, timeframe):
        """Get most recent version of a dataset"""
        version_key = f"{symbol}_{timeframe}"
        if version_key not in self.versions:
            return None
        return self.versions[version_key][-1]
    
    def get_version(self, symbol, timeframe, version_number):
        """Get specific version of a dataset"""
        version_key = f"{symbol}_{timeframe}"
        if version_key not in self.versions:
            return None
        
        for v in self.versions[version_key]:
            if v['version'] == version_number:
                return v
        return None


# Usage in data collection pipeline
version_manager = DataVersionManager()

# After collecting data
for symbol in symbols:
    for timeframe in timeframes:
        filepath = f"data/raw/{symbol}_{timeframe}_20210101_20240101.csv"
        
        version_manager.register_dataset(
            filepath=filepath,
            symbol=symbol,
            timeframe=timeframe,
            metadata={
                'source': 'MT5',
                'broker': 'XM',
                'date_range': '2021-01-01 to 2024-01-01',
                'collection_date': datetime.now().isoformat(),
            }
        )
```

#### Experiment Tracking with Model-Data Linkage

```python
class ExperimentTracker:
    """
    Link model experiments to exact data versions.
    """
    
    def __init__(self, log_file='experiments.json'):
        self.log_file = log_file
        self.experiments = self.load_experiments()
    
    def load_experiments(self):
        try:
            with open(self.log_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def save_experiments(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.experiments, f, indent=2, default=str)
    
    def log_experiment(self, name, data_versions, model_params, results):
        """
        Log an experiment with full reproducibility info.
        
        Parameters:
        -----------
        name : str, experiment name
        data_versions : dict, {symbol_timeframe: version_hash}
        model_params : dict, hyperparameters
        results : dict, performance metrics
        """
        experiment = {
            'id': len(self.experiments) + 1,
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'data_versions': data_versions,
            'model_params': model_params,
            'results': results,
            'git_commit': self.get_git_commit(),  # Track code version too
        }
        
        self.experiments.append(experiment)
        self.save_experiments()
        
        print(f"Logged experiment #{experiment['id']}: {name}")
    
    def get_git_commit(self):
        """Get current git commit hash"""
        try:
            import subprocess
            result = subprocess.run(['git', 'rev-parse', 'HEAD'], 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        except:
            return None
    
    def reproduce_experiment(self, experiment_id):
        """
        Get all info needed to reproduce an experiment.
        """
        exp = next((e for e in self.experiments if e['id'] == experiment_id), None)
        if not exp:
            print(f"Experiment {experiment_id} not found")
            return None
        
        print(f"=== EXPERIMENT #{experiment_id}: {exp['name']} ===")
        print(f"Timestamp: {exp['timestamp']}")
        print(f"Git Commit: {exp['git_commit']}")
        print(f"\nData Versions:")
        for key, version_hash in exp['data_versions'].items():
            print(f"  {key}: {version_hash}")
        print(f"\nModel Params:")
        print(json.dumps(exp['model_params'], indent=2))
        print(f"\nResults:")
        print(json.dumps(exp['results'], indent=2))
        
        return exp


# Usage in model training
tracker = ExperimentTracker()

# Before training
data_versions = {
    'EURUSD_1H': version_manager.get_latest_version('EURUSD', '1H')['hash'],
    'GBPUSD_1H': version_manager.get_latest_version('GBPUSD', '1H')['hash'],
    # ... etc
}

model_params = {
    'model_type': 'XGBoost',
    'max_depth': 6,
    'learning_rate': 0.05,
    # ... etc
}

# After training
results = {
    'val_roc_auc': 0.687,
    'val_accuracy': 0.623,
    # ... etc
}

tracker.log_experiment(
    name="XGBoost_v1_4pairs",
    data_versions=data_versions,
    model_params=model_params,
    results=results
)
```

#### Backup Strategy

```bash
#!/bin/bash
# backup_data.sh - Run daily via cron

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/fvg_data/$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup raw data
tar -czf $BACKUP_DIR/raw_data.tar.gz data/raw/

# Backup database
pg_dump fvg_data > $BACKUP_DIR/database_dump.sql

# Backup version manifests
cp data_versions.json $BACKUP_DIR/
cp experiments.json $BACKUP_DIR/

# Upload to S3 (or other cloud storage)
# aws s3 sync $BACKUP_DIR s3://fvg-backups/$DATE/

# Keep only last 30 days of local backups
find /backups/fvg_data/ -type d -mtime +30 -exec rm -rf {} +

echo "Backup completed: $BACKUP_DIR"
```

### Deliverables

- [ ] Data requirements document finalized
- [ ] MT5 connection established (or alternative broker API)
- [ ] Data collection scripts tested and running
- [ ] 4 currency pairs × 3 timeframes downloaded (minimum 3 years history)
- [ ] Data validation pipeline implemented
- [ ] TimescaleDB database setup and populated
- [ ] Data quality report generated (all datasets PASS or WARNING status)
- [ ] Data versioning system implemented (version hashes tracked)
- [ ] Experiment tracking system configured
- [ ] Backup strategy implemented (daily exports to S3 or local storage)
- [ ] Documentation: data dictionary, source attribution, known issues

### Success Criteria

- [ ] 100% of target instruments have complete 3-year history (1H, 4H, Daily)
- [ ] All datasets pass OHLC consistency checks (no critical issues)
- [ ] Missing data < 0.5% of expected candles (excluding weekends)
- [ ] Database queries return data in < 100ms for typical ranges
- [ ] Team can independently run data collection pipeline
- [ ] Data provenance documented (know exactly where each dataset came from)

### Common Pitfalls to Avoid

1. **Timezone confusion**: Always use UTC. Forex is 24/5, but different brokers may use different server times.
2. **Weekend gaps**: Don't flag Friday-to-Monday gaps as errors
3. **Daylight saving time**: Can cause duplicate/missing hourly candles in March/November
4. **Broker-specific differences**: EURUSD on one broker might differ slightly from another (spread, liquidity)
5. **Historical data revisions**: Some brokers revise historical data; version your datasets
6. **Tick volume vs real volume**: Forex tick volume != actual traded volume (not available in retail)

### Estimated Costs

| Option | Setup Cost | Monthly Cost | Notes |
|--------|------------|--------------|-------|
| MT5 (broker) | $0 | $0 | Free with broker account (may require minimum deposit) |
| Dukascopy download | $0 | $0 | Manual downloads, one-time |
| Alpha Vantage Free | $0 | $0 | Rate limited |
| Alpha Vantage Premium | $0 | $49.99 | Better for production |
| Polygon.io | $0 | $199 | Professional grade |
| Interactive Brokers | Account minimum | $0 | $0 if maintain minimum balance |
| TimescaleDB (cloud) | $0 | $50-200 | Can self-host for $0 |

**Recommended budget**: $0-50/month for development phase (MT5 + self-hosted DB)

---

## Phase 0: FVG Identification & Validation (Week 1)

### Objective
Establish rigorous FVG detection algorithm and validate identification accuracy through visual confirmation and statistical testing.

### Tasks

#### 0.1 Formalize FVG Definition
Implement precise detection logic for 1-hour timeframe FVGs:

```python
def identify_fvg(candles, index):
    """
    Identify Fair Value Gap at given candle index.
    
    Parameters:
    -----------
    candles : pd.DataFrame with columns [open, high, low, close, volume]
    index : int, position of middle candle (requires index-1 and index+1)
    
    Returns:
    --------
    dict or None: {
        'type': 'bullish' or 'bearish',
        'gap_high': float,
        'gap_low': float,
        'gap_mid': float,
        'candle_1': dict,
        'candle_2': dict,
        'candle_3': dict,
        'formation_time': datetime
    }
    """
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
            'candle_1': c1.to_dict(),
            'candle_2': c2.to_dict(),
            'candle_3': c3.to_dict(),
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
            'candle_1': c1.to_dict(),
            'candle_2': c2.to_dict(),
            'candle_3': c3.to_dict(),
            'formation_index': index,
            'formation_time': candles.index[index]
        }
    
    return None


def scan_all_fvgs(candles):
    """Scan entire dataset for all FVGs"""
    fvgs = []
    for i in range(1, len(candles) - 1):
        fvg = identify_fvg(candles, i)
        if fvg is not None:
            fvgs.append(fvg)
    return fvgs
```

#### 0.2 Visual Validation System
Create visualization tool to confirm FVG detection accuracy:

```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import DateFormatter
import mplfinance as mpf

def visualize_fvg(candles, fvg, window_candles=50):
    """
    Plot candlestick chart with FVG highlighted.
    
    Parameters:
    -----------
    candles : pd.DataFrame with datetime index
    fvg : dict from identify_fvg()
    window_candles : int, number of candles to show around FVG
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
    gap_start_date = plot_data.index[0]
    gap_end_date = plot_data.index[-1]
    
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
    
    return fig


def visualize_competing_fvgs(candles, upper_fvg, lower_fvg, 
                              current_price_idx, window_candles=100):
    """
    Visualize setup with competing FVGs above and below current price.
    
    Parameters:
    -----------
    candles : pd.DataFrame
    upper_fvg : dict, bearish FVG above price
    lower_fvg : dict, bullish FVG below price
    current_price_idx : int, current candle position
    window_candles : int, total candles to display
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
    
    return fig
```

#### 0.3 Statistical Validation
Confirm FVG mean reversion behavior exists:

```python
def test_fvg_reversion(candles, fvg, max_candles_forward=100):
    """
    Test if price returns to FVG and measure statistics.
    
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
    end_idx = min(len(candles), idx + max_candles_forward)
    
    for i in range(idx + 1, end_idx):
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


def compute_fvg_statistics(candles, fvgs):
    """Aggregate statistics across all FVGs"""
    results = []
    
    for fvg in fvgs:
        reversion_stats = test_fvg_reversion(candles, fvg)
        results.append({
            **fvg,
            **reversion_stats
        })
    
    df_results = pd.DataFrame(results)
    
    print("=== FVG REVERSION STATISTICS ===")
    print(f"Total FVGs detected: {len(df_results)}")
    print(f"FVGs touched: {df_results['touched'].sum()} ({df_results['touched'].mean()*100:.1f}%)")
    print(f"Average candles to touch: {df_results[df_results['touched']]['candles_to_touch'].mean():.1f}")
    print(f"Median candles to touch: {df_results[df_results['touched']]['candles_to_touch'].median():.1f}")
    print(f"FVGs fully filled: {df_results['fully_filled'].sum()} ({df_results['fully_filled'].mean()*100:.1f}%)")
    
    return df_results
```

### Deliverables
- [ ] FVG detection algorithm implemented and unit tested
- [ ] Visual validation tool completed
- [ ] Manual review of 50+ random FVG identifications with chart screenshots
- [ ] Statistical report confirming mean reversion (>60% reversion rate expected)
- [ ] Database schema for storing detected FVGs

### Success Criteria
- Detection algorithm produces zero false positives on manual review
- Statistical reversion rate matches prior testing results
- Team consensus on FVG definition and edge cases (e.g., overlapping FVGs, gap size minimums)

---

## Phase 1: Feature Engineering (Week 2)

### Objective
Design and compute features that capture market context around competing FVG scenarios.

### Feature Categories

#### 1.1 Distance Features
```python
def compute_distance_features(current_price, upper_fvg, lower_fvg, atr):
    """
    Compute normalized distance metrics.
    
    Parameters:
    -----------
    current_price : float
    upper_fvg : dict, bearish FVG above
    lower_fvg : dict, bullish FVG below
    atr : float, current Average True Range (14-period)
    """
    return {
        # Absolute distances
        'dist_to_upper_mid': upper_fvg['gap_mid'] - current_price,
        'dist_to_lower_mid': current_price - lower_fvg['gap_mid'],
        
        # ATR-normalized distances
        'dist_to_upper_atr': (upper_fvg['gap_mid'] - current_price) / atr,
        'dist_to_lower_atr': (current_price - lower_fvg['gap_mid']) / atr,
        
        # Distance ratio (asymmetry)
        'distance_ratio': (upper_fvg['gap_mid'] - current_price) / (current_price - lower_fvg['gap_mid']),
        
        # Closest edge distances
        'dist_to_upper_edge': upper_fvg['gap_low'] - current_price,
        'dist_to_lower_edge': current_price - lower_fvg['gap_high'],
    }
```

#### 1.2 Momentum Features
```python
def compute_momentum_features(candles, current_idx):
    """
    Compute multi-timeframe momentum.
    """
    current_price = candles.iloc[current_idx]['close']
    
    return {
        # Short-term momentum
        'return_5': (current_price / candles.iloc[current_idx-5]['close'] - 1) if current_idx >= 5 else 0,
        'return_10': (current_price / candles.iloc[current_idx-10]['close'] - 1) if current_idx >= 10 else 0,
        'return_20': (current_price / candles.iloc[current_idx-20]['close'] - 1) if current_idx >= 20 else 0,
        
        # Momentum direction
        'is_uptrend_10': 1 if current_idx >= 10 and current_price > candles.iloc[current_idx-10]['close'] else 0,
        'is_uptrend_20': 1 if current_idx >= 20 and current_price > candles.iloc[current_idx-20]['close'] else 0,
        
        # Moving average position
        'ma_20': candles.iloc[max(0, current_idx-20):current_idx+1]['close'].mean(),
        'price_vs_ma20': (current_price / candles.iloc[max(0, current_idx-20):current_idx+1]['close'].mean() - 1) if current_idx >= 20 else 0,
    }
```

#### 1.3 FVG Characteristics
```python
def compute_fvg_features(upper_fvg, lower_fvg, current_idx, atr):
    """
    Features describing the FVGs themselves.
    """
    return {
        # FVG sizes
        'upper_fvg_size': upper_fvg['gap_size'],
        'lower_fvg_size': lower_fvg['gap_size'],
        'upper_fvg_size_atr': upper_fvg['gap_size'] / atr,
        'lower_fvg_size_atr': lower_fvg['gap_size'] / atr,
        
        # FVG ages
        'upper_fvg_age': current_idx - upper_fvg['formation_index'],
        'lower_fvg_age': current_idx - lower_fvg['formation_index'],
        
        # FVG strength (volume on formation)
        'upper_fvg_volume': upper_fvg['candle_2']['volume'],
        'lower_fvg_volume': lower_fvg['candle_2']['volume'],
        
        # Impulse strength
        'upper_fvg_impulse_size': abs(upper_fvg['candle_2']['close'] - upper_fvg['candle_2']['open']),
        'lower_fvg_impulse_size': abs(lower_fvg['candle_2']['close'] - lower_fvg['candle_2']['open']),
    }
```

#### 1.4 Volatility & Market Regime
```python
def compute_volatility_features(candles, current_idx, period=20):
    """
    Volatility and market regime indicators.
    """
    if current_idx < period:
        return {}
    
    recent_candles = candles.iloc[current_idx-period:current_idx+1]
    
    # True Range calculation
    high = recent_candles['high']
    low = recent_candles['low']
    close_prev = recent_candles['close'].shift(1)
    
    tr = pd.concat([
        high - low,
        abs(high - close_prev),
        abs(low - close_prev)
    ], axis=1).max(axis=1)
    
    atr = tr.mean()
    
    # Returns volatility
    returns = recent_candles['close'].pct_change().dropna()
    realized_vol = returns.std() * np.sqrt(252 * 24)  # Annualized for hourly
    
    return {
        'atr': atr,
        'realized_volatility': realized_vol,
        'volatility_percentile': None,  # Compute later with historical context
        
        # Range metrics
        'avg_candle_range': (recent_candles['high'] - recent_candles['low']).mean(),
        'current_vs_avg_range': (candles.iloc[current_idx]['high'] - candles.iloc[current_idx]['low']) / (recent_candles['high'] - recent_candles['low']).mean(),
    }
```

#### 1.5 Higher Timeframe Context
```python
def compute_htf_features(candles_4h, candles_daily, current_time):
    """
    Higher timeframe bias (requires 4H and Daily data).
    
    Parameters:
    -----------
    candles_4h : pd.DataFrame, 4-hour candles
    candles_daily : pd.DataFrame, daily candles
    current_time : datetime, current candle timestamp
    """
    # Find corresponding candles
    idx_4h = candles_4h.index.get_indexer([current_time], method='ffill')[0]
    idx_daily = candles_daily.index.get_indexer([current_time], method='ffill')[0]
    
    # 4H trend
    if idx_4h >= 10:
        trend_4h = 1 if candles_4h.iloc[idx_4h]['close'] > candles_4h.iloc[idx_4h-10]['close'] else -1
    else:
        trend_4h = 0
    
    # Daily trend
    if idx_daily >= 5:
        trend_daily = 1 if candles_daily.iloc[idx_daily]['close'] > candles_daily.iloc[idx_daily-5]['close'] else -1
    else:
        trend_daily = 0
    
    return {
        'htf_trend_4h': trend_4h,
        'htf_trend_daily': trend_daily,
        'htf_alignment': 1 if trend_4h == trend_daily else 0,
    }
```

### Feature Pipeline
```python
def extract_all_features(candles_1h, candles_4h, candles_daily, 
                         upper_fvg, lower_fvg, current_idx):
    """
    Master function to extract all features for a competing FVG scenario.
    """
    current_price = candles_1h.iloc[current_idx]['close']
    current_time = candles_1h.index[current_idx]
    
    # Compute ATR first (needed for normalization)
    vol_features = compute_volatility_features(candles_1h, current_idx)
    atr = vol_features['atr']
    
    # Combine all features
    features = {
        **compute_distance_features(current_price, upper_fvg, lower_fvg, atr),
        **compute_momentum_features(candles_1h, current_idx),
        **compute_fvg_features(upper_fvg, lower_fvg, current_idx, atr),
        **vol_features,
        **compute_htf_features(candles_4h, candles_daily, current_time),
    }
    
    return features
```

### Deliverables
- [ ] Feature extraction pipeline implemented
- [ ] Feature correlation analysis (identify and remove redundant features)
- [ ] Feature distribution plots (check for outliers, normalization needs)
- [ ] Feature engineering documentation
- [ ] Unit tests for all feature functions

### Success Criteria
- All features computable without lookahead bias
- Features show reasonable variance across samples
- No features with >0.95 correlation with others (multicollinearity)

---

## Phase 2: Historical Labeling & Dataset Creation (Week 3)

### Objective
Scan historical data to identify all competing FVG scenarios and label outcomes.

### 2.1 Competing FVG Scanner
```python
def find_competing_fvg_scenarios(candles_1h, min_age_candles=2, max_age_candles=100):
    """
    Scan for all instances where:
    - A bearish FVG exists above current price
    - A bullish FVG exists below current price
    - Both FVGs are unmitigated (not yet filled)
    
    Returns list of scenario dictionaries.
    """
    # Step 1: Detect all FVGs
    all_fvgs = scan_all_fvgs(candles_1h)
    
    scenarios = []
    
    # Step 2: For each candle, check if competing FVGs exist
    for current_idx in range(100, len(candles_1h) - 100):  # Leave buffer for forward-looking labels
        current_price = candles_1h.iloc[current_idx]['close']
        
        # Find active FVGs (formed before current candle, not too old)
        active_bearish_fvgs = [
            fvg for fvg in all_fvgs
            if fvg['type'] == 'bearish'
            and fvg['formation_index'] < current_idx
            and (current_idx - fvg['formation_index']) <= max_age_candles
            and fvg['gap_low'] > current_price  # FVG is above price
            # Check if still unmitigated
            and not is_fvg_mitigated(candles_1h, fvg, current_idx)
        ]
        
        active_bullish_fvgs = [
            fvg for fvg in all_fvgs
            if fvg['type'] == 'bullish'
            and fvg['formation_index'] < current_idx
            and (current_idx - fvg['formation_index']) <= max_age_candles
            and fvg['gap_high'] < current_price  # FVG is below price
            and not is_fvg_mitigated(candles_1h, fvg, current_idx)
        ]
        
        # If we have at least one of each, we have a competing scenario
        if active_bearish_fvgs and active_bullish_fvgs:
            # Take the nearest FVG in each direction
            upper_fvg = min(active_bearish_fvgs, key=lambda x: x['gap_mid'] - current_price)
            lower_fvg = max(active_bullish_fvgs, key=lambda x: current_price - x['gap_mid'])
            
            scenarios.append({
                'current_idx': current_idx,
                'current_price': current_price,
                'current_time': candles_1h.index[current_idx],
                'upper_fvg': upper_fvg,
                'lower_fvg': lower_fvg,
            })
    
    return scenarios


def is_fvg_mitigated(candles, fvg, current_idx):
    """
    Check if FVG has been filled/mitigated before current_idx.
    """
    for i in range(fvg['formation_index'] + 1, current_idx):
        candle = candles.iloc[i]
        
        if fvg['type'] == 'bullish':
            # FVG is filled if price enters the gap
            if candle['low'] <= fvg['gap_high']:
                return True
        else:  # bearish
            if candle['high'] >= fvg['gap_low']:
                return True
    
    return False
```

### 2.2 Label Outcomes
```python
def label_scenario_outcome(candles, scenario, max_candles_forward=100):
    """
    Label which FVG gets hit first and how many candles it takes.
    
    Returns:
    --------
    dict: {
        'outcome': 'upper' | 'lower' | 'neither' | 'both_same_candle',
        'candles_to_hit': int or None,
        'hit_candle_idx': int or None,
    }
    """
    current_idx = scenario['current_idx']
    upper_fvg = scenario['upper_fvg']
    lower_fvg = scenario['lower_fvg']
    
    end_idx = min(len(candles), current_idx + max_candles_forward)
    
    for i in range(current_idx + 1, end_idx):
        candle = candles.iloc[i]
        
        # Check if upper (bearish) FVG is hit
        upper_hit = candle['high'] >= upper_fvg['gap_low']
        
        # Check if lower (bullish) FVG is hit
        lower_hit = candle['low'] <= lower_fvg['gap_high']
        
        if upper_hit and lower_hit:
            # Both hit in same candle (rare but possible in high volatility)
            return {
                'outcome': 'both_same_candle',
                'candles_to_hit': i - current_idx,
                'hit_candle_idx': i,
            }
        elif upper_hit:
            return {
                'outcome': 'upper',
                'candles_to_hit': i - current_idx,
                'hit_candle_idx': i,
            }
        elif lower_hit:
            return {
                'outcome': 'lower',
                'candles_to_hit': i - current_idx,
                'hit_candle_idx': i,
            }
    
    # Neither hit within the window
    return {
        'outcome': 'neither',
        'candles_to_hit': None,
        'hit_candle_idx': None,
    }
```

### 2.3 Create Training Dataset
```python
def build_training_dataset(candles_1h, candles_4h, candles_daily):
    """
    Build complete labeled dataset.
    """
    # Find all competing scenarios
    scenarios = find_competing_fvg_scenarios(candles_1h)
    
    print(f"Found {len(scenarios)} competing FVG scenarios")
    
    dataset = []
    
    for scenario in scenarios:
        # Label outcome
        outcome = label_scenario_outcome(candles_1h, scenario)
        
        # Skip scenarios that don't resolve
        if outcome['outcome'] == 'neither':
            continue
        
        # Skip ambiguous cases for now
        if outcome['outcome'] == 'both_same_candle':
            continue
        
        # Extract features
        features = extract_all_features(
            candles_1h, candles_4h, candles_daily,
            scenario['upper_fvg'],
            scenario['lower_fvg'],
            scenario['current_idx']
        )
        
        # Combine everything
        row = {
            **features,
            'target': 1 if outcome['outcome'] == 'upper' else 0,  # 1 = upper, 0 = lower
            'candles_to_hit': outcome['candles_to_hit'],
            'scenario_idx': scenario['current_idx'],
            'scenario_time': scenario['current_time'],
        }
        
        dataset.append(row)
    
    df = pd.DataFrame(dataset)
    
    print(f"Dataset size: {len(df)} samples")
    print(f"Upper hits: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
    print(f"Lower hits: {(1-df['target']).sum()} ({(1-df['target']).mean()*100:.1f}%)")
    print(f"Avg candles to hit: {df['candles_to_hit'].mean():.1f}")
    
    return df
```

### 2.4 Data Quality Checks
```python
def validate_dataset(df):
    """
    Perform data quality checks.
    """
    print("=== DATASET VALIDATION ===")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("WARNING: Missing values detected:")
        print(missing[missing > 0])
    
    # Check feature ranges
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ['target', 'candles_to_hit', 'scenario_idx']:
            continue
        
        q1 = df[col].quantile(0.01)
        q99 = df[col].quantile(0.99)
        
        # Flag extreme outliers
        outliers = df[(df[col] < q1 - 3*(q99-q1)) | (df[col] > q99 + 3*(q99-q1))].shape[0]
        if outliers > 0:
            print(f"{col}: {outliers} extreme outliers detected")
    
    # Check for lookahead bias
    # Ensure no features use information from after the scenario timestamp
    print("\nManual check required: Review feature extraction for lookahead bias")
    
    # Check class balance
    print(f"\nClass balance: {df['target'].value_counts(normalize=True)}")
    
    return True
```

### Deliverables
- [ ] Scenario scanner implemented and tested
- [ ] Complete labeled dataset (minimum 1000 samples, ideally 5000+)
- [ ] Data validation report
- [ ] Train/validation/test split (60/20/20 with time-based splits)
- [ ] Dataset saved to database and parquet files

### Success Criteria
- At least 1000 labeled scenarios (more is better)
- No lookahead bias confirmed through code review
- Class balance not more skewed than 65/35 (if skewed, consider balanced sampling)
- All features populated without missing values

---

## Phase 3: Model Development (Weeks 4-5)

### Objective
Train and evaluate predictive models for FVG competition outcomes.

### 3.1 Baseline Models

#### Distance Ratio Model (Simplest)
```python
def distance_ratio_baseline(df):
    """
    Baseline: Predict based purely on distance ratio.
    Closer FVG should have higher probability.
    """
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
    
    # Predict upper if distance to upper < distance to lower
    df['pred_baseline'] = (df['dist_to_upper_atr'] < df['dist_to_lower_atr']).astype(int)
    
    # Probability: simple logistic function of log distance ratio
    df['prob_upper_baseline'] = 1 / (1 + np.exp(-2 * np.log(df['dist_to_lower_atr'] / df['dist_to_upper_atr'])))
    
    print("=== BASELINE MODEL (Distance Ratio) ===")
    print(f"Accuracy: {accuracy_score(df['target'], df['pred_baseline']):.3f}")
    print(f"ROC-AUC: {roc_auc_score(df['target'], df['prob_upper_baseline']):.3f}")
    print(f"Log Loss: {log_loss(df['target'], df['prob_upper_baseline']):.3f}")
    
    return df
```

#### Logistic Regression
```python
def train_logistic_regression(df_train, df_val):
    """
    Logistic regression with regularization.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report, roc_auc_score, log_loss
    
    # Select features
    feature_cols = [col for col in df_train.columns 
                    if col not in ['target', 'candles_to_hit', 'scenario_idx', 'scenario_time']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target']
    X_val = df_val[feature_cols]
    y_val = df_val['target']
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train with L2 regularization
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred_val = model.predict(X_val_scaled)
    y_prob_val = model.predict_proba(X_val_scaled)[:, 1]
    
    # Evaluate
    print("=== LOGISTIC REGRESSION ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob_val):.3f}")
    print(f"Log Loss: {log_loss(y_val, y_prob_val):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val, target_names=['Lower', 'Upper']))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', key=abs, ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10))
    
    return model, scaler, feature_importance
```

### 3.2 Advanced Models

#### Gradient Boosting
```python
def train_gradient_boosting(df_train, df_val):
    """
    Gradient Boosting (XGBoost or LightGBM).
    """
    import xgboost as xgb
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
    
    feature_cols = [col for col in df_train.columns 
                    if col not in ['target', 'candles_to_hit', 'scenario_idx', 'scenario_time']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target']
    X_val = df_val[feature_cols]
    y_val = df_val['target']
    
    # XGBoost parameters
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.05,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'eval_metric': 'logloss'
    }
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=20,
        verbose=False
    )
    
    # Predictions
    y_prob_val = model.predict_proba(X_val)[:, 1]
    y_pred_val = (y_prob_val > 0.5).astype(int)
    
    # Evaluate
    print("=== GRADIENT BOOSTING (XGBoost) ===")
    print(f"Accuracy: {accuracy_score(y_val, y_pred_val):.3f}")
    print(f"ROC-AUC: {roc_auc_score(y_val, y_prob_val):.3f}")
    print(f"Log Loss: {log_loss(y_val, y_prob_val):.3f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(feature_importance.head(10))
    
    return model, feature_importance
```

#### Survival Analysis (Time-to-Event)
```python
def train_survival_model(df_train, df_val):
    """
    Cox Proportional Hazards model to predict time until each FVG is hit.
    This gives us probability over time, not just binary outcome.
    """
    from lifelines import CoxPHFitter
    
    # We need to reshape data: each scenario gets TWO rows (one for each FVG)
    train_rows = []
    
    for idx, row in df_train.iterrows():
        # Upper FVG row
        train_rows.append({
            **{k: v for k, v in row.items() if k in feature_cols},
            'event': 1 if row['target'] == 1 else 0,  # 1 if upper was hit
            'duration': row['candles_to_hit'] if row['target'] == 1 else 100,  # censored if not hit
            'fvg_type': 'upper'
        })
        
        # Lower FVG row
        train_rows.append({
            **{k: v for k, v in row.items() if k in feature_cols},
            'event': 1 if row['target'] == 0 else 0,  # 1 if lower was hit
            'duration': row['candles_to_hit'] if row['target'] == 0 else 100,
            'fvg_type': 'lower'
        })
    
    df_survival = pd.DataFrame(train_rows)
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(df_survival, duration_col='duration', event_col='event')
    
    print("=== COX SURVIVAL MODEL ===")
    print(cph.summary[['coef', 'exp(coef)', 'p']])
    
    return cph
```

### 3.3 Model Calibration
```python
def calibrate_probabilities(y_true, y_prob):
    """
    Calibrate probability predictions using isotonic regression.
    """
    from sklearn.calibration import calibration_curve, CalibratedClassifierCV
    import matplotlib.pyplot as plt
    
    # Plot calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect Calibration')
    plt.xlabel('Predicted Probability')
    plt.ylabel('True Probability')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return plt
```

### 3.4 Model Selection
```python
def compare_models(results_dict):
    """
    Compare all models on validation set.
    
    Parameters:
    -----------
    results_dict : dict of {model_name: (y_pred, y_prob)}
    """
    from sklearn.metrics import accuracy_score, roc_auc_score, log_loss, brier_score_loss
    
    comparison = []
    
    for model_name, (y_true, y_pred, y_prob) in results_dict.items():
        comparison.append({
            'Model': model_name,
            'Accuracy': accuracy_score(y_true, y_pred),
            'ROC-AUC': roc_auc_score(y_true, y_prob),
            'Log Loss': log_loss(y_true, y_prob),
            'Brier Score': brier_score_loss(y_true, y_prob),
        })
    
    df_comparison = pd.DataFrame(comparison).sort_values('ROC-AUC', ascending=False)
    
    print("=== MODEL COMPARISON ===")
    print(df_comparison.to_string(index=False))
    
    return df_comparison
```

### Deliverables
- [ ] All baseline and advanced models trained
- [ ] Model comparison report with metrics
- [ ] Calibration curves for top 2 models
- [ ] Feature importance analysis
- [ ] Selected production model with hyperparameters documented
- [ ] Model artifacts saved (pickle/joblib files)

### Success Criteria
- Best model achieves ROC-AUC > 0.65 (substantial improvement over random)
- Probability calibration shows reasonable fit (within 0.05 of diagonal)
- Feature importance aligns with intuition (distance, momentum should be top features)

---

## Phase 4: Backtesting & Validation (Week 6)

### Objective
Validate model performance in realistic trading simulation.

### 4.1 Walk-Forward Validation
```python
def walk_forward_test(candles_1h, candles_4h, candles_daily, 
                      model, scaler, feature_cols,
                      train_window_days=365, test_window_days=90):
    """
    Time-series walk-forward validation.
    """
    results = []
    
    # Split data into consecutive windows
    total_candles = len(candles_1h)
    train_window_candles = train_window_days * 24  # 1h candles
    test_window_candles = test_window_days * 24
    
    start_idx = train_window_candles
    
    while start_idx + test_window_candles < total_candles:
        # Training period
        train_start = start_idx - train_window_candles
        train_end = start_idx
        
        # Test period
        test_start = start_idx
        test_end = start_idx + test_window_candles
        
        # Build datasets for this window
        # [Implementation similar to Phase 2]
        
        # Train model on this window
        # [Retrain with same hyperparameters]
        
        # Test on forward period
        # [Collect predictions]
        
        results.append({
            'test_period_start': candles_1h.index[test_start],
            'test_period_end': candles_1h.index[test_end],
            'accuracy': accuracy,
            'roc_auc': roc_auc,
        })
        
        # Slide window forward
        start_idx += test_window_candles
    
    df_results = pd.DataFrame(results)
    
    print("=== WALK-FORWARD VALIDATION ===")
    print(f"Number of test windows: {len(df_results)}")
    print(f"Average Accuracy: {df_results['accuracy'].mean():.3f} ± {df_results['accuracy'].std():.3f}")
    print(f"Average ROC-AUC: {df_results['roc_auc'].mean():.3f} ± {df_results['roc_auc'].std():.3f}")
    
    return df_results
```

### 4.2 Cross-Pair Validation (Critical for Robustness)

**Objective**: Ensure model generalizes across different currency pairs, not just the training pair.

**Why This Matters**: A model that only works on EUR/USD has overfit to that pair's specific characteristics (ECB policy cycles, European trading hours, specific volatility regime). True robustness means performing on GBP/USD, USD/JPY, etc.

```python
def cross_pair_validation(all_pairs_data, model_class, feature_cols):
    """
    Leave-one-pair-out cross-validation.
    
    Train on 3 pairs, test on the 4th, repeat for all pairs.
    """
    pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
    results = []
    
    for test_pair in pairs:
        print(f"\n=== Testing on {test_pair} (held out) ===")
        
        # Train on all OTHER pairs
        train_pairs = [p for p in pairs if p != test_pair]
        
        # Combine training data from multiple pairs
        train_data = pd.concat([
            all_pairs_data[pair] for pair in train_pairs
        ], ignore_index=True)
        
        test_data = all_pairs_data[test_pair]
        
        # Train model
        model = train_model(train_data, feature_cols)
        
        # Evaluate on held-out pair
        X_test = test_data[feature_cols]
        y_test = test_data['target']
        
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob > 0.5).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'test_pair': test_pair,
            'train_pairs': train_pairs,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'n_samples': len(test_data),
        })
        
        print(f"Accuracy: {accuracy:.3f}")
        print(f"ROC-AUC: {roc_auc:.3f}")
    
    df_results = pd.DataFrame(results)
    
    print("\n=== CROSS-PAIR VALIDATION SUMMARY ===")
    print(df_results[['test_pair', 'accuracy', 'roc_auc']])
    print(f"\nMean Accuracy: {df_results['accuracy'].mean():.3f} ± {df_results['accuracy'].std():.3f}")
    print(f"Mean ROC-AUC: {df_results['roc_auc'].mean():.3f} ± {df_results['roc_auc'].std():.3f}")
    
    # Flag if any pair performs significantly worse
    min_auc = df_results['roc_auc'].min()
    if min_auc < 0.55:
        print(f"\n⚠️  WARNING: Worst pair ({df_results.loc[df_results['roc_auc'].idxmin(), 'test_pair']}) has ROC-AUC of {min_auc:.3f}")
        print("Model may not generalize well. Consider:")
        print("  - Adding pair-specific features (volatility regime, typical spread)")
        print("  - Training separate models per pair")
        print("  - Collecting more data for underperforming pair")
    
    return df_results


def pair_specific_analysis(all_pairs_data):
    """
    Analyze how FVG behavior differs across pairs.
    """
    pair_stats = []
    
    for pair, data in all_pairs_data.items():
        stats = {
            'pair': pair,
            'total_scenarios': len(data),
            'upper_hit_rate': data['target'].mean(),
            'avg_candles_to_hit': data['candles_to_hit'].mean(),
            'avg_distance_ratio': (data['dist_to_upper_atr'] / data['dist_to_lower_atr']).mean(),
            'avg_volatility': data['atr'].mean() if 'atr' in data.columns else None,
        }
        pair_stats.append(stats)
    
    df_stats = pd.DataFrame(pair_stats)
    
    print("=== PAIR-SPECIFIC FVG BEHAVIOR ===")
    print(df_stats.to_string(index=False))
    
    # Visual comparison
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Upper hit rate by pair
    axes[0, 0].bar(df_stats['pair'], df_stats['upper_hit_rate'])
    axes[0, 0].set_title('Upper FVG Hit Rate by Pair')
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', label='50% baseline')
    axes[0, 0].legend()
    
    # Avg candles to hit
    axes[0, 1].bar(df_stats['pair'], df_stats['avg_candles_to_hit'])
    axes[0, 1].set_title('Avg Candles Until FVG Hit by Pair')
    
    # Total scenarios
    axes[1, 0].bar(df_stats['pair'], df_stats['total_scenarios'])
    axes[1, 0].set_title('Number of Competing FVG Scenarios by Pair')
    
    # Avg volatility
    if df_stats['avg_volatility'].notna().any():
        axes[1, 1].bar(df_stats['pair'], df_stats['avg_volatility'])
        axes[1, 1].set_title('Avg ATR by Pair')
    
    plt.tight_layout()
    plt.savefig('cross_pair_analysis.png', dpi=150)
    
    return df_stats
```

**Key Insights to Look For**:
1. **Consistent performance**: ROC-AUC should be within 0.05 across all pairs
2. **Different base rates**: Some pairs may naturally favor upper vs lower FVGs (trend bias)
3. **Volatility impact**: High-vol pairs (GBP/JPY) may behave differently than low-vol pairs (EUR/USD)
4. **Time to reversion**: Asian pairs (USD/JPY) may take longer to revert than European pairs

**If Cross-Pair Performance Varies Significantly**:
- Add pair-specific normalization (volatility percentile instead of raw ATR)
- Consider ensemble: different models for different volatility regimes
- Add "pair" as a categorical feature (one-hot encoded)
- Collect more data for underperforming pairs

### 4.3 Trading Simulation
```python
def backtest_trading_strategy(candles, scenarios, model, scaler, feature_cols,
                               prob_threshold=0.6, risk_per_trade=0.01):
    """
    Simulate trading based on model predictions.
    
    Strategy:
    - If P(upper) > threshold: Go LONG targeting upper FVG
    - If P(lower) > threshold: Go SHORT targeting lower FVG
    - Stop loss: opposite FVG
    """
    trades = []
    
    for scenario in scenarios:
        # Extract features
        features = extract_all_features(...)
        X = scaler.transform([features[col] for col in feature_cols])
        
        # Get probability
        prob_upper = model.predict_proba(X)[0, 1]
        
        # Trading decision
        if prob_upper > prob_threshold:
            # Go LONG targeting upper FVG
            entry_price = scenario['current_price']
            target = scenario['upper_fvg']['gap_mid']
            stop_loss = scenario['lower_fvg']['gap_mid']
            direction = 'LONG'
            
        elif prob_upper < (1 - prob_threshold):
            # Go SHORT targeting lower FVG
            entry_price = scenario['current_price']
            target = scenario['lower_fvg']['gap_mid']
            stop_loss = scenario['upper_fvg']['gap_mid']
            direction = 'SHORT'
        else:
            continue  # No trade (insufficient confidence)
        
        # Simulate trade outcome
        outcome = simulate_trade_outcome(candles, scenario['current_idx'], 
                                          entry_price, target, stop_loss, direction)
        
        trades.append({
            'entry_time': scenario['current_time'],
            'direction': direction,
            'entry_price': entry_price,
            'target': target,
            'stop_loss': stop_loss,
            'prob_confidence': prob_upper if direction == 'LONG' else 1 - prob_upper,
            'pnl': outcome['pnl'],
            'result': outcome['result'],  # 'WIN', 'LOSS', 'OPEN'
        })
    
    df_trades = pd.DataFrame(trades)
    
    # Performance metrics
    win_rate = (df_trades['result'] == 'WIN').mean()
    avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
    avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean()
    profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    print("=== BACKTEST RESULTS ===")
    print(f"Total Trades: {len(df_trades)}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Win: {avg_win:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    print(f"Profit Factor: {profit_factor:.2f}")
    
    return df_trades


def simulate_trade_outcome(candles, entry_idx, entry_price, target, stop_loss, direction):
    """
    Simulate how a trade would play out.
    """
    max_hold_candles = 100
    
    for i in range(entry_idx + 1, min(len(candles), entry_idx + max_hold_candles)):
        candle = candles.iloc[i]
        
        if direction == 'LONG':
            # Check if target hit
            if candle['high'] >= target:
                pnl = target - entry_price
                return {'result': 'WIN', 'pnl': pnl, 'exit_candles': i - entry_idx}
            
            # Check if stop loss hit
            if candle['low'] <= stop_loss:
                pnl = stop_loss - entry_price
                return {'result': 'LOSS', 'pnl': pnl, 'exit_candles': i - entry_idx}
        
        else:  # SHORT
            # Check if target hit
            if candle['low'] <= target:
                pnl = entry_price - target
                return {'result': 'WIN', 'pnl': pnl, 'exit_candles': i - entry_idx}
            
            # Check if stop loss hit
            if candle['high'] >= stop_loss:
                pnl = entry_price - stop_loss
                return {'result': 'LOSS', 'pnl': pnl, 'exit_candles': i - entry_idx}
    
    # Trade still open at end of window
    return {'result': 'OPEN', 'pnl': 0, 'exit_candles': None}
```

### 4.4 Sensitivity Analysis
```python
def analyze_probability_thresholds(df_trades_all, prob_threshold_range):
    """
    Test different probability thresholds for trade entry.
    """
    results = []
    
    for threshold in prob_threshold_range:
        # Filter trades by threshold
        df_filtered = df_trades_all[df_trades_all['prob_confidence'] >= threshold]
        
        if len(df_filtered) > 0:
            win_rate = (df_filtered['result'] == 'WIN').mean()
            avg_pnl = df_filtered['pnl'].mean()
            num_trades = len(df_filtered)
            
            results.append({
                'threshold': threshold,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'total_pnl': df_filtered['pnl'].sum(),
            })
    
    df_sensitivity = pd.DataFrame(results)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].plot(df_sensitivity['threshold'], df_sensitivity['num_trades'])
    axes[0, 0].set_title('Number of Trades vs Threshold')
    
    axes[0, 1].plot(df_sensitivity['threshold'], df_sensitivity['win_rate'])
    axes[0, 1].set_title('Win Rate vs Threshold')
    
    axes[1, 0].plot(df_sensitivity['threshold'], df_sensitivity['avg_pnl'])
    axes[1, 0].set_title('Average PnL vs Threshold')
    
    axes[1, 1].plot(df_sensitivity['threshold'], df_sensitivity['total_pnl'])
    axes[1, 1].set_title('Total PnL vs Threshold')
    
    plt.tight_layout()
    
    return df_sensitivity, fig
```

### Deliverables
- [ ] Walk-forward validation results (minimum 3 test windows)
- [ ] Backtest performance report with equity curve
- [ ] Sensitivity analysis for probability thresholds
- [ ] Statistical significance tests (permutation test for Sharpe ratio)
- [ ] Final model performance summary

### Success Criteria
- Walk-forward validation shows consistent performance (ROC-AUC variance < 0.1)
- Backtest achieves profit factor > 1.5
- Win rate > 55% at optimal probability threshold
- Results are statistically significant (p < 0.05 vs random trading)

---

## Phase 5: Production Implementation (Week 7)

### Objective
Deploy model to production environment for real-time predictions.

### 5.1 Real-Time Feature Calculator
```python
class RealTimeFVGPredictor:
    """
    Production-ready FVG probability predictor.
    """
    
    def __init__(self, model, scaler, feature_cols):
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.fvg_cache = []  # Store recent FVGs
        
    def update_candle(self, new_candle_1h, new_candle_4h, new_candle_daily):
        """
        Called every hour when new candle closes.
        """
        # Detect new FVGs
        new_fvg = identify_fvg(self.candles_1h, len(self.candles_1h) - 1)
        if new_fvg:
            self.fvg_cache.append(new_fvg)
        
        # Remove old/mitigated FVGs
        self.clean_fvg_cache()
        
        # Check for competing FVG setup
        prediction = self.check_competing_setup()
        
        return prediction
    
    def check_competing_setup(self):
        """
        Check if we currently have competing FVGs.
        """
        current_price = self.candles_1h.iloc[-1]['close']
        
        # Find active FVGs
        active_bearish = [fvg for fvg in self.fvg_cache 
                          if fvg['type'] == 'bearish' and fvg['gap_low'] > current_price]
        active_bullish = [fvg for fvg in self.fvg_cache 
                          if fvg['type'] == 'bullish' and fvg['gap_high'] < current_price]
        
        if not active_bearish or not active_bullish:
            return None
        
        # Take nearest FVGs
        upper_fvg = min(active_bearish, key=lambda x: x['gap_mid'])
        lower_fvg = max(active_bullish, key=lambda x: x['gap_mid'])
        
        # Extract features
        features = extract_all_features(
            self.candles_1h, self.candles_4h, self.candles_daily,
            upper_fvg, lower_fvg, len(self.candles_1h) - 1
        )
        
        # Predict
        X = self.scaler.transform([[features[col] for col in self.feature_cols]])
        prob_upper = self.model.predict_proba(X)[0, 1]
        
        return {
            'timestamp': self.candles_1h.index[-1],
            'current_price': current_price,
            'upper_fvg': upper_fvg,
            'lower_fvg': lower_fvg,
            'prob_upper': prob_upper,
            'prob_lower': 1 - prob_upper,
            'bias': 'LONG' if prob_upper > 0.6 else 'SHORT' if prob_upper < 0.4 else 'NEUTRAL',
            'confidence': max(prob_upper, 1 - prob_upper),
        }
    
    def clean_fvg_cache(self):
        """
        Remove FVGs that are too old or have been mitigated.
        """
        max_age_candles = 100
        current_idx = len(self.candles_1h) - 1
        
        self.fvg_cache = [
            fvg for fvg in self.fvg_cache
            if (current_idx - fvg['formation_index']) <= max_age_candles
            and not is_fvg_mitigated(self.candles_1h, fvg, current_idx)
        ]
```

### 5.2 API Endpoint
```python
from flask import Flask, jsonify, request
import pandas as pd

app = Flask(__name__)

# Load model at startup
predictor = RealTimeFVGPredictor.load('production_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for FVG probability prediction.
    
    Expected input:
    {
        "candles_1h": [...],  # Recent 1h candles
        "candles_4h": [...],  # Recent 4h candles
        "candles_daily": [...]  # Recent daily candles
    }
    """
    data = request.json
    
    # Update predictor with new data
    predictor.update_data(
        pd.DataFrame(data['candles_1h']),
        pd.DataFrame(data['candles_4h']),
        pd.DataFrame(data['candles_daily'])
    )
    
    # Get prediction
    prediction = predictor.check_competing_setup()
    
    if prediction is None:
        return jsonify({
            'status': 'no_setup',
            'message': 'No competing FVGs detected'
        })
    
    return jsonify({
        'status': 'prediction_ready',
        'timestamp': prediction['timestamp'].isoformat(),
        'current_price': float(prediction['current_price']),
        'upper_fvg': {
            'type': prediction['upper_fvg']['type'],
            'gap_low': float(prediction['upper_fvg']['gap_low']),
            'gap_high': float(prediction['upper_fvg']['gap_high']),
            'gap_mid': float(prediction['upper_fvg']['gap_mid']),
        },
        'lower_fvg': {
            'type': prediction['lower_fvg']['type'],
            'gap_low': float(prediction['lower_fvg']['gap_low']),
            'gap_high': float(prediction['lower_fvg']['gap_high']),
            'gap_mid': float(prediction['lower_fvg']['gap_mid']),
        },
        'probabilities': {
            'upper': float(prediction['prob_upper']),
            'lower': float(prediction['prob_lower']),
        },
        'bias': prediction['bias'],
        'confidence': float(prediction['confidence']),
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### 5.3 Monitoring & Alerts
```python
class ModelMonitor:
    """
    Monitor model performance in production.
    """
    
    def __init__(self, alert_threshold_accuracy=0.55):
        self.predictions = []
        self.alert_threshold = alert_threshold_accuracy
        
    def log_prediction(self, prediction, features):
        """
        Log prediction for later evaluation.
        """
        self.predictions.append({
            'timestamp': prediction['timestamp'],
            'prob_upper': prediction['prob_upper'],
            'features': features,
            'outcome': None,  # Will be filled when outcome is known
        })
    
    def log_outcome(self, timestamp, outcome):
        """
        Log actual outcome (which FVG was hit first).
        """
        for pred in self.predictions:
            if pred['timestamp'] == timestamp:
                pred['outcome'] = outcome
                break
        
        # Check if we need to alert
        self.check_performance_drift()
    
    def check_performance_drift(self, window=50):
        """
        Alert if recent performance degrades.
        """
        recent = [p for p in self.predictions if p['outcome'] is not None][-window:]
        
        if len(recent) < window:
            return
        
        # Calculate recent accuracy
        correct = sum(1 for p in recent 
                     if (p['outcome'] == 'upper' and p['prob_upper'] > 0.5) or
                        (p['outcome'] == 'lower' and p['prob_upper'] < 0.5))
        
        accuracy = correct / len(recent)
        
        if accuracy < self.alert_threshold:
            self.send_alert(f"Model accuracy dropped to {accuracy:.1%} (last {window} predictions)")
    
    def send_alert(self, message):
        """
        Send alert to team (email, Slack, etc.)
        """
        print(f"ALERT: {message}")
        # Implement actual alerting mechanism
```

### Deliverables
- [ ] Production-ready predictor class
- [ ] REST API with documentation
- [ ] Docker container for deployment
- [ ] Monitoring dashboard (Grafana/custom)
- [ ] Alerting system for model drift
- [ ] Deployment runbook

### Success Criteria
- API response time < 100ms
- Model predictions logged with timestamps for audit trail
- Monitoring system tracks predictions vs outcomes
- Zero downtime deployment process documented

---

## Phase 6: Optimization & Iteration (Week 8)

### Objective
Refine model based on production data and feedback.

### Tasks
- [ ] Collect 2+ weeks of production predictions and outcomes
- [ ] Re-evaluate model performance on live data
- [ ] Identify feature drift or data distribution changes
- [ ] Retrain model if necessary
- [ ] A/B test model variations
- [ ] Optimize probability thresholds based on realized results

### Continuous Improvements
1. **Feature additions**: Test new features based on domain insights
2. **Ensemble methods**: Combine multiple models for robustness
3. **Higher timeframes**: Test on 4H FVGs (requires more data)
4. **Multi-class models**: Predict "neither" outcome explicitly
5. **Deep learning**: Experiment with LSTM/Transformer if sufficient data

---

## Key Performance Indicators (KPIs)

Track these metrics throughout the project:

| Metric | Target | Phase |
|--------|--------|-------|
| FVG Detection Accuracy | 100% | 0 |
| Dataset Size | 1000+ scenarios | 2 |
| Model ROC-AUC | > 0.65 | 3 |
| Calibration Error | < 0.05 | 3 |
| Cross-Pair ROC-AUC Variance | < 0.1 | 4 |
| Backtest Win Rate | > 55% | 4 |
| Backtest Profit Factor | > 1.5 | 4 |
| API Response Time | < 100ms | 5 |
| Production Accuracy (30-day) | > 60% | 6 |

---

## Risk Management

### Technical Risks
1. **Insufficient data**: Mitigation → Collect data from multiple currency pairs
2. **Overfitting**: Mitigation → Strict train/val/test splits, regularization, walk-forward validation
3. **Lookahead bias**: Mitigation → Code review, unit tests for feature extraction timing
4. **Model drift**: Mitigation → Continuous monitoring, periodic retraining

### Trading Risks
1. **Market regime changes**: FVG behavior may differ in crisis vs normal markets
2. **Execution slippage**: Model assumes perfect entry at closing prices
3. **Position sizing**: Model provides bias, not complete trading system

---

## Team Roles

- **Lead Quant Engineer**: Overall architecture, model selection
- **Data Engineer**: Pipeline for data collection, feature storage
- **ML Engineer**: Model training, hyperparameter tuning, deployment
- **QA Engineer**: Testing, validation, code review
- **DevOps Engineer**: Production infrastructure, monitoring

---

## Tools & Technologies

### Development
- Python 3.9+
- pandas, numpy, scikit-learn
- XGBoost / LightGBM
- lifelines (survival analysis)
- matplotlib, seaborn (visualization)

### Production
- Flask / FastAPI (API)
- Docker (containerization)
- PostgreSQL / TimescaleDB (time-series data)
- Redis (caching)
- Prometheus + Grafana (monitoring)

### Version Control
- Git for code
- DVC for data/model versioning
- MLflow for experiment tracking

---

## Timeline Summary

| Week | Phase | Deliverables |
|------|-------|--------------|
| 0 | Phase -1 | Data infrastructure & collection |
| 1 | Phase 0 | FVG detection validated |
| 2 | Phase 1 | Feature pipeline complete |
| 3 | Phase 2 | Labeled dataset created |
| 4-5 | Phase 3 | Models trained & compared |
| 6 | Phase 4 | Backtest complete |
| 7 | Phase 5 | Production deployment |
| 8-9 | Phase 6 | Optimization & monitoring |

**Total Duration**: 9-10 weeks

---

## Next Steps

1. **Team kickoff meeting**: Review this document, assign roles
2. **Set up infrastructure**: Dev environment, data storage, Git repo
3. **Begin Phase 0**: Implement FVG detection algorithm

**Questions or clarifications**: [Contact project lead]

---

**Document Version**: 1.0  
**Last Updated**: [Date]  
**Project Lead**: [Name]
