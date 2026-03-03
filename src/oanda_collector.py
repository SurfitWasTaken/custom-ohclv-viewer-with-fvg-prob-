"""
OANDA Data Collector for FVG Probability Modeling
Adapted from MT5DataCollector in workflow to use OANDA v20 REST API
"""

import os
import pandas as pd
from datetime import datetime, timedelta
import time
from oandapyV20 import API
from oandapyV20.exceptions import V20Error
import oandapyV20.endpoints.instruments as instruments
from dotenv import load_dotenv


class OANDADataCollector:
    """
    Collect historical OHLCV data from OANDA v20 API.
    
    Deviation from workflow: Uses OANDA REST API instead of MT5.
    Maintains same interface for compatibility with downstream phases.
    """
    
    def __init__(self, access_token=None, account_id=None, environment="practice"):
        """
        Initialize OANDA API connection.
        
        Parameters:
        -----------
        access_token : str, OANDA API access token
        account_id : str, OANDA account ID
        environment : str, 'practice' or 'live'
        """
        # Load from environment if not provided
        if access_token is None:
            load_dotenv()
            access_token = os.getenv('OANDA_ACCESS_TOKEN')
            account_id = os.getenv('OANDA_ACCOUNT_ID')
        
        if not access_token:
            raise ValueError("OANDA_ACCESS_TOKEN not found in environment")
        
        self.access_token = access_token
        self.account_id = account_id
        self.environment = environment
        self.client = API(access_token=access_token, environment=environment)
        
        print(f"OANDA API initialized: {environment} environment")
        if account_id:
            masked_id = f"{account_id[:3]}...{account_id[-3:]}"
            print(f"Account ID: {masked_id}")
    
    def _oanda_instrument_name(self, symbol):
        """
        Convert standard symbol to OANDA format.
        
        Examples:
        EURUSD -> EUR_USD
        GBPUSD -> GBP_USD
        """
        if '_' in symbol:
            return symbol  # Already in OANDA format
        
        # Standard forex pairs are 6 characters
        if len(symbol) == 6:
            return f"{symbol[:3]}_{symbol[3:]}"
        
        raise ValueError(f"Invalid symbol format: {symbol}")
    
    # All OANDA-native granularities we support
    GRANULARITY_MAP = {
        'S5':  'S5',   '5S':  'S5',
        'S10': 'S10',  '10S': 'S10',
        'S15': 'S15',  '15S': 'S15',
        'S30': 'S30',  '30S': 'S30',
        'M1':  'M1',   '1M':  'M1',
        'M5':  'M5',   '5M':  'M5',
        'M15': 'M15',  '15M': 'M15',
        'M30': 'M30',  '30M': 'M30',
        'H1':  'H1',   '1H':  'H1',
        'H2':  'H2',   '2H':  'H2',
        'H4':  'H4',   '4H':  'H4',
        'D':   'D',    '1D':  'D',
        'W':   'W',    '1W':  'W',
    }

    # Max chunk sizes (in days) to stay under 5000-candle OANDA limit
    _CHUNK_DAYS = {
        'S5': 0.25, 'S10': 0.5, 'S15': 0.75, 'S30': 1.5,
        'M1': 3, 'M5': 15, 'M15': 45, 'M30': 90,
        'H1': 180, 'H2': 360, 'H4': 720,
        'D': 4000, 'W': 4000,
    }

    def _oanda_granularity(self, timeframe_str):
        """
        Map timeframe strings to OANDA granularity codes.
        
        Supports: M1, M5, M15, M30, H1, H2, H4, D, W
        (and aliases like 1M, 5M, 15M, 1H, 4H, 1D, 1W)
        """
        granularity = self.GRANULARITY_MAP.get(timeframe_str.upper())
        if not granularity:
            raise ValueError(
                f"Unsupported timeframe: {timeframe_str}. "
                f"Supported: {sorted(set(self.GRANULARITY_MAP.values()))}")
        return granularity
    
    def get_historical_data(self, symbol, timeframe_str, start_date, end_date):
        """
        Fetch historical OHLCV data from OANDA.
        
        Parameters:
        -----------
        symbol : str, e.g. 'EURUSD' or 'EUR_USD'
        timeframe_str : str, '1H', '4H', '1D', '15M'
        start_date : datetime
        end_date : datetime
        
        Returns:
        --------
        pd.DataFrame with OHLCV data
        
        Notes:
        ------
        OANDA API limits:
        - Max 5000 candles per request
        - Rate limit: ~120 requests/second (generous)
        - Historical data: Up to 5 years for daily, varies for intraday
        
        Strategy: Break large date ranges into chunks to stay under 5000 candle limit
        """
        instrument = self._oanda_instrument_name(symbol)
        granularity = self._oanda_granularity(timeframe_str)
        
        print(f"Fetching {symbol} {timeframe_str} from {start_date} to {end_date}...")
        
        # Calculate chunk size based on timeframe to stay under 5000 candles
        chunk_days = self._CHUNK_DAYS.get(granularity, 180)
        
        all_candles = []
        current_start = start_date
        
        while current_start < end_date:
            # Calculate chunk end date
            chunk_end = min(current_start + timedelta(days=chunk_days), end_date)
            
            # Convert dates to RFC3339 format (OANDA requirement)
            from_time = current_start.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            to_time = chunk_end.strftime('%Y-%m-%dT%H:%M:%S.000000000Z')
            
            params = {
                "from": from_time,
                "to": to_time,
                "granularity": granularity,
                "price": "M",  # Midpoint prices (can also use 'BA' for bid/ask)
            }
            
            try:
                request = instruments.InstrumentsCandles(instrument=instrument, params=params)
                response = self.client.request(request)
                
                candles = response.get('candles', [])
                
                if candles:
                    all_candles.extend(candles)
                    print(f"  Fetched {len(candles)} candles ({current_start.date()} to {chunk_end.date()}), total: {len(all_candles)}")
                
                # Rate limiting - be nice to OANDA servers
                time.sleep(0.2)
                
            except V20Error as e:
                print(f"OANDA API Error: {e}")
                raise
            
            # Move to next chunk
            current_start = chunk_end
        
        if not all_candles:
            raise ValueError(f"No data returned for {symbol} {timeframe_str}")
        
        # Convert to DataFrame
        df = self._candles_to_dataframe(all_candles)
        
        print(f"✓ Fetched {len(df)} candles for {symbol} {timeframe_str}")
        
        return df
    
    def _candles_to_dataframe(self, candles, include_incomplete=False):
        """
        Convert OANDA candle format to standard OHLCV DataFrame.
        
        OANDA candle structure:
        {
            'time': '2021-01-04T00:00:00.000000000Z',
            'volume': 12345,
            'complete': True,
            'mid': {'o': '1.22', 'h': '1.23', 'l': '1.21', 'c': '1.22'}
        }
        """
        data = []
        
        for candle in candles:
            if not include_incomplete and not candle.get('complete', False):
                continue
            
            mid = candle['mid']
            
            data.append({
                'timestamp': pd.to_datetime(candle['time']),
                'open': float(mid['o']),
                'high': float(mid['h']),
                'low': float(mid['l']),
                'close': float(mid['c']),
                'volume': int(candle['volume']),
                'complete': candle.get('complete', True),
            })
        
        df = pd.DataFrame(data)
        df = df.set_index('timestamp')
        df = df.sort_index()
        
        return df

    def fetch_recent(self, symbol, timeframe_str, count=500,
                     include_incomplete=True):
        """
        Fetch the N most recent candles — optimised for dashboard use.

        Uses OANDA's 'count' parameter so no date-range chunking is needed.

        Parameters
        ----------
        symbol : str, e.g. 'EURUSD'
        timeframe_str : str, e.g. 'H1', '1H', 'M5'
        count : int, number of candles (max 5000)
        include_incomplete : bool, include the currently forming candle

        Returns
        -------
        pd.DataFrame with OHLCV data
        """
        instrument = self._oanda_instrument_name(symbol)
        granularity = self._oanda_granularity(timeframe_str)

        params = {
            'granularity': granularity,
            'count': min(count, 5000),
            'price': 'M',
        }

        request = instruments.InstrumentsCandles(
            instrument=instrument, params=params)
        response = self.client.request(request)
        candles = response.get('candles', [])

        return self._candles_to_dataframe(
            candles, include_incomplete=include_incomplete)
    
    def collect_all_instruments(self, symbols, timeframes, start_date, end_date, 
                                  output_dir='./data/raw'):
        """
        Collect data for multiple symbols and timeframes.
        
        Parameters:
        -----------
        symbols : list of str, e.g. ['EURUSD', 'GBPUSD']
        timeframes : list of str, e.g. ['1H', '4H', '1D']
        start_date : datetime
        end_date : datetime
        output_dir : str, directory to save CSV files
        
        Returns:
        --------
        dict, collection results summary
        """
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
                    
                    print(f"✓ Saved {symbol} {timeframe}: {len(df)} candles to {filename}")
                    
                    # Rate limiting between instruments
                    time.sleep(1)
                    
                except Exception as e:
                    results[f"{symbol}_{timeframe}"] = {
                        'status': 'failed',
                        'error': str(e)
                    }
                    print(f"✗ Failed {symbol} {timeframe}: {e}")
        
        return results


# Test script
if __name__ == '__main__':
    # Initialize collector
    collector = OANDADataCollector(environment="practice")
    
    # Test with single instrument first
    print("\n=== TESTING SINGLE INSTRUMENT ===")
    try:
        df = collector.get_historical_data(
            symbol='EURUSD',
            timeframe_str='1H',
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 2, 1)
        )
        print(f"\nSample data:")
        print(df.head())
        print(f"\nData shape: {df.shape}")
        print(f"Date range: {df.index[0]} to {df.index[-1]}")
    except Exception as e:
        print(f"Test failed: {e}")
