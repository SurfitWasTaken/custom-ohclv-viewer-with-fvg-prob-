"""
V2 Visual Dashboard — Per-FVG Fill Probability
Interactive dashboard with Lightweight Charts, MTF support, and NN overlays.
"""

import os, sys, json, glob, argparse, traceback
from datetime import datetime, timedelta, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import joblib

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.fvg_detector import scan_all_fvgs
from src.dataset_builder import is_fvg_mitigated
from src.fill_predictor import FillPredictor
from src.candle_aggregator import aggregate_candles
from src.indicator_engine import compute_ema, compute_vwap, compute_bollinger

DATA_DIR  = os.path.join(_PROJECT_ROOT, 'data', 'raw')
MODEL_DIR = os.path.join(_PROJECT_ROOT, 'models')
PAIRS     = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

_USE_LIVE = True
_COLLECTOR = None
_PREDICTOR = None
_CACHE = {}
_CACHE_TTL = 10

def _get_collector():
    global _COLLECTOR
    if _COLLECTOR is not None:
        return _COLLECTOR
    try:
        from src.oanda_collector import OANDADataCollector
        _COLLECTOR = OANDADataCollector(environment="practice")
        return _COLLECTOR
    except Exception as e:
        print(f"  ⚠ OANDA: {e}")
        return None

def _get_predictor():
    global _PREDICTOR
    if _PREDICTOR is not None:
        return _PREDICTOR
    try:
        _PREDICTOR = FillPredictor(MODEL_DIR)
        return _PREDICTOR
    except Exception as e:
        print(f"  ⚠ Model not loaded: {e}")
        return None

def _load_csv(symbol, tf):
    tf_map = {'S5': '5S', 'S10': '10S', 'S15': '15S', 'S30': '30S', 'H1': '1H', 'H2': '2H', 'H4': '4H', 'D': '1D', 'W': '1W', 'M1': '1M', 'M5': '5M', 'M15': '15M', 'M30': '30M'}
    file_tf = tf_map.get(tf, tf)
    pattern = os.path.join(DATA_DIR, f'{symbol}_{file_tf}_*.csv')
    files = glob.glob(pattern)
    if not files: return None
    return pd.read_csv(files[0], index_col='timestamp', parse_dates=True)

def _get_candles(symbol, tf, before_ts=None, count=500):
    if not _USE_LIVE:
        df = _load_csv(symbol, tf)
        if df is None: return pd.DataFrame()
        if before_ts:
            dt = datetime.fromtimestamp(before_ts, tz=timezone.utc)
            # Make sure df index has timezone to compare
            if df.index.tzinfo is None:
                df.index = df.index.tz_localize('UTC')
            df = df[df.index < dt]
        return df.tail(count)
    
    c = _get_collector()
    if not c: return pd.DataFrame()
    
    try:
        if before_ts:
            dt = datetime.fromtimestamp(before_ts, tz=timezone.utc)
            # Find the best chunk
            days = c._CHUNK_DAYS.get(c._oanda_granularity(tf), 180)
            start = dt - timedelta(days=days)
            df = c.get_historical_data(symbol, tf, start, dt)
            return df.tail(count)
        else:
            return c.fetch_recent(symbol, tf, count=count)
    except Exception as e:
        print(f"Error fetching candles: {e}")
        return pd.DataFrame()

def api_candles(symbol, tf, before_ts, count, base_tf, interval):
    if tf == 'custom':
        # fetch base tf
        df = _get_candles(symbol, base_tf, before_ts, count * interval)
        if df.empty: return []
        df = aggregate_candles(df, interval)
        df = df.tail(count)
    else:
        df = _get_candles(symbol, tf, before_ts, count)
        
    if df.empty: return []
    
    candles = []
    for ts, r in df.iterrows():
        candles.append({
            'time': ts.timestamp(),
            'open': round(float(r['open']), 5),
            'high': round(float(r['high']), 5),
            'low': round(float(r['low']), 5),
            'close': round(float(r['close']), 5),
            'volume': round(float(r['volume']), 2) if 'volume' in df.columns else 0
        })
    return candles

def api_fvgs(symbol):
    # FVGs need H1, H4, D
    c1h = _get_candles(symbol, 'H1', count=300)
    if c1h.empty: return {'active_fvgs': [], 'bias_summary': {}}
    c4h = _get_candles(symbol, 'H4', count=300)
    c1d = _get_candles(symbol, 'D', count=300)
    
    predictor = _get_predictor()
    if predictor:
        try:
            res = predictor.predict_all(c1h, c4h, c1d)
            # Add timestamps for LWC to plot shapes
            for f in res['active_fvgs']:
                f['time'] = c1h.index[f['formation_idx']].timestamp()
            return res
        except Exception as e:
            print(f"Prediction error: {e}")
            pass
            
    # fallback
    all_fvgs = scan_all_fvgs(c1h)
    active = []
    current_idx = len(c1h) - 1
    for f in all_fvgs:
        age = current_idx - f['formation_index']
        if age < 2 or age > 100: continue
        if is_fvg_mitigated(c1h, f, current_idx + 1): continue
        active.append({
            'fvg_type': f['type'],
            'gap_low': round(float(f['gap_low']), 5),
            'gap_high': round(float(f['gap_high']), 5),
            'time': c1h.index[f['formation_index']].timestamp(),
            'age_candles': age,
            'urgency': 'unknown',
            'fill_probabilities': {}
        })
    return {'active_fvgs': active, 'bias_summary': {}}

def api_indicators(symbol, tf, ind_type, period, std_dev=2.0):
    df = _get_candles(symbol, tf, count=500)
    if df.empty: return {}
    
    res = []
    try:
        if ind_type == 'ema':
            s = compute_ema(df, period)
            for ts, v in s.dropna().items():
                res.append({'time': ts.timestamp(), 'value': v})
            return {'data': res}
        elif ind_type == 'vwap':
            s = compute_vwap(df)
            for ts, v in s.dropna().items():
                res.append({'time': ts.timestamp(), 'value': v})
            return {'data': res}
        elif ind_type == 'bollinger':
            b = compute_bollinger(df, period, std_dev)
            upper, lower, mid = [], [], []
            for ts in b['mid'].dropna().index:
                t = ts.timestamp()
                upper.append({'time': t, 'value': b['upper'][ts]})
                lower.append({'time': t, 'value': b['lower'][ts]})
                mid.append({'time': t, 'value': b['mid'][ts]})
            return {'upper': upper, 'lower': lower, 'mid': mid}
    except Exception as e:
        print(f"Indicator error: {e}")
    return {}

def _build_html(symbol='EURUSD'):
    available = [p for p in PAIRS if _load_csv(p, 'H1') is not None or _USE_LIVE]
    pair_opts = ''.join(f'<option value="{p}" {"selected" if p == symbol else ""}>{p}</option>' for p in available)
    html_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        return html.replace('{pair_opts}', pair_opts).replace('{symbol}', symbol)
    return "<html><body><h1>index.html not found</h1></body></html>"

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        sym = qs.get('pair', ['EURUSD'])[0].upper()
        
        try:
            if parsed.path == '/api/candles':
                tf = qs.get('tf', ['H1'])[0]
                before = float(qs.get('before', [0])[0]) or None
                count = int(qs.get('count', [500])[0])
                base_tf = qs.get('base', ['M1'])[0]
                interval = int(qs.get('interval', [12])[0])
                data = api_candles(sym, tf, before, count, base_tf, interval)
            elif parsed.path == '/api/fvgs':
                data = api_fvgs(sym)
            elif parsed.path == '/api/indicators':
                tf = qs.get('tf', ['H1'])[0]
                ind_type = qs.get('type', ['ema'])[0]
                period = int(qs.get('period', [20])[0])
                data = api_indicators(sym, tf, ind_type, period)
            else:
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.end_headers()
                self.wfile.write(_build_html(sym).encode('utf-8'))
                return
                
            body = json.dumps(data, default=str).encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)
        except Exception as e:
            traceback.print_exc()
            self.send_response(500)
            self.end_headers()
            self.wfile.write(json.dumps({'error': str(e)}).encode('utf-8'))
            
    def log_message(self, format, *args): pass


def main():
    global _USE_LIVE
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5001)
    parser.add_argument('--offline', action='store_true')
    args = parser.parse_args()

    if args.offline:
        _USE_LIVE = False
        print("OFFLINE mode (CSV data)")
    else:
        print("LIVE mode (OANDA API)")
        _get_collector()

    _get_predictor()

    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Pairs: {', '.join(PAIRS)}")
    print("Press Ctrl+C to stop.\n")

    server = HTTPServer(('0.0.0.0', args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == '__main__':
    main()
