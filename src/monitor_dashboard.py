"""
V2 Visual Dashboard — Per-FVG Fill Probability

Interactive dashboard showing ALL active (unmitigated) FVGs with
fill probability timelines.

Features:
  - Candlestick chart with FVGs color-coded by urgency
  - Click any FVG to see its probability timeline
  - Sidebar with all active FVGs and their predictions
  - Live OANDA data integration
  - Auto-refresh every 60s

Usage:
    python3 src/monitor_dashboard.py             # port 5001
    python3 src/monitor_dashboard.py --offline    # CSV fallback
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

DATA_DIR  = os.path.join(_PROJECT_ROOT, 'data', 'raw')
MODEL_DIR = os.path.join(_PROJECT_ROOT, 'models')
PAIRS     = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
N_DISPLAY = 300

_USE_LIVE = True
_COLLECTOR = None
_PREDICTOR = None
_CACHE = {}
_CACHE_TTL = 55


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


def _fetch_live(symbol, tf, n=500):
    c = _get_collector()
    if c is None:
        return None
    try:
        now = datetime.now(tz=timezone.utc)
        start = now - timedelta(hours=n + 10)
        return c.get_historical_data(symbol, tf, start, now)
    except:
        return None


def _load_csv(symbol, tf):
    pattern = os.path.join(DATA_DIR, f'{symbol}_{tf}_*.csv')
    files = glob.glob(pattern)
    if not files:
        return None
    return pd.read_csv(files[0], index_col='timestamp', parse_dates=True)


def _load(symbol, tf):
    if _USE_LIVE:
        df = _fetch_live(symbol, tf)
        if df is not None and len(df) > 50:
            return df
    return _load_csv(symbol, tf)


def build_chart_data(symbol):
    now_ts = datetime.now(tz=timezone.utc).timestamp()
    if symbol in _CACHE:
        ts, data = _CACHE[symbol]
        if now_ts - ts < _CACHE_TTL:
            return data

    c1h = _load(symbol, '1H')
    c4h = _load(symbol, '4H')
    c1d = _load(symbol, '1D')
    if c1h is None:
        return {'error': f'No data for {symbol}'}

    display = c1h.iloc[-N_DISPLAY:]
    current_price = float(c1h.iloc[-1]['close'])
    current_idx = len(c1h) - 1
    display_start = len(c1h) - N_DISPLAY

    # Check if live
    last_ts = display.index[-1]
    if hasattr(last_ts, 'tz_localize') and last_ts.tzinfo is None:
        last_ts = last_ts.tz_localize('UTC')
    is_live = (datetime.now(tz=timezone.utc) - last_ts).total_seconds() / 3600 < 2

    candles = [{'time': ts.strftime('%Y-%m-%d %H:%M'),
                'open': round(float(r['open']), 5),
                'high': round(float(r['high']), 5),
                'low': round(float(r['low']), 5),
                'close': round(float(r['close']), 5)}
               for ts, r in display.iterrows()]

    # Get predictions for all active FVGs
    predictor = _get_predictor()
    active_fvgs = []
    bias_summary = {'bullish_below': 0, 'bearish_above': 0, 'net_bias': 'no model', 'total_active': 0}

    if predictor:
        try:
            result = predictor.predict_all(c1h, c4h, c1d)
            active_fvgs = result['active_fvgs']
            bias_summary = result['bias_summary']

            # Map formation_idx to display-relative indices
            for f in active_fvgs:
                f['display_idx'] = max(0, f['formation_idx'] - display_start)
        except Exception as e:
            active_fvgs = []
            bias_summary['error'] = str(e)
    else:
        # Fallback: show FVGs without predictions
        all_fvgs = scan_all_fvgs(c1h)
        for fvg in all_fvgs:
            age = current_idx - fvg['formation_index']
            if age < 2 or age > 100:
                continue
            if is_fvg_mitigated(c1h, fvg, current_idx + 1):
                continue
            if fvg['formation_index'] < display_start:
                continue
            active_fvgs.append({
                'fvg_type': fvg['type'],
                'gap_low': round(float(fvg['gap_low']), 5),
                'gap_high': round(float(fvg['gap_high']), 5),
                'gap_mid': round(float(fvg['gap_mid']), 5),
                'gap_size': round(float(fvg['gap_size']), 5),
                'formation_idx': fvg['formation_index'],
                'display_idx': max(0, fvg['formation_index'] - display_start),
                'age_candles': age,
                'fill_probabilities': {},
                'urgency': 'unknown',
            })

    data = {
        'symbol': symbol,
        'candles': candles,
        'active_fvgs': active_fvgs,
        'bias_summary': bias_summary,
        'current_price': round(current_price, 5),
        'is_live': is_live,
        'last_candle': candles[-1]['time'] if candles else '',
        'data_source': 'OANDA Live' if (is_live and _USE_LIVE) else 'CSV (offline)',
        'has_model': predictor is not None,
    }
    _CACHE[symbol] = (now_ts, data)
    return data


def _build_html(symbol='EURUSD'):
    available = [p for p in PAIRS if _load_csv(p, '1H') is not None or _USE_LIVE]
    pair_opts = ''.join(
        f'<option value="{p}" {"selected" if p == symbol else ""}>{p}</option>'
        for p in available)

    return f'''<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>FVG Fill Probability — {symbol}</title>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
       background:#0f1117; color:#e0e0e0; }}
.hdr {{ display:flex; align-items:center; gap:16px; padding:14px 24px;
        background:#1a1d29; border-bottom:1px solid #2a2d3a; }}
.hdr h1 {{ font-size:1.2rem; color:#fff; }}
.hdr select {{ background:#252836; color:#e0e0e0; border:1px solid #3a3d4a;
               padding:8px 12px; border-radius:8px; font-size:0.9rem; }}
.badge {{ padding:4px 12px; border-radius:20px; font-size:0.75rem; font-weight:600; color:#fff; }}
.badge.competing {{ background:#1565c0; }}
.badge.bearish_pull {{ background:#c62828; }}
.badge.bullish_pull {{ background:#2e7d32; }}
.badge.none {{ background:#555; }}
.live-dot {{ width:8px; height:8px; border-radius:50%; display:inline-block; margin-right:4px; }}
.live-dot.on {{ background:#4caf50; animation:pulse 1.5s infinite; }}
.live-dot.off {{ background:#c62828; }}
@keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.4}} }}

.main {{ display:grid; grid-template-columns:1fr 360px; height:calc(100vh - 56px); }}
.chart-area {{ padding:12px; }}
#chart {{ width:100%; height:100%; border-radius:12px; }}

.side {{ background:#1a1d29; border-left:1px solid #2a2d3a; padding:16px; overflow-y:auto; }}
.panel {{ background:#252836; border-radius:12px; padding:14px; margin-bottom:14px;
          border:1px solid #2a2d3a; }}
.panel h3 {{ font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px; margin-bottom:10px; }}
.stat {{ display:flex; justify-content:space-between; padding:5px 0;
         border-bottom:1px solid #2a2d3a; font-size:0.85rem; }}
.stat:last-child {{ border-bottom:none; }}
.lbl {{ color:#888; }}
.val {{ font-weight:600; }}
.val.g {{ color:#4caf50; }} .val.r {{ color:#ef5350; }} .val.b {{ color:#42a5f5; }}

.fvg-card {{ background:#1e2130; border-radius:10px; padding:12px; margin-bottom:10px;
             border:1px solid #2a2d3a; cursor:pointer; transition:border-color 0.2s; }}
.fvg-card:hover {{ border-color:#42a5f5; }}
.fvg-card.selected {{ border-color:#42a5f5; box-shadow:0 0 10px rgba(66,165,245,0.2); }}
.fvg-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
.fvg-type {{ font-weight:700; font-size:0.85rem; }}
.fvg-type.bullish {{ color:#4caf50; }}
.fvg-type.bearish {{ color:#ef5350; }}
.fvg-range {{ color:#888; font-size:0.8rem; }}
.urgency-tag {{ padding:2px 8px; border-radius:12px; font-size:0.7rem; font-weight:600; }}
.urgency-tag.imminent {{ background:rgba(239,83,80,0.2); color:#ef5350; }}
.urgency-tag.moderate {{ background:rgba(255,152,0,0.2); color:#ff9800; }}
.urgency-tag.low {{ background:rgba(76,175,80,0.2); color:#4caf50; }}

.prob-row {{ display:flex; align-items:center; gap:8px; padding:3px 0; font-size:0.8rem; }}
.prob-label {{ width:55px; color:#888; text-align:right; flex-shrink:0; }}
.prob-bar {{ flex:1; height:16px; background:#1a1d29; border-radius:4px; overflow:hidden; }}
.prob-fill {{ height:100%; border-radius:4px; transition:width 0.3s; }}
.prob-fill.hi {{ background:linear-gradient(90deg, #ef5350, #ff7043); }}
.prob-fill.md {{ background:linear-gradient(90deg, #ff9800, #ffc107); }}
.prob-fill.lo {{ background:linear-gradient(90deg, #4caf50, #66bb6a); }}
.prob-val {{ width:38px; font-weight:600; font-size:0.75rem; flex-shrink:0; }}

.empty {{ text-align:center; color:#555; padding:20px; font-style:italic; }}
</style>
</head><body>

<div class="hdr">
  <h1>📊 FVG Fill Probability</h1>
  <select id="pair" onchange="window.location='/?pair='+this.value">{pair_opts}</select>
  <span id="bias-badge" class="badge none"></span>
  <span style="flex:1"></span>
  <span id="live" style="color:#888;font-size:0.8rem;"></span>
</div>

<div class="main">
  <div class="chart-area"><div id="chart"></div></div>
  <div class="side" id="side">
    <div class="panel"><h3>Loading...</h3><div class="empty">Fetching data...</div></div>
  </div>
</div>

<script>
const SYM = new URLSearchParams(location.search).get('pair')||'EURUSD';
document.getElementById('pair').value = SYM;

async function load() {{
  return (await fetch('/api/data?pair='+SYM)).json();
}}

function urgencyColor(p) {{
  if(p>0.8) return 'hi';
  if(p>0.4) return 'md';
  return 'lo';
}}

function renderChart(d) {{
  const c = d.candles;
  const trace = {{
    x:c.map(v=>v.time), open:c.map(v=>v.open), high:c.map(v=>v.high),
    low:c.map(v=>v.low), close:c.map(v=>v.close), type:'candlestick',
    increasing:{{line:{{color:'#26a69a'}},fillcolor:'#26a69a'}},
    decreasing:{{line:{{color:'#ef5350'}},fillcolor:'#ef5350'}}, name:SYM
  }};

  const shapes = [];
  d.active_fvgs.forEach(f => {{
    const si = Math.max(0, (f.display_idx||0) - 1);
    if(si >= c.length) return;
    const p10 = (f.fill_probabilities||{{}})['10'] || 0;
    let alpha, border;
    if(p10 > 0.8) {{ alpha='0.25'; border='rgba(239,83,80,0.8)'; }}
    else if(p10 > 0.4) {{ alpha='0.15'; border='rgba(255,152,0,0.6)'; }}
    else {{ alpha='0.10'; border = f.fvg_type==='bullish' ? 'rgba(76,175,80,0.5)' : 'rgba(239,83,80,0.5)'; }}

    const base = f.fvg_type==='bullish' ? 'rgba(76,175,80,'+alpha+')' : 'rgba(239,83,80,'+alpha+')';
    shapes.push({{
      type:'rect', x0:c[si].time, x1:c[c.length-1].time,
      y0:f.gap_low, y1:f.gap_high,
      fillcolor:base, line:{{color:border,width:p10>0.4?2:1}}, layer:'below'
    }});
  }});

  Plotly.newPlot('chart', [trace], {{
    paper_bgcolor:'#0f1117', plot_bgcolor:'#0f1117',
    font:{{color:'#888',size:11}},
    xaxis:{{rangeslider:{{visible:false}},gridcolor:'#1e2130',type:'category',nticks:10}},
    yaxis:{{gridcolor:'#1e2130',side:'right'}},
    margin:{{l:10,r:60,t:10,b:40}}, shapes, dragmode:'pan'
  }}, {{responsive:true,displayModeBar:true,scrollZoom:true,
        modeBarButtonsToRemove:['toImage','lasso2d','select2d']}});
}}

function renderSide(d) {{
  const b = d.bias_summary;
  const badge = document.getElementById('bias-badge');
  const live = document.getElementById('live');

  if(d.is_live) live.innerHTML='<span class="live-dot on"></span>LIVE — '+d.data_source;
  else live.innerHTML='<span class="live-dot off"></span>OFFLINE — '+d.data_source;

  let biasClass='none', biasText='NO FVGS';
  if(b.net_bias==='competing') {{ biasClass='competing'; biasText='⚔ COMPETING'; }}
  else if(b.net_bias==='bearish_pull') {{ biasClass='bearish_pull'; biasText='▼ BEARISH PULL'; }}
  else if(b.net_bias==='bullish_pull') {{ biasClass='bullish_pull'; biasText='▲ BULLISH PULL'; }}
  badge.textContent=biasText; badge.className='badge '+biasClass;

  let html=`<div class="panel"><h3>Summary</h3>
    <div class="stat"><span class="lbl">Active FVGs</span><span class="val">${{b.total_active||d.active_fvgs.length}}</span></div>
    <div class="stat"><span class="lbl">Bullish (below)</span><span class="val g">${{b.bullish_below||0}}</span></div>
    <div class="stat"><span class="lbl">Bearish (above)</span><span class="val r">${{b.bearish_above||0}}</span></div>
    <div class="stat"><span class="lbl">Net Bias</span><span class="val b">${{b.net_bias||'—'}}</span></div>
    <div class="stat"><span class="lbl">Price</span><span class="val">${{d.current_price}}</span></div>
    <div class="stat"><span class="lbl">Source</span><span class="val">${{d.data_source}}</span></div>
  </div>`;

  if(!d.active_fvgs.length) {{
    html+=`<div class="panel"><h3>FVGs</h3><div class="empty">No active FVGs for ${{SYM}}</div></div>`;
  }} else {{
    html+=`<div class="panel"><h3>Active FVGs (${{d.active_fvgs.length}})</h3>`;
    d.active_fvgs.forEach(f => {{
      const probs = f.fill_probabilities || {{}};
      const horizons = [1,2,5,10,20,50,100];

      html+=`<div class="fvg-card">
        <div class="fvg-header">
          <span class="fvg-type ${{f.fvg_type}}">${{f.fvg_type==='bullish'?'▲ BULLISH':'▼ BEARISH'}}</span>
          <span class="urgency-tag ${{f.urgency||'unknown'}}">${{(f.urgency||'?').toUpperCase()}}</span>
        </div>
        <div class="fvg-range">${{f.gap_low}} — ${{f.gap_high}} &nbsp;·&nbsp; ${{f.age_candles||'?'}}h ago</div>`;

      if(Object.keys(probs).length) {{
        horizons.forEach(h => {{
          const p = probs[h];
          if(p === undefined) return;
          const pct = Math.round(p * 100);
          const cls = urgencyColor(p);
          html+=`<div class="prob-row">
            <span class="prob-label">${{h}} candle${{h>1?'s':''}}</span>
            <div class="prob-bar"><div class="prob-fill ${{cls}}" style="width:${{pct}}%"></div></div>
            <span class="prob-val">${{pct}}%</span>
          </div>`;
        }});
      }} else {{
        html+=`<div class="empty" style="padding:8px;font-size:0.8rem;">Model not loaded</div>`;
      }}
      html+=`</div>`;
    }});
    html+=`</div>`;
  }}

  document.getElementById('side').innerHTML = html;
}}

load().then(d => {{ renderChart(d); renderSide(d); }});
setInterval(async () => {{ const d = await load(); renderChart(d); renderSide(d); }}, 60000);
</script>
</body></html>'''


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/data':
            qs = parse_qs(parsed.query)
            sym = qs.get('pair', ['EURUSD'])[0].upper()
            if sym not in PAIRS: sym = 'EURUSD'
            try:
                data = build_chart_data(sym)
            except Exception as e:
                data = {{'error': str(e)}}
            body = json.dumps(data, default=str).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(body)
        else:
            qs = parse_qs(parsed.query)
            sym = qs.get('pair', ['EURUSD'])[0].upper()
            if sym not in PAIRS: sym = 'EURUSD'
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(_build_html(sym).encode())

    def log_message(self, fmt, *args):
        pass


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
