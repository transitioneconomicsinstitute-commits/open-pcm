"""Minimal local web UI for OPEN-PCM.

Serves a single page on the loopback interface, loads an example case,
runs dispatch, and shows a summary table plus a simple hourly price chart.
No third-party web framework is used: the server is Python's stdlib
http.server. Only cases under the packaged examples/ directory can be loaded.
"""
from __future__ import annotations

import argparse, json, os, sys, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from . import Case, CaseError, __version__, dispatch
from .run import MODES

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / 'examples'
PAGE = Path(__file__).with_name('gui_page.html')
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 8760
MAX_HOURS = 8760

def example_names():
    """Example folders that contain a case.json, in name order."""
    names = []
    if EXAMPLES.is_dir():
        for p in sorted(EXAMPLES.iterdir()):
            if p.is_dir() and (p / 'case.json').is_file():
                names.append(p.name)
    return names

def resolve_case(name):
    """Resolve a case name to an examples/ folder. Rejects path traversal."""
    if not name or not isinstance(name, str):
        raise ValueError('choose an example case')
    if name in ('.', '..') or '/' in name or '\\' in name or os.path.sep in name:
        raise ValueError('case must be an example folder name')
    folder = (EXAMPLES / name).resolve()
    try:
        folder.relative_to(EXAMPLES.resolve())
    except ValueError:
        raise ValueError('case must be an example folder name')
    if not (folder / 'case.json').is_file():
        raise ValueError(f'no example case named {name!r}')
    return folder

def _int(v, field):
    try:
        return int(v)
    except (TypeError, ValueError):
        raise ValueError(f'{field} must be an integer')

def parse_run(body):
    """Read and check a run request. Returns kwargs for run_case."""
    if not isinstance(body, dict):
        raise ValueError('request must be a JSON object')
    case_name = body.get('case') or body.get('example')
    year = _int(body.get('year'), 'year')
    start = _int(body.get('start', 0), 'start')
    end = _int(body.get('end', 167), 'end')
    mode = body.get('mode') or None
    if mode is not None:
        mode = str(mode).strip().lower()
        if mode not in MODES:
            raise ValueError('mode must be one of ' + ', '.join(MODES))
    if start < 0 or end > 8759 or start > end:
        raise ValueError('hours must satisfy 0 <= start <= end <= 8759')
    if end - start + 1 > MAX_HOURS:
        raise ValueError(f'a run can cover at most {MAX_HOURS} hours')
    return dict(case_name=case_name, year=year, start=start, end=end, mode=mode)

def run_case(case_name, year, start, end, mode=None):
    """Solve the chosen example and return JSON-ready summary and chart series."""
    folder = resolve_case(case_name)
    hours = range(start, end + 1)
    result = dispatch(Case(folder), year, hours, mode=mode, log=lambda _s: None)
    return summarise(result, case_name=case_name, year=year, start=start, end=end)

def summarise(result, case_name, year, start, end):
    price = result['price']
    gen = result['gen']
    use = result['use']
    price_cols = [c for c in price.columns if c != 'hour']
    hour = [int(h) for h in price['hour'].tolist()]
    # Mean nodal (or zonal) price each hour, for a single overview trace.
    mean_price = price[price_cols].mean(axis=1).astype(float).tolist() if price_cols else [0.0] * len(hour)
    traces = {str(c): [float(v) for v in price[c].tolist()] for c in price_cols}
    total_gen = float(gen['MWh'].sum()) if len(gen) else 0.0
    total_use = float(use['unserved_MWh'].sum()) if len(use) else 0.0
    total_load = float(use['load_MWh'].sum()) if len(use) else 0.0
    by_unit = [
        dict(name=str(r['name']), node=int(r['node']) if 'node' in r and r['node']==r['node'] else r.get('node'),
             type=str(r['type']), MWh=round(float(r['MWh']), 3))
        for r in gen.to_dict('records')
    ]
    meta = result.get('meta') or {}
    status = meta.get('status') or 'Optimal'
    out = dict(
        case=case_name,
        year=int(year),
        start=int(start),
        end=int(end),
        hours=int(end) - int(start) + 1,
        mode=result.get('mode'),
        status=status,
        flagged=bool(meta.get('flagged', False)),
        load_MWh=round(total_load, 3),
        generation_MWh=round(total_gen, 3),
        unserved_MWh=round(total_use, 3),
        price_mean_usd_mwh=round(float(sum(mean_price) / len(mean_price)), 3) if mean_price else None,
        price_min_usd_mwh=round(float(min(mean_price)), 3) if mean_price else None,
        price_max_usd_mwh=round(float(max(mean_price)), 3) if mean_price else None,
        generation=by_unit,
        chart=dict(hour=hour, mean_price_usd_mwh=mean_price, series=traces),
    )
    if 'zone' in gen.columns:
        out['generation'] = [
            dict(row, zone=str(r['zone'])) for row, r in zip(out['generation'], gen.to_dict('records'))
        ]
    return out

class Handler(BaseHTTPRequestHandler):
    server_version = 'OPEN-PCM-GUI/' + __version__

    def log_message(self, fmt, *args):
        sys.stderr.write('%s - %s\n' % (self.address_string(), fmt % args))

    def _send(self, status, body, content_type='application/json; charset=utf-8'):
        data = body if isinstance(body, (bytes, bytearray)) else body.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(data)))
        self.send_header('Cache-Control', 'no-store')
        self.end_headers()
        self.wfile.write(data)

    def _json(self, status, obj):
        self._send(status, json.dumps(obj, allow_nan=False), 'application/json; charset=utf-8')

    def do_GET(self):
        path = urlparse(self.path).path
        if path in ('/', '/index.html'):
            self._send(200, PAGE.read_text(encoding='utf-8'), 'text/html; charset=utf-8')
            return
        if path == '/api/examples':
            self._json(200, dict(examples=example_names(), modes=list(MODES), version=__version__))
            return
        self._json(404, dict(error='not found'))

    def do_POST(self):
        path = urlparse(self.path).path
        if path != '/api/run':
            self._json(404, dict(error='not found'))
            return
        length = int(self.headers.get('Content-Length') or 0)
        if length > 1_000_000:
            self._json(400, dict(error='request too large'))
            return
        raw = self.rfile.read(length) if length else b'{}'
        try:
            body = json.loads(raw.decode('utf-8') or '{}')
            self._json(200, run_case(**parse_run(body)))
        except (ValueError, CaseError, json.JSONDecodeError) as e:
            self._json(400, dict(error=str(e)))
        except Exception:
            traceback.print_exc()
            self._json(500, dict(error='solve failed'))

def make_server(host=DEFAULT_HOST, port=DEFAULT_PORT):
    if not PAGE.is_file():
        raise FileNotFoundError('gui page is missing: ' + str(PAGE))
    return ThreadingHTTPServer((host, port), Handler)

def serve(host=DEFAULT_HOST, port=DEFAULT_PORT, ready=None):
    httpd = make_server(host, port)
    if ready is not None:
        ready(httpd)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
    return httpd

def main(argv=None):
    p = argparse.ArgumentParser(prog='open-pcm-gui',
        description='Open a local web page to run an OPEN-PCM example and view the results.')
    p.add_argument('--host', default=DEFAULT_HOST, help='bind address (default 127.0.0.1)')
    p.add_argument('--port', type=int, default=DEFAULT_PORT, help='port (default 8760)')
    a = p.parse_args(argv)
    names = example_names()
    print(f'OPEN-PCM {__version__} GUI  http://{a.host}:{a.port}/')
    if names:
        print('examples: ' + ', '.join(names))
    serve(a.host, a.port)

if __name__ == '__main__':
    main()
