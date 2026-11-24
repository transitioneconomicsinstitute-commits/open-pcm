import json, threading, urllib.error, urllib.request
import pytest
import open_pcm as P
from open_pcm import gui

CASE = 'synthetic_4node'

@pytest.fixture(scope='module')
def server():
    httpd = gui.make_server('127.0.0.1', 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    yield f'http://{host}:{port}'
    httpd.shutdown()
    httpd.server_close()

def get(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return r.status, r.headers.get_content_type(), r.read()

def post(url, obj):
    data = json.dumps(obj).encode()
    req = urllib.request.Request(url, data=data, headers={'Content-Type':'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())

def test_example_names():
    assert CASE in gui.example_names()

def test_resolve_rejects_traversal():
    with pytest.raises(ValueError):
        gui.resolve_case('../synthetic_4node')
    with pytest.raises(ValueError):
        gui.resolve_case('synthetic_4node/../../open_pcm')
    with pytest.raises(ValueError):
        gui.resolve_case('/tmp')

def test_page(server):
    status, typ, body = get(server + '/')
    assert status == 200 and typ == 'text/html'
    text = body.decode()
    assert 'OPEN-PCM' in text and '/api/run' in text
    assert 'id="series"' in text

def test_examples_api(server):
    status, typ, body = get(server + '/api/examples')
    assert status == 200
    data = json.loads(body.decode())
    assert CASE in data['examples']
    assert data['modes'] == ['nodal', 'zonal', 'zonal-nodal']

def test_run_matches_dispatch(server):
    status, out = post(server + '/api/run', dict(case=CASE, year=2027, start=0, end=23, mode='nodal'))
    assert status == 200
    assert out['status'] == 'Optimal'
    assert out['mode'] == 'nodal'
    assert out['hours'] == 24 and len(out['chart']['hour']) == 24
    assert len(out['chart']['mean_price_usd_mwh']) == 24
    assert out['chart']['series'] and all(len(v) == 24 for v in out['chart']['series'].values())
    assert out['generation_MWh'] > 0
    assert out['unserved_MWh'] >= 0
    direct = P.dispatch(P.Case(gui.resolve_case(CASE)), 2027, range(24), log=lambda s: None)
    assert out['generation_MWh'] == pytest.approx(float(direct['gen']['MWh'].sum()), rel=1e-6, abs=1e-6)
    assert out['load_MWh'] == pytest.approx(float(direct['use']['load_MWh'].sum()), rel=1e-6, abs=1e-6)
    assert out['unserved_MWh'] == pytest.approx(float(direct['use']['unserved_MWh'].sum()), rel=1e-6, abs=1e-6)

def test_run_rejects_bad_case(server):
    status, out = post(server + '/api/run', dict(case='../secret', year=2027, start=0, end=3, mode='nodal'))
    assert status == 400
    assert 'example' in out['error']

def test_cli_help():
    with pytest.raises(SystemExit) as e:
        gui.main(['--help'])
    assert e.value.code == 0
