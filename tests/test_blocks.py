"""Consecutive blocks, storage carried across a join, and the time-limit gap rule."""
import json, os, subprocess, sys
import numpy as np, pandas as pd, pytest
import open_pcm as P
from open_pcm.lp import LP, check_status, reported_figure
from open_pcm.model import terminal_storage

CASE=os.path.join(os.path.dirname(__file__),'..','examples','synthetic_4node')
QUIET=lambda s: None

def write_case(path, nodes, gens, storage, load=1.0):
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(dict(hour=np.arange(8760), load=np.full(8760, float(load)))).to_csv(
        os.path.join(path,'profiles.csv'), index=False)
    pd.DataFrame(nodes).to_csv(os.path.join(path,'nodes.csv'), index=False)
    pd.DataFrame(gens).to_csv(os.path.join(path,'generators.csv'), index=False)
    pd.DataFrame(columns=['name','from','to','MVA','x_pu','loss_pct']).to_csv(
        os.path.join(path,'branches.csv'), index=False)
    if storage is not None:
        pd.DataFrame(storage).to_csv(os.path.join(path,'storage.csv'), index=False)
    cfg=dict(base_year=2026, ref_node=1, voll_usd_mwh=1500, demand_growth=0,
             line_loss_frac=0, flow_cost_usd_mwh=0,
             description='SYNTHETIC block test. Not a real power system.')
    with open(os.path.join(path,'case.json'),'w') as f: json.dump(cfg, f)
    return path

def tank_case(path):
    """10 MW load, 10 MWh of stored energy, a $50 thermal unit behind it."""
    return write_case(path,
        [dict(node=1, name='A', peak_MW=10, zone='A')],
        [dict(name='Thermal', node=1, type='thermal', mw=10, heat_rate=10,
              fuel_usd_mmbtu=5, vom=0, profile='')],
        [dict(name='Tank', node=1, mw=10, mwh=10, efficiency=1, initial_mwh=10, vom=1)])

def close(a, b, atol):
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True),
                                  check_dtype=False, check_exact=False, atol=atol, rtol=1e-5)

@pytest.fixture(scope='module')
def example():
    return P.Case(CASE)

def test_split_matches_unsplit(example):
    hours=range(8)
    whole=P.dispatch(example, 2026, hours, log=QUIET)
    by_len=P.dispatch(example, 2026, hours, blocks=3, log=QUIET)
    by_list=P.dispatch(example, 2026, hours, blocks=(2,2,2,2), log=QUIET)
    assert 'meta' not in whole and 'storage' not in whole
    assert by_len['meta']['blocks']==[3,3,2] and by_len['meta']['flagged'] is False
    assert by_list['meta']['status']=='Optimal'
    for part in (by_len, by_list):
        close(whole['price'], part['price'], 1e-4)
        close(whole['gen'], part['gen'], 1e-3)
        close(whole['use'], part['use'], 1e-3)
        close(whole['flows'], part['flows'], 1e-3)
        assert list(part['price'].hour)==list(hours)

def test_zonal_blocks_match_unsplit(example):
    hours=range(4)
    for mode in ('zonal','zonal-nodal'):
        whole=P.dispatch(example, 2026, hours, mode=mode, log=QUIET)
        part=P.dispatch(example, 2026, hours, mode=mode, blocks=(2,2), log=QUIET)
        close(whole['price'], part['price'], 1e-3)
        close(whole['gen'], part['gen'], 1e-2)
        close(whole['net_position'], part['net_position'], 1e-2)
        assert part['mode']==mode and part['meta']['blocks']==[2,2]
    whole=P.dispatch(example, 2026, range(2), mode='zonal-nodal', log=QUIET)
    both=P.dispatch(example, 2026, range(2), mode='zonal-nodal', blocks=(1,1), log=QUIET)
    close(whole['imbalance'], both['imbalance'], 1e-2)
    close(whole['balancing_flows'], both['balancing_flows'], 1e-2)
    close(whole['zonal']['price'], both['zonal']['price'], 1e-3)

def test_storage_continues_across_the_join(tmp_path):
    case=P.Case(tank_case(tmp_path/'tank'))
    whole=P.dispatch(case, 2026, range(2), log=QUIET)
    split=P.dispatch(case, 2026, range(2), blocks=(1,1), log=QUIET)
    # 20 MWh of load, 10 MWh in the tank, so 10 MWh comes from the thermal unit.
    for res in (whole, split):
        assert res['gen'].set_index('name').MWh['Thermal']==pytest.approx(10, abs=1e-3)
        st=res['storage']
        assert st.discharge_mw.sum()==pytest.approx(10, abs=1e-3)
    st=split['storage'].set_index('hour')
    # The first hour uses the tank. The second hour has to see an empty tank, not the original 10 MWh.
    assert st.loc[0,'discharge_mw']==pytest.approx(10, abs=1e-3)
    assert st.loc[0,'soc_mwh']==pytest.approx(0, abs=1e-3)
    assert st.loc[1,'discharge_mw']==pytest.approx(0, abs=1e-3)
    assert st.loc[1,'soc_mwh']==pytest.approx(0, abs=1e-3)
    eta=1.0
    assert st.loc[0,'soc_mwh']==pytest.approx(10+eta*st.loc[0,'charge_mw']-st.loc[0,'discharge_mw'], abs=1e-4)
    assert st.loc[1,'soc_mwh']==pytest.approx(
        st.loc[0,'soc_mwh']+eta*st.loc[1,'charge_mw']-st.loc[1,'discharge_mw'], abs=1e-4)
    # The level handed to a fresh solve of hour 1 matches that hour of the blocked run.
    m0=P.solve(P.build(case, [0], year=2026))
    end=terminal_storage(m0)
    assert end['Tank']==pytest.approx(st.loc[0,'soc_mwh'], abs=1e-4)
    m1=P.solve(P.build(case, [1], year=2026, storage_state=end))
    got=P.extract(m1)['storage'].iloc[0]
    assert got.soc_mwh==pytest.approx(st.loc[1,'soc_mwh'], abs=1e-4)
    assert got.discharge_mw==pytest.approx(st.loc[1,'discharge_mw'], abs=1e-4)

def test_zonal_storage_also_carries(tmp_path):
    case=P.Case(tank_case(tmp_path/'ztank'))
    split=P.dispatch(case, 2026, range(2), mode='zonal', blocks=(1,1), log=QUIET)
    st=split['storage'].set_index('hour')
    assert st.loc[0,'discharge_mw']==pytest.approx(10, abs=1e-3)
    assert st.loc[1,'discharge_mw']==pytest.approx(0, abs=1e-3)
    with pytest.raises(ValueError, match='storage'):
        P.dispatch(case, 2026, range(2), mode='zonal-nodal', blocks=(1,1), log=QUIET)

def test_bad_blocks_and_storage(example, tmp_path):
    with pytest.raises(ValueError, match='sum'):
        P.dispatch(example, 2026, range(5), blocks=(2,2), log=QUIET)
    with pytest.raises(ValueError, match='positive'):
        P.dispatch(example, 2026, range(4), blocks=0, log=QUIET)
    case=P.Case(tank_case(tmp_path/'gap'))
    with pytest.raises(ValueError, match='contiguous'):
        P.dispatch(case, 2026, [0,2], blocks=(1,1), log=QUIET)
    path=tank_case(tmp_path/'bad')
    stor=pd.read_csv(os.path.join(path,'storage.csv'))
    stor['efficiency']=1.5
    stor.to_csv(os.path.join(path,'storage.csv'), index=False)
    with pytest.raises(ValueError, match='efficiency'):
        P.dispatch(P.Case(path), 2026, range(1), log=QUIET)

def test_reported_figure_and_threshold():
    assert reported_figure(0.2, 1e-8, True)==pytest.approx(0.2)
    assert reported_figure(np.inf, 1e-5, True)==pytest.approx(1e-5)
    assert reported_figure(np.inf, 0.0, False) is None
    ok=dict(gap=None, objective_error=1e-4, feasible=True, reported=1e-4)
    meta=check_status('Time limit reached', ok, 1e-4, 'dispatch')
    assert meta['flagged'] is True and meta['status']=='Time limit reached'
    assert meta['objective_error']==pytest.approx(1e-4)
    with pytest.raises(RuntimeError, match='Time limit'):
        check_status('Time limit reached', dict(ok, reported=1e-3, objective_error=1e-3), 1e-4, 'dispatch')
    with pytest.raises(RuntimeError, match='Time limit'):
        check_status('Time limit reached', dict(gap=None, objective_error=0.0, feasible=False, reported=None), 1.0, 'dispatch')
    with pytest.raises(RuntimeError, match='Infeasible'):
        check_status('Infeasible', dict(gap=None, objective_error=None, feasible=False, reported=None), 1.0, 'dispatch')
    plain=check_status('Optimal', dict(gap=None, objective_error=0.0, feasible=True, reported=0.0), None, 'dispatch')
    assert plain['flagged'] is False
    with pytest.raises(ValueError, match='accept_gap'):
        check_status('Time limit reached', ok, -1, 'dispatch')

def test_optimal_solve_is_not_flagged(example):
    base=P.dispatch(example, 2026, range(2), log=QUIET)
    held=P.dispatch(example, 2026, range(2), time_limit=60, accept_gap=1e-4, log=QUIET)
    assert held['meta']['flagged'] is False and held['meta']['status']=='Optimal'
    assert held['meta']['gap'] is None
    assert held['meta']['objective_error']==pytest.approx(0, abs=1e-6)
    close(base['price'], held['price'], 1e-6)
    assert 'meta' not in base

def test_time_limit_without_a_solution_is_rejected(example):
    with pytest.raises(RuntimeError, match='Time limit'):
        P.dispatch(example, 2026, range(24), time_limit=0, accept_gap=1, log=QUIET)

def test_dispatch_flags_a_close_time_limit(monkeypatch, example):
    real=LP.solve
    def wrapped(self, method='ipm', tol=None, threads=None, time_limit=None, accept_gap=None):
        st,x,dual=real(self, method)
        self.report=dict(gap=None, objective_error=2e-5, feasible=True, reported=2e-5)
        return 'Time limit reached', x, dual
    monkeypatch.setattr(LP, 'solve', wrapped)
    kept=P.dispatch(example, 2026, range(2), accept_gap=1e-4, log=QUIET)
    assert kept['meta']['flagged'] is True
    assert kept['meta']['status']=='Time limit reached'
    assert kept['meta']['objective_error']==pytest.approx(2e-5)
    assert len(kept['price'])==2
    with pytest.raises(RuntimeError, match='Time limit'):
        P.dispatch(example, 2026, range(2), accept_gap=1e-6, log=QUIET)
    with pytest.raises(RuntimeError, match='Time limit'):
        P.dispatch(example, 2026, range(2), log=QUIET)

def test_cli_blocks_and_meta(tmp_path):
    out=tmp_path/'out'
    subprocess.check_call([sys.executable,'-m','open_pcm', CASE, str(out),
                           '--year','2026','--hours','0','5','--blocks','2'],
                          cwd=os.path.join(os.path.dirname(__file__),'..'))
    price=pd.read_csv(os.path.join(out,'prices_hourly_2026.csv'))
    assert list(price.hour)==[0,1,2,3,4,5]
    meta=json.load(open(os.path.join(out,'meta.json')))
    assert meta['flagged'] is False and meta['blocks']==[2,2,2]
    assert not os.path.exists(os.path.join(out,'storage_2026.csv'))
