"""Solve modes: nodal, zonal, and zonal-nodal on small synthetic networks."""
import json, os, subprocess, sys
import numpy as np, pandas as pd, pytest
import open_pcm as P
from open_pcm.zonal import aggregate, zone_map

CASE=os.path.join(os.path.dirname(__file__),'..','examples','synthetic_4node')
HOURS=range(3)

def write_case(path, nodes, gens, branches, cfg, load=1.0):
    os.makedirs(path, exist_ok=True)
    pd.DataFrame(dict(hour=np.arange(8760), load=np.full(8760, float(load)))).to_csv(
        os.path.join(path,'profiles.csv'), index=False)
    pd.DataFrame(nodes).to_csv(os.path.join(path,'nodes.csv'), index=False)
    pd.DataFrame(gens).to_csv(os.path.join(path,'generators.csv'), index=False)
    pd.DataFrame(branches).to_csv(os.path.join(path,'branches.csv'), index=False)
    with open(os.path.join(path,'case.json'),'w') as f: json.dump(cfg, f)
    return path

def thermal(name, node, mw, cost):
    # cost = heat_rate * fuel, vom 0. heat_rate 10 keeps the arithmetic obvious.
    return dict(name=name, node=node, type='thermal', mw=mw, heat_rate=10,
                fuel_usd_mmbtu=cost/10, vom=0, profile='')

def base_cfg(**extra):
    cfg=dict(base_year=2026, ref_node=1, voll_usd_mwh=1500, demand_growth=0,
             line_loss_frac=0, flow_cost_usd_mwh=0,
             description='SYNTHETIC mode test. Not a real power system.')
    cfg.update(extra)
    return cfg

def branch(name, a, b, mva, loss=0, x=0.1):
    return dict(name=name, **{'from':a, 'to':b}, MVA=mva, x_pu=x, loss_pct=loss)

def open_network(path):
    """Four buses, two zones, no congestion, no losses, no flow cost."""
    return write_case(path,
        [dict(node=1, name='A1', peak_MW=100, zone='A'),
         dict(node=2, name='A2', peak_MW=100, zone='A'),
         dict(node=3, name='B1', peak_MW=100, zone='B'),
         dict(node=4, name='B2', peak_MW=100, zone='B')],
        [thermal('Cheap', 1, 800, 40), thermal('Mid', 3, 400, 70)],
        [branch('A',1,2,1e6), branch('B',3,4,1e6), branch('T1',2,3,1e6), branch('T2',1,4,1e6)],
        base_cfg())

def tight_network(path):
    """Zone N cannot physically support the export the zonal market schedules."""
    return write_case(path,
        [dict(node=1, name='N1', peak_MW=0, zone='N'),
         dict(node=2, name='N2', peak_MW=50, zone='N'),
         dict(node=3, name='S1', peak_MW=80, zone='S')],
        [thermal('Cheap', 1, 200, 20), thermal('Dear', 3, 200, 100)],
        [branch('internal',1,2,5), branch('tie',2,3,200)],
        base_cfg(flow_cost_usd_mwh=0.01))

def mwh(res):
    return res['gen'].set_index('name').MWh

@pytest.fixture(scope='module')
def example():
    return P.Case(CASE)

def test_nodal_results_unchanged(example):
    # Column and row counts of the 48-hour nodal LP on the 4-node example.
    built=P.build(example, np.arange(48), year=2026)
    assert built['short'] is None and built['spill'] is None
    assert (built['lp'].n, built['lp'].m)==(1008, 432)
    r=P.dispatch(example, 2026, range(48), log=lambda s: None)
    same=P.dispatch(example, 2026, range(48), mode='nodal', log=lambda s: None)
    pd.testing.assert_frame_equal(r['price'], same['price'])
    g=mwh(r)
    assert r['mode']=='nodal'
    assert g['Gas CC A']==pytest.approx(21600, abs=1e-3)
    assert g['Gas peaker B']==pytest.approx(9600, abs=1e-3)
    assert g['Oil engines C']==pytest.approx(5760, abs=1e-3)
    assert g['Solar E']==pytest.approx(1126.970, abs=1e-2)
    assert g['Wind F']==pytest.approx(2175.984, abs=1e-2)

def test_example_runs_all_modes(example):
    for mode in ('nodal','zonal','zonal-nodal'):
        r=P.dispatch(example, 2026, range(4), mode=mode, log=lambda s: None)
        assert r['mode']==mode and len(r['price'])==4
    z=P.dispatch(example, 2026, range(1), mode='zonal', log=lambda s: None)
    assert set(z['zones'].zone)=={'NE','SW'}
    tr=z['transfers'].set_index('name').loc['NE-SW']
    assert tr.frm=='NE' and tr.to=='SW'
    assert tr.mva==pytest.approx(400)
    assert tr.loss_pct==pytest.approx(1.625)

def test_uncongested_modes_agree(tmp_path):
    case=P.Case(open_network(tmp_path/'open'))
    nodal=P.dispatch(case, 2026, HOURS, mode='nodal', log=lambda s: None)
    zonal=P.dispatch(case, 2026, HOURS, mode='zonal', log=lambda s: None)
    both=P.dispatch(case, 2026, HOURS, mode='zonal-nodal', log=lambda s: None)
    # 400 MW at $40, three hours. The $70 unit stays off. No losses, so exports net to zero.
    for res in (nodal, zonal, both):
        g=mwh(res)
        assert g['Cheap']==pytest.approx(1200, abs=1e-3)
        assert g['Mid']==pytest.approx(0, abs=1e-3)
    np.testing.assert_allclose(nodal['price'].drop(columns='hour').values, 40, atol=1e-4)
    np.testing.assert_allclose(zonal['price'].drop(columns='hour').values, 40, atol=1e-4)
    npf=zonal['net_position']
    np.testing.assert_allclose(npf.loc[npf.zone=='A','net_export_MW'], 200, atol=1e-3)
    np.testing.assert_allclose(npf.loc[npf.zone=='B','net_export_MW'], -200, atol=1e-3)
    assert both['imbalance'].short_MW.sum()==pytest.approx(0, abs=1e-3)
    assert both['imbalance'].spill_MW.sum()==pytest.approx(0, abs=1e-3)
    assert both['imbalance'].penalty_usd.sum()==pytest.approx(0, abs=1e-2)

def test_transfer_limit_is_sum_of_tie_ratings(tmp_path):
    path=write_case(tmp_path/'cap',
        [dict(node=1, name='A', peak_MW=0, zone='A'),
         dict(node=2, name='B', peak_MW=150, zone='B')],
        [thermal('Cheap', 1, 500, 10), thermal('Dear', 2, 500, 100)],
        [branch('one',1,2,40), branch('two',2,1,60)],
        base_cfg(flow_cost_usd_mwh=0.01))
    case=P.Case(path)
    view, meta=aggregate(case)
    assert len(meta['ties'])==1 and meta['ties'][0]['mva']==pytest.approx(100)
    r=P.dispatch(case, 2026, range(1), mode='zonal', log=lambda s: None)
    assert mwh(r)['Cheap']==pytest.approx(100, abs=1e-3)
    assert mwh(r)['Dear']==pytest.approx(50, abs=1e-3)

def test_loss_percent_is_rating_weighted(tmp_path):
    path=write_case(tmp_path/'loss',
        [dict(node=1, name='A', peak_MW=10, zone='A'),
         dict(node=2, name='B', peak_MW=10, zone='B')],
        [thermal('G', 1, 50, 20)],
        [branch('fat',1,2,100, loss=2), branch('thin',1,2,50, loss=8)],
        base_cfg())
    r=P.dispatch(P.Case(path), 2026, range(1), mode='zonal', log=lambda s: None)
    tr=r['transfers'].iloc[0]
    assert tr.mva==pytest.approx(150)
    assert tr.loss_pct==pytest.approx(4)

def test_imbalance_stage_reports_slack_and_penalty(tmp_path):
    case=P.Case(tight_network(tmp_path/'tight'))
    zonal=P.dispatch(case, 2026, range(2), mode='zonal', log=lambda s: None)
    nod=P.dispatch(case, 2026, range(2), mode='nodal', log=lambda s: None)
    r=P.dispatch(case, 2026, range(2), mode='zonal-nodal', workers=2, log=lambda s: None)
    # The zonal market ignores the 5 MW line inside N and exports 80 MW.
    ex=zonal['net_position']
    np.testing.assert_allclose(ex.loc[ex.zone=='N','net_export_MW'], 80, atol=1e-3)
    # Nodal sees that line, so the cheap plant cannot serve the southern load.
    assert mwh(nod)['Cheap']==pytest.approx(10, abs=1e-2)
    assert mwh(nod)['Dear']==pytest.approx(250, abs=1e-2)
    imb=r['imbalance']
    assert {'short_MW','spill_MW','balancing_up_MW','balancing_down_MW','penalty_usd'}<=set(imb.columns)
    n=imb[imb.zone=='N']
    s=imb[imb.zone=='S']
    np.testing.assert_allclose(n.short_MW, 75, atol=1e-2)
    np.testing.assert_allclose(n.spill_MW, 0, atol=1e-3)
    np.testing.assert_allclose(s.short_MW, 0, atol=1e-3)
    # Stage C keeps the 5 MW that can actually cross and prices the other 75 in S.
    np.testing.assert_allclose(s.balancing_up_MW, 75, atol=1e-2)
    np.testing.assert_allclose(n.balancing_up_MW, 0, atol=1e-2)
    assert imb.penalty_usd.sum()==pytest.approx(2*75*1500, rel=1e-4)
    flows=r['balancing_flows']
    np.testing.assert_allclose(flows.scheduled_fwd_MW, 80, atol=1e-2)
    np.testing.assert_allclose(flows.fwd_MW, 5, atol=1e-2)
    again=P.dispatch(case, 2026, range(2), mode='zonal-nodal', workers=1, log=lambda s: None)
    pd.testing.assert_frame_equal(r['imbalance'], again['imbalance'], check_exact=False, atol=1e-6, rtol=1e-6)

def test_zone_column_overrides_map(tmp_path):
    path=write_case(tmp_path/'map',
        [dict(node=1, name='A', peak_MW=10, zone='North'),
         dict(node=2, name='B', peak_MW=10, zone='')],
        [thermal('G', 1, 30, 25)],
        [branch('L',1,2,50)],
        base_cfg(zones={'1':'South','2':'South'}))
    case=P.Case(path)
    assert zone_map(case)=={1:'North', 2:'South'}
    r=P.dispatch(case, 2026, range(1), mode='zonal', log=lambda s: None)
    assert set(r['price'].columns)=={'hour','North','South'}
    assert set(r['gen'].zone)=={'North'}

def test_missing_zone_and_bad_mode(tmp_path, example):
    path=write_case(tmp_path/'bare',
        [dict(node=1, name='A', peak_MW=10)],
        [thermal('G', 1, 20, 30)],
        [branch('L',1,1,10)],
        base_cfg())
    with pytest.raises(ValueError, match='zone'):
        P.dispatch(P.Case(path), 2026, range(1), mode='zonal', log=lambda s: None)
    with pytest.raises(ValueError, match='mode'):
        P.dispatch(example, 2026, range(1), mode='copper', log=lambda s: None)

def test_config_mode_and_cli_flag(tmp_path):
    path=open_network(tmp_path/'cli')
    with open(os.path.join(path,'case.json')) as f: cfg=json.load(f)
    cfg['mode']='zonal'
    with open(os.path.join(path,'case.json'),'w') as f: json.dump(cfg, f)
    case=P.Case(path)
    assert P.dispatch(case, 2026, range(1), log=lambda s: None)['mode']=='zonal'
    out=tmp_path/'out'
    subprocess.check_call([sys.executable,'-m','open_pcm', path, str(out),
                           '--year','2026','--hours','0','1','--mode','nodal'],
                          cwd=os.path.join(os.path.dirname(__file__),'..'))
    price=pd.read_csv(os.path.join(out,'prices_hourly_2026.csv'))
    assert [str(c) for c in price.columns]==['hour','1','2','3','4']
    assert not os.path.exists(os.path.join(out,'net_position_2026.csv'))
    out2=tmp_path/'out2'
    cfg['mode']='zonal-nodal'
    with open(os.path.join(path,'case.json'),'w') as f: json.dump(cfg, f)
    subprocess.check_call([sys.executable,'-m','open_pcm', path, str(out2),
                           '--year','2026','--hours','0','0'],
                          cwd=os.path.join(os.path.dirname(__file__),'..'))
    imb=pd.read_csv(os.path.join(out2,'imbalance_2026.csv'))
    assert 'penalty_usd' in imb.columns
    assert os.path.exists(os.path.join(out2,'balancing_flows_2026.csv'))
    assert os.path.exists(os.path.join(out2,'zonal_prices_2026.csv'))
