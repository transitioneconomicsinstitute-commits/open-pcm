"""Zonal market, and the zonal-then-nodal sequence.

Zones come from a `zone` column on nodes.csv, or from the `zones` map in
case.json (bus id to zone name). A non-empty column value wins; the map
fills blanks.

The zonal market is a transport model: one bus per zone, internal branches
dropped, and one link between each pair of zones. The link rating is the
sum of the tie-line MVAs. Its loss percent is the rating-weighted average.

Zonal-nodal then does three things:
  A. the zonal solve, which sets net positions and inter-zone flows
  B. a nodal DC solve inside each zone, with those flows imposed as fixed
     injections at the border buses (zones run on a thread pool)
  C. a border balance that moves the inter-zone flows within their ratings
     and covers anything still open with penalized slack
"""
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from .case import Case
from .lp import LP
from .model import build, solve, extract

class Slice(Case):
    """In-memory case. Load growth matches Case."""
    def __init__(s, cfg, nodes, generators, branches, profiles):
        s.path=None
        s.cfg=dict(cfg)
        s.nodes=nodes.reset_index(drop=True).copy()
        s.generators=generators.reset_index(drop=True).copy()
        s.branches=branches.reset_index(drop=True).copy()
        s.profiles=profiles.copy()

def zone_map(case):
    """bus id -> zone name. Raises if any bus has no zone."""
    nodes=case.nodes
    col=nodes['zone'] if 'zone' in nodes.columns else None
    raw=case.cfg.get('zones') or {}
    if not isinstance(raw, dict):
        raise ValueError('case.json zones must be an object of bus id to zone name')
    mapping={}
    for k,v in raw.items():
        if v is None or str(v).strip()=='': continue
        mapping[int(k)]=str(v).strip()
    out={}; missing=[]
    for i,n in enumerate(nodes.node.astype(int)):
        z=None
        if col is not None:
            v=col.iloc[i]
            if pd.notna(v) and str(v).strip()!='': z=str(v).strip()
        if z is None: z=mapping.get(int(n))
        if z is None: missing.append(int(n))
        else: out[int(n)]=z
    if missing:
        raise ValueError(
            'zonal solve needs a zone for every bus; missing '
            + ', '.join(str(n) for n in missing)
            + '. Add a zone column on nodes.csv or a zones map in case.json.')
    return out

def _loss_frac(row, cfg):
    return float(row.get('loss_pct', cfg['line_loss_frac']*100))/100

def aggregate(case):
    """Collapse the case to one bus per zone. Returns (Slice, meta)."""
    zmap=zone_map(case)
    zones=sorted(set(zmap.values()))
    zid={z:i+1 for i,z in enumerate(zones)}
    peak=defaultdict(float)
    for rec in case.nodes.to_dict('records'):
        peak[zmap[int(rec['node'])]]+=float(rec['peak_MW'])
    nodes=pd.DataFrame([dict(node=zid[z], name=z, peak_MW=peak[z]) for z in zones])
    gens=case.generators.copy()
    gens['node']=gens.node.astype(int).map(lambda n: zid[zmap[int(n)]])
    groups=defaultdict(list)
    for r in case.branches.to_dict('records'):
        u,v=int(r['from']),int(r['to'])
        za,zb=zmap[u],zmap[v]
        if za==zb: continue
        mva=float(r['MVA'])
        if mva<=0: continue
        groups[tuple(sorted((za,zb)))].append((u,v,mva,_loss_frac(r, case.cfg)))
    ties=[]
    for za,zb in sorted(groups):
        members=groups[(za,zb)]
        mva=sum(m[2] for m in members)
        # Match build(), which does float(loss_pct) / 100.
        loss_pct=sum(m[2]*m[3] for m in members)/mva*100
        ties.append(dict(name=f'{za}-{zb}', za=za, zb=zb, a=zid[za], b=zid[zb],
                          mva=mva, loss=float(loss_pct)/100, members=members))
    branches=pd.DataFrame([dict(name=t['name'], **{'from':t['a'], 'to':t['b']},
                                 MVA=t['mva'], x_pu=0.1, loss_pct=t['loss']*100) for t in ties])
    if branches.empty:
        branches=pd.DataFrame(columns=['name','from','to','MVA','x_pu','loss_pct'])
    cfg=dict(case.cfg)
    cfg['ref_node']=zid[zmap[int(case.cfg['ref_node'])]]
    meta=dict(zmap=zmap, zones=zones, zid=zid, ties=ties)
    return Slice(cfg, nodes, gens, branches, case.profiles), meta

def transfers_frame(meta):
    return pd.DataFrame([dict(name=t['name'], frm=t['za'], to=t['zb'],
                               mva=t['mva'], loss_pct=t['loss']*100) for t in meta['ties']])

def zones_frame(case, meta):
    zid=meta['zid']; zmap=meta['zmap']
    return pd.DataFrame([dict(node=int(n), zone=zmap[int(n)], zone_id=zid[zmap[int(n)]])
                         for n in case.nodes.node.astype(int)])

def _hourly_gen(model):
    """Generation (nodes, hours) aligned with model['nid']."""
    x=model['x']; nid=[int(b) for b in model['nid']]; T=len(model['hours'])
    gen=np.zeros((len(nid), T))
    index={b:i for i,b in enumerate(nid)}
    for u,idx in model['units']:
        gen[index[int(u['bus'])]]+=x[idx]
    return gen

def net_position_frame(model, meta):
    """Hourly zonal balance. Positive net_export_MW means generation above load."""
    gen=_hourly_gen(model); use=model['x'][model['use']]; load=model['L']
    idz={meta['zid'][z]:z for z in meta['zones']}
    rows=[]
    for i,bus in enumerate(int(b) for b in model['nid']):
        zone=idz[bus]
        for t,h in enumerate(model['hours']):
            g,u,L=float(gen[i,t]),float(use[i,t]),float(load[i,t])
            rows.append(dict(hour=int(h), zone=zone, load_MW=L, gen_MW=g,
                             unserved_MW=u, net_export_MW=g+u-L))
    return pd.DataFrame(rows)

def _label_zonal(result, meta):
    """Replace zone bus ids with zone names on the tables extract() returns."""
    idz={meta['zid'][z]:z for z in meta['zones']}
    price=result['price'].copy()
    price.columns=[idz.get(int(c), c) if c!='hour' else c for c in price.columns]
    result['price']=price
    for frame,col in ((result['gen'],'node'), (result['use'],'node')):
        frame['zone']=frame[col].astype(int).map(idz)
    if len(result['flows']):
        result['flows']=result['flows'].copy()
        result['flows']['frm']=result['flows']['frm'].astype(int).map(idz)
        result['flows']['to']=result['flows']['to'].astype(int).map(idz)
    return result

def _zonal_tables(model, case, meta):
    out=_label_zonal(extract(model), meta)
    out['net_position']=net_position_frame(model, meta)
    out['zones']=zones_frame(case, meta)
    out['transfers']=transfers_frame(meta)
    return out

def _optimal(status, what):
    if status!='Optimal': raise RuntimeError(f'{what}: {status}')

def penalty_price(case):
    """$/MWh on stage-C slack. Defaults to the value of lost load."""
    v=case.cfg.get('imbalance_penalty_usd_mwh')
    if v is None: return float(case.cfg['voll_usd_mwh'])
    return float(v)

def feasibility_penalty(case):
    """Stage B slack, priced above load shedding so it is a last resort."""
    return max(10*float(case.cfg['voll_usd_mwh']), 1e4)

def tie_receipts(model, meta):
    """Power received at each end of each aggregate tie, from a solved model.

    Positive means power arriving in that zone. With losses the two ends do
    not sum to zero.
    """
    x=model['x']
    by_name={l[3]:l for l in model['lines']}
    out=[]
    for tie in meta['ties']:
        line=by_name[tie['name']]
        fp,fm=x[line[4]],x[line[5]]
        loss=tie['loss']  # same fraction the zonal LP used
        rec_a=(1-loss)*fm-fp
        rec_b=(1-loss)*fp-fm
        out.append((tie, rec_a, rec_b, fp, fm))
    return out

def border_injections(case, meta, receipts, T):
    """Map aggregate tie receipts onto border buses, weighted by tie MVA."""
    inj={int(n):np.zeros(T) for n in case.nodes.node.astype(int)}
    for tie, rec_a, rec_b, fp, fm in receipts:
        wa,wb=defaultdict(float),defaultdict(float)
        zmap=meta['zmap']
        for u,v,mva,loss in tie['members']:
            if zmap[u]==tie['za']:
                wa[u]+=mva; wb[v]+=mva
            else:
                wa[v]+=mva; wb[u]+=mva
        if abs(sum(wa.values())-tie['mva'])>1e-6 or abs(sum(wb.values())-tie['mva'])>1e-6:
            raise RuntimeError(f'tie {tie["name"]} endpoints do not sum to its rating')
        for n,w in wa.items(): inj[n]+=rec_a*(w/tie['mva'])
        for n,w in wb.items(): inj[n]+=rec_b*(w/tie['mva'])
    return inj

def component_refs(nodes, branches, prefer):
    """One reference bus per connected component. `prefer` wins in its island."""
    ids=[int(n) for n in nodes.node.values]
    parent={n:n for n in ids}
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        a,b=find(a),find(b)
        if a!=b: parent[b]=a
    for r in branches.to_dict('records'):
        union(int(r['from']), int(r['to']))
    rep={}
    for n in ids: rep.setdefault(find(n), n)
    if prefer in parent: rep[find(prefer)]=prefer
    return list(rep.values())

def zone_slice(case, zone, zmap):
    """Nodal case for one zone: its buses, plants and internal branches."""
    keep=set(n for n,z in zmap.items() if z==zone)
    nodes=case.nodes[case.nodes.node.astype(int).isin(keep)].copy()
    gens=case.generators[case.generators.node.astype(int).isin(keep)].copy()
    br=case.branches
    if len(br):
        mask=[int(a) in keep and int(b) in keep for a,b in zip(br['from'], br['to'])]
        branches=br[mask].copy()
    else:
        branches=br.copy()
    cfg=dict(case.cfg)
    prefer=int(cfg['ref_node'])
    if prefer not in keep: prefer=int(nodes.node.iloc[0])
    cfg['ref_node']=prefer
    sl=Slice(cfg, nodes, gens, branches, case.profiles)
    return sl, component_refs(sl.nodes, sl.branches, prefer)

def _solve_zone(job):
    sl, refs, hours, year, method, inj, pen, zone=job
    m=solve(build(sl, hours, year=year, injections=inj, feas_penalty=pen, refs=refs), method)
    _optimal(m['status'], f'zonal-nodal stage B zone {zone}')
    x=m['x']
    return dict(zone=zone, result=extract(m),
                short=np.array(x[m['short']].sum(0), float),
                spill=np.array(x[m['spill']].sum(0), float))

def _pool_map(fn, jobs, workers):
    n=len(jobs)
    if n==0: return []
    if workers is None:
        w=min(n, os.cpu_count() or 1, 32)
    else:
        w=max(1, min(int(workers), n, 32))
    if w==1: return [fn(j) for j in jobs]
    with ThreadPoolExecutor(max_workers=w) as ex:
        return list(ex.map(fn, jobs))

def _stitch(case, parts):
    """Join per-zone nodal tables. Row order does not depend on which thread finished first."""
    price=None
    for p in parts:
        pr=p['result']['price']
        price=pr if price is None else price.merge(pr, on='hour')
    rename={c:int(c) for c in price.columns if c!='hour'}
    price=price.rename(columns=rename)
    price=price[['hour']+[int(n) for n in case.nodes.node.astype(int)]]
    names=list(case.generators.name)
    gen=pd.concat([p['result']['gen'] for p in parts], ignore_index=True)
    gen['_o']=gen.name.map({n:i for i,n in enumerate(names)})
    gen=gen.sort_values('_o').drop(columns='_o').reset_index(drop=True)
    use=pd.concat([p['result']['use'] for p in parts], ignore_index=True)
    use=use.sort_values('node').reset_index(drop=True)
    flows=pd.concat([p['result']['flows'] for p in parts], ignore_index=True)
    if len(flows): flows=flows.sort_values(['name','frm','to']).reset_index(drop=True)
    return dict(price=price, gen=gen, use=use, flows=flows)

def stage_c(meta, hours, received, deficit, penalty, flow_cost, method):
    """Move inter-zone flows to cover `deficit`, then price any remainder.

    received and deficit are zone -> (hours,) arrays. The balance asked of
    each zone is the stage A receipt plus the stage B deficit.
    """
    zones=meta['zones']; T=len(hours); w=np.ones(T); zi={z:i for i,z in enumerate(zones)}
    tgt=np.vstack([np.asarray(received[z],float)+np.asarray(deficit[z],float) for z in zones])
    lp=LP(); bal=lp.rows((len(zones), T), tgt, tgt)
    up=lp.var((len(zones), T), float(penalty)*w, 0, np.inf); lp.a(bal, up)
    down=lp.var((len(zones), T), float(penalty)*w, 0, np.inf); lp.a(bal, down, -1)
    flow_vars=[]
    for tie in meta['ties']:
        ia,ib=zi[tie['za']],zi[tie['zb']]; loss=tie['loss']
        fp=lp.var((T,), float(flow_cost)*w, 0, tie['mva'])
        fm=lp.var((T,), float(flow_cost)*w, 0, tie['mva'])
        lp.a(bal[ia], fp, -1); lp.a(bal[ia], fm, 1-loss)
        lp.a(bal[ib], fp, 1-loss); lp.a(bal[ib], fm, -1)
        flow_vars.append((tie, fp, fm))
    status,x,_dual=lp.solve(method)
    _optimal(status, 'zonal-nodal stage C')
    up_x=np.clip(x[up], 0, None); down_x=np.clip(x[down], 0, None)
    flows=[]
    for tie,fp,fm in flow_vars:
        flows.append((tie, np.clip(x[fp],0,None), np.clip(x[fm],0,None)))
    return up_x, down_x, flows

def _imbalance_frame(meta, hours, short, spill, up, down, penalty):
    zi={z:i for i,z in enumerate(meta['zones'])}
    rows=[]
    for z in meta['zones']:
        i=zi[z]
        for t,h in enumerate(hours):
            u,d=float(up[i,t]),float(down[i,t])
            rows.append(dict(hour=int(h), zone=z, short_MW=float(short[z][t]), spill_MW=float(spill[z][t]),
                             balancing_up_MW=u, balancing_down_MW=d, penalty_usd=float(penalty)*(u+d)))
    return pd.DataFrame(rows)

def _balancing_flows(hours, receipts, cleared):
    sched={(tie['name']):(fp,fm) for tie,ra,rb,fp,fm in receipts}
    rows=[]
    for tie,fp,fm in cleared:
        sfp,sfm=sched[tie['name']]
        for t,h in enumerate(hours):
            rows.append(dict(hour=int(h), name=tie['name'], frm=tie['za'], to=tie['zb'],
                             mva=tie['mva'], loss_pct=tie['loss']*100,
                             scheduled_fwd_MW=float(sfp[t]), scheduled_rev_MW=float(sfm[t]),
                             fwd_MW=float(fp[t]), rev_MW=float(fm[t])))
    return pd.DataFrame(rows)

def run_zonal(case, hours, year, method='ipm', log=print):
    view, meta=aggregate(case)
    m=solve(build(view, hours, year=year, transport=True), method)
    _optimal(m['status'], 'zonal solve')
    out=_zonal_tables(m, case, meta)
    out['mode']='zonal'
    log(f'dispatch {year} hours {int(hours[0])}-{int(hours[-1])} {m["status"]} mode zonal')
    return out

def run_zonal_nodal(case, hours, year, method='ipm', workers=None, log=print):
    view, meta=aggregate(case)
    mA=solve(build(view, hours, year=year, transport=True), method)
    _optimal(mA['status'], 'zonal-nodal stage A')
    zonal=_zonal_tables(mA, case, meta)
    zonal['mode']='zonal'
    receipts=tie_receipts(mA, meta)
    T=len(hours)
    inj=border_injections(case, meta, receipts, T)
    received={z:np.zeros(T) for z in meta['zones']}
    for tie,ra,rb,fp,fm in receipts:
        received[tie['za']]+=ra; received[tie['zb']]+=rb
    zmap=meta['zmap']
    for z in meta['zones']:
        total=sum((inj[n] for n,zz in zmap.items() if zz==z), np.zeros(T))
        if T and np.max(np.abs(total-received[z]))>1e-5:
            raise RuntimeError(f'receipt split for zone {z} does not add up')
    pen=feasibility_penalty(case)
    jobs=[]
    for zone in meta['zones']:
        sl, refs=zone_slice(case, zone, zmap)
        ids=[int(n) for n in sl.nodes.node.values]
        arr=np.vstack([inj[i] for i in ids]) if T else np.zeros((len(ids), 0))
        jobs.append((sl, refs, hours, year, method, arr, pen, zone))
    parts=_pool_map(_solve_zone, jobs, workers)
    out=_stitch(case, parts)
    out['gen']=out['gen'].copy()
    out['gen']['zone']=out['gen'].node.astype(int).map(zmap)
    out['use']=out['use'].copy()
    out['use']['zone']=out['use'].node.astype(int).map(zmap)
    short={p['zone']:p['short'] for p in parts}
    spill={p['zone']:p['spill'] for p in parts}
    deficit={z:short[z]-spill[z] for z in meta['zones']}
    price=penalty_price(case)
    up,down,cleared=stage_c(meta, hours, received, deficit, price, case.cfg['flow_cost_usd_mwh'], method)
    out['mode']='zonal-nodal'
    out['zones']=zonal['zones']
    out['transfers']=zonal['transfers']
    out['net_position']=zonal['net_position']
    out['zonal']=zonal
    out['imbalance']=_imbalance_frame(meta, hours, short, spill, up, down, price)
    out['balancing_flows']=_balancing_flows(hours, receipts, cleared)
    total=float(out['imbalance'].penalty_usd.sum())
    log(f'dispatch {year} hours {int(hours[0])}-{int(hours[-1])} Optimal mode zonal-nodal penalty {total:.2f}')
    return out
