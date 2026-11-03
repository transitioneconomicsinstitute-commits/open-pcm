"""Nodal production cost model: thermal and renewable plants, DC power flow, unserved energy.
Solved as one LP with HiGHS. Optional storage links the hours. No unit commitment, hydro budgets or expansion.
transport=True drops the DC angles and keeps branch ratings as transfer limits."""
import numpy as np, pandas as pd
from .lp import LP
from .case import num

def has_storage(case):
    st=getattr(case,'storage',None)
    return st is not None and len(st)>0

def build(case, hours, year=None, transport=False, injections=None, feas_penalty=None, refs=None, storage_state=None):
    """Build a single-block dispatch LP for the given hour-of-year indices.

    transport keeps ratings as transfer limits and skips DC angles.
    injections is MW into each bus, shape (nodes, hours); positive injects.
    feas_penalty, if set, is the $/MWh cost of shortage and spill slacks.
    refs fixes the angle at each listed bus. The default is the case reference bus.
    storage_state, if set, is name -> starting MWh and overrides storage.csv.
    """
    c=case.cfg; nodes=case.nodes; nid=nodes.node.values.astype(int); ni={b:i for i,b in enumerate(nid)}; N=len(nid)
    H=np.asarray(hours); T=len(H); w=np.ones(T); y=case.cfg['base_year'] if year is None else year
    L=case.load(y,H)
    if injections is None:
        rhs=L
    else:
        inj=np.asarray(injections,float)
        if inj.shape!=(N,T): raise ValueError(f'injections must have shape {(N,T)}, got {inj.shape}')
        rhs=L-inj
    lp=LP(); bal=lp.rows((N,T),rhs,rhs); units=[]; lines=[]
    for g in case.generators.to_dict('records'):
        b=ni[int(g['node'])]; cap=float(g['mw']); t=g['type']
        if t=='thermal':
            cost=(num(g.get('heat_rate'),0)*num(g.get('fuel_usd_mmbtu'),0)+num(g.get('vom'),0))*w
            gen=lp.var((T,),cost,0,cap)
        elif t in ('solar','wind'):
            cf=case.profiles[g['profile']].values[H]
            gen=lp.var((T,),num(g.get('vom'),0)*w,0,cap*cf)
        else:
            raise ValueError(f'core dispatch supports thermal, solar and wind; got {t!r}')
        lp.a(bal[b],gen); units.append((dict(name=g['name'],bus=int(g['node']),type=t,cap=cap),gen))
    # Angle variables stay in this position so the default nodal LP is unchanged.
    if not transport:
        th=lp.var((N,T),0,-np.inf,np.inf)
    for r in case.branches.to_dict('records'):
        a_,b_=int(r['from']),int(r['to']); mva=float(r['MVA']); x=float(r['x_pu'])
        loss=float(r.get('loss_pct',c['line_loss_frac']*100))/100
        ia,ib=ni[a_],ni[b_]
        fp=lp.var((T,),c['flow_cost_usd_mwh']*w,0,mva); fm=lp.var((T,),c['flow_cost_usd_mwh']*w,0,mva)
        if not transport:
            row=lp.rows((T,),0,0); lp.a(row,fp); lp.a(row,fm,-1); lp.a(row,th[ia],-1/x); lp.a(row,th[ib],1/x)
        lp.a(bal[ia],fp,-1); lp.a(bal[ia],fm,1-loss); lp.a(bal[ib],fp,1-loss); lp.a(bal[ib],fm,-1)
        lines.append((a_,b_,mva,r.get('name',f'{a_}-{b_}'),fp,fm))
    if not transport:
        if refs is None: refs=[int(c['ref_node'])]
        for b in refs:
            rr=lp.rows((T,),0,0); lp.a(rr,th[ni[int(b)]])
    use=lp.var((N,T),c['voll_usd_mwh']*w[None,:],0,L); lp.a(bal,use)
    short=spill=None
    if feas_penalty is not None:
        pen=float(feas_penalty)*w
        short=lp.var((N,T),pen,0,np.inf); lp.a(bal,short)
        spill=lp.var((N,T),pen,0,np.inf); lp.a(bal,spill,-1)
    stores=_storage(case,lp,bal,ni,T,w,storage_state)
    return dict(lp=lp,bal=bal,units=units,lines=lines,use=use,L=L,w=w,nid=nid,hours=H,short=short,spill=spill,stores=stores)

def _storage(case, lp, bal, ni, T, w, storage_state):
    """Tank balance. efficiency is the share of charged energy that is stored; discharge leaves the tank 1:1."""
    if not has_storage(case): return []
    state=storage_state or {}
    stores=[]; seen=set()
    for g in case.storage.to_dict('records'):
        name=str(g['name'])
        if name in seen: raise ValueError(f'storage name {name!r} is repeated')
        seen.add(name)
        node=int(g['node'])
        if node not in ni: raise ValueError(f'storage {name} is on unknown bus {node}')
        mw=float(g['mw']); mwh=float(g['mwh'])
        if mw<=0 or mwh<=0: raise ValueError(f'storage {name}: mw and mwh must be positive')
        eta=num(g.get('efficiency'),1.)
        if eta<0 or eta>1: raise ValueError(f'storage {name}: efficiency must be between 0 and 1')
        vom=num(g.get('vom'),0.)
        s0=float(state[name]) if name in state else num(g.get('initial_mwh'),0.)
        if s0<-1e-6 or s0>mwh+1e-6: raise ValueError(f'storage {name}: initial level must lie within 0 and {mwh}')
        s0=float(min(max(s0,0.),mwh))
        ch=lp.var((T,),vom*w,0,mw); dis=lp.var((T,),vom*w,0,mw); soc=lp.var((T,),0,0,mwh)
        rhs=np.zeros(T); rhs[0]=s0
        row=lp.rows((T,),rhs,rhs)
        lp.a(row,soc); lp.a(row,dis); lp.a(row,ch,-eta)
        if T>1: lp.a(row[1:],soc[:-1],-1)
        lp.a(bal[ni[node]],dis); lp.a(bal[ni[node]],ch,-1)
        stores.append(dict(name=name,bus=node,soc=soc,charge=ch,discharge=dis,eta=eta,mwh=mwh))
    return stores

def solve(model,method='ipm',tol=None,threads=None,time_limit=None,accept_gap=None):
    st,x,dual=model['lp'].solve(method,tol=tol,threads=threads,time_limit=time_limit,accept_gap=accept_gap)
    model.update(status=st,x=x,dual=dual,report=model['lp'].report); return model

def terminal_storage(m):
    """MWh in each tank at the last hour. This is the level the next block starts from."""
    x=m['x']
    return {s['name']:float(x[s['soc'][-1]]) for s in m.get('stores') or []}

def extract(m):
    """Hourly prices, generation by unit, unserved energy and line flows."""
    x=m['x']; du=m['dual']; nid=list(m['nid']); w=m['w']; H=m['hours']
    pr=du[m['bal']]/w[None,:]
    price=pd.DataFrame(pr.T,columns=nid); price.insert(0,'hour',H)
    gen=pd.DataFrame([dict(name=u['name'],node=u['bus'],type=u['type'],MWh=float(x[g].sum())) for u,g in m['units']])
    use=pd.DataFrame(dict(node=nid,unserved_MWh=x[m['use']].sum(1),load_MWh=m['L'].sum(1)))
    fl=pd.DataFrame([dict(frm=l[0],to=l[1],MVA=l[2],name=l[3],
        fwd_MWh=float(x[l[4]].sum()),rev_MWh=float(x[l[5]].sum())) for l in m['lines']])
    out=dict(price=price,gen=gen,use=use,flows=fl)
    if m.get('stores'):
        rows=[]
        for s in m['stores']:
            for t,h in enumerate(H):
                rows.append(dict(name=s['name'],hour=int(h),soc_mwh=float(x[s['soc'][t]]),
                                 charge_mw=float(x[s['charge'][t]]),discharge_mw=float(x[s['discharge'][t]])))
        out['storage']=pd.DataFrame(rows)
    return out
