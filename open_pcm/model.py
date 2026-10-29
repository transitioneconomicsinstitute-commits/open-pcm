"""Nodal production cost model: thermal and renewable plants, DC power flow, unserved energy.
Solved as one LP with HiGHS. No unit commitment, storage, hydro budgets or expansion in the core.
transport=True drops the DC angles and keeps branch ratings as transfer limits."""
import numpy as np, pandas as pd
from .lp import LP
from .case import num

def build(case, hours, year=None, transport=False, injections=None, feas_penalty=None, refs=None):
    """Build a single-block dispatch LP for the given hour-of-year indices.

    transport keeps ratings as transfer limits and skips DC angles.
    injections is MW into each bus, shape (nodes, hours); positive injects.
    feas_penalty, if set, is the $/MWh cost of shortage and spill slacks.
    refs fixes the angle at each listed bus. The default is the case reference bus.
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
    return dict(lp=lp,bal=bal,units=units,lines=lines,use=use,L=L,w=w,nid=nid,hours=H,short=short,spill=spill)

def solve(model,method='ipm'):
    st,x,dual=model['lp'].solve(method); model.update(status=st,x=x,dual=dual); return model

def extract(m):
    """Hourly prices, generation by unit, unserved energy and line flows."""
    x=m['x']; du=m['dual']; nid=list(m['nid']); w=m['w']; H=m['hours']
    pr=du[m['bal']]/w[None,:]
    price=pd.DataFrame(pr.T,columns=nid); price.insert(0,'hour',H)
    gen=pd.DataFrame([dict(name=u['name'],node=u['bus'],type=u['type'],MWh=float(x[g].sum())) for u,g in m['units']])
    use=pd.DataFrame(dict(node=nid,unserved_MWh=x[m['use']].sum(1),load_MWh=m['L'].sum(1)))
    fl=pd.DataFrame([dict(frm=l[0],to=l[1],MVA=l[2],name=l[3],
        fwd_MWh=float(x[l[4]].sum()),rev_MWh=float(x[l[5]].sum())) for l in m['lines']])
    return dict(price=price,gen=gen,use=use,flows=fl)
