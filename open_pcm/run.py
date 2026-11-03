"""Contiguous-hour dispatch.

blocks splits the horizon into consecutive solves. The level in each storage tank
at the end of a block is the starting level of the next. The core has no unit
commitment, so there is no on/off status to carry. Results are stitched back into
one set of tables, the same shape as a single solve.
"""
import numpy as np, pandas as pd
from . import model as M
from . import zonal
from .lp import check_status, gap_limit

MODES=('nodal','zonal','zonal-nodal')

def resolve_mode(case, mode):
    """An explicit mode wins. Otherwise case.json, otherwise nodal."""
    if mode is None: mode=case.cfg.get('mode', 'nodal')
    mode=str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}; got {mode!r}')
    return mode

def block_sizes(n, blocks):
    """None leaves the horizon as one solve. An int repeats; the tail may be shorter.
    A sequence of lengths must be positive and sum to n."""
    if blocks is None: return None
    if isinstance(blocks, (int, np.integer)):
        step=int(blocks)
        if step<=0: raise ValueError('blocks must be positive')
        out=[]; left=int(n)
        while left>0:
            out.append(min(step, left)); left-=out[-1]
        return out
    try:
        sizes=[int(b) for b in blocks]
    except TypeError:
        raise ValueError('blocks must be a length or a sequence of lengths')
    if not sizes or any(s<=0 for s in sizes) or sum(sizes)!=int(n):
        raise ValueError(f'block lengths must be positive and sum to the {int(n)} hours')
    return sizes

def _sum_aligned(frames, cols):
    """Add value columns. Identity columns have to match, in the same order, in every block."""
    out=frames[0].copy()
    if len(out)==0: return out
    ids=[c for c in out.columns if c not in cols]
    for f in frames[1:]:
        same=len(f)==len(out) and out[ids].reset_index(drop=True).equals(f[ids].reset_index(drop=True))
        if not same: raise RuntimeError('block results do not line up')
        for c in cols:
            out[c]=out[c].to_numpy(dtype=float)+f[c].to_numpy(dtype=float)
    return out

def _tuples(frame, keys):
    if len(keys)==1: return [(v,) for v in frame[keys[0]].tolist()]
    return list(zip(*[frame[c].tolist() for c in keys]))

def _stack(frames, keys):
    """Stack block rows, then order them as one solve does: each key, then hour."""
    out=pd.concat(frames, ignore_index=True)
    if not len(out): return out
    rank={}
    for rec in _tuples(frames[0], keys):
        if rec not in rank: rank[rec]=len(rank)
    for f in frames[1:]:
        if set(_tuples(f, keys))!=set(rank): raise RuntimeError('block results do not line up')
    out=out.copy()
    out['_o']=[rank[rec] for rec in _tuples(out, keys)]
    out=out.sort_values(['_o','hour'], kind='mergesort').drop(columns='_o')
    return out.reset_index(drop=True)

def stitch(parts):
    """Join block results into the tables a single solve would have returned."""
    out=dict(mode=parts[0]['mode'])
    out['price']=pd.concat([p['price'] for p in parts], ignore_index=True)
    out['gen']=_sum_aligned([p['gen'] for p in parts], ['MWh'])
    out['use']=_sum_aligned([p['use'] for p in parts], ['unserved_MWh','load_MWh'])
    out['flows']=_sum_aligned([p['flows'] for p in parts], ['fwd_MWh','rev_MWh'])
    if 'storage' in parts[0]:
        out['storage']=_stack([p['storage'] for p in parts], ['name'])
    for key in ('zones','transfers'):
        if key in parts[0]: out[key]=parts[0][key]
    if 'net_position' in parts[0]:
        out['net_position']=_stack([p['net_position'] for p in parts], ['zone'])
    if 'imbalance' in parts[0]:
        out['imbalance']=_stack([p['imbalance'] for p in parts], ['zone'])
    if 'balancing_flows' in parts[0]:
        out['balancing_flows']=_stack([p['balancing_flows'] for p in parts], ['name'])
    if 'zonal' in parts[0]:
        out['zonal']=stitch([p['zonal'] for p in parts])
    return out

def combine_meta(flags):
    """One metadata dict for every accepted solve in the run. flagged if any part was."""
    def worst(key):
        vals=[f[key] for f in flags if f.get(key) is not None]
        return max(vals) if vals else None
    flagged=any(f['flagged'] for f in flags)
    return dict(status='Time limit reached' if flagged else 'Optimal', flagged=flagged,
                gap=worst('gap'), objective_error=worst('objective_error'))

def _run(case, year, hours, log, mode, method, workers, time_limit, accept_gap, storage_state):
    if mode=='nodal':
        m=M.solve(M.build(case, hours, year=year, storage_state=storage_state), method,
                  time_limit=time_limit, accept_gap=accept_gap)
        solved=check_status(m['status'], m['report'], accept_gap, 'dispatch')
        log(f'dispatch {year} hours {int(hours[0])}-{int(hours[-1])} {m["status"]}')
        out=M.extract(m); out['mode']=mode
        return out, M.terminal_storage(m), [solved]
    if mode=='zonal':
        return zonal.run_zonal(case, hours, year, method=method, log=log, time_limit=time_limit,
                               accept_gap=accept_gap, storage_state=storage_state)
    return zonal.run_zonal_nodal(case, hours, year, method=method, workers=workers, log=log,
                                 time_limit=time_limit, accept_gap=accept_gap, storage_state=storage_state)

def dispatch(case, year, hours=None, log=print, mode=None, method='ipm', workers=None,
             blocks=None, time_limit=None, accept_gap=None):
    """Solve `hours` (default: the first 168). Returns the result tables.

    mode is nodal, zonal or zonal-nodal. None reads case.json `mode`, which
    defaults to nodal. method is passed to HiGHS. workers is how many zones
    stage B of zonal-nodal solves at once.

    blocks splits those hours into consecutive solves. An int is a block length,
    and the last block is shorter when it does not divide. A sequence of lengths
    must sum to the number of hours. Storage levels cross each join. Leave blocks
    unset for one LP, which is the default.

    time_limit is seconds. accept_gap keeps a solve that hit the limit when its
    relative gap or primal-dual objective error is within the tolerance and HiGHS
    has a feasible primal solution. That result has meta['flagged'] true.
    """
    H=np.arange(168) if hours is None else np.asarray(hours,dtype=int)
    if H.size==0: raise ValueError('dispatch: hours is empty (is START greater than END?)')
    if H.min()<0 or H.max()>8759: raise ValueError('dispatch: hours must lie in 0..8759')
    mode=resolve_mode(case, mode)
    accept_gap=gap_limit(accept_gap)
    if time_limit is not None and float(time_limit)<0: raise ValueError('time_limit must be >= 0')
    sizes=block_sizes(len(H), blocks)
    if M.has_storage(case):
        if mode=='zonal-nodal': raise ValueError('zonal-nodal does not schedule storage')
        if len(H)>1 and np.any(np.diff(H)!=1): raise ValueError('storage dispatch needs contiguous hours')
    explicit=sizes is not None
    if sizes is None: sizes=[len(H)]
    parts=[]; flags=[]; state=None; i=0
    for n in sizes:
        chunk=H[i:i+n]; i+=n
        part, state, part_flags=_run(case, year, chunk, log, mode, method, workers,
                                     time_limit, accept_gap, state)
        parts.append(part); flags.extend(part_flags)
    out=parts[0] if len(parts)==1 else stitch(parts)
    if explicit or time_limit is not None or accept_gap is not None:
        out['meta']=combine_meta(flags)
        if explicit: out['meta']['blocks']=[int(s) for s in sizes]
    return out
