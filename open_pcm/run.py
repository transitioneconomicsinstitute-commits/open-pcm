"""Simple contiguous-hour dispatch (no windowing or checkpoints in the core)."""
import numpy as np
from . import model as M
from . import zonal

MODES=('nodal','zonal','zonal-nodal')

def resolve_mode(case, mode):
    """An explicit mode wins. Otherwise case.json, otherwise nodal."""
    if mode is None: mode=case.cfg.get('mode', 'nodal')
    mode=str(mode).strip().lower()
    if mode not in MODES:
        raise ValueError(f'mode must be one of {", ".join(MODES)}; got {mode!r}')
    return mode

def dispatch(case, year, hours=None, log=print, mode=None, method='ipm', workers=None):
    """Solve `hours` (default: the first 168). Returns the result tables.

    mode is nodal, zonal or zonal-nodal. None reads case.json `mode`, which
    defaults to nodal. method is passed to HiGHS. workers is how many zones
    stage B of zonal-nodal solves at once.
    """
    H=np.arange(168) if hours is None else np.asarray(hours,dtype=int)
    if H.size==0: raise ValueError('dispatch: hours is empty (is START greater than END?)')
    if H.min()<0 or H.max()>8759: raise ValueError('dispatch: hours must lie in 0..8759')
    mode=resolve_mode(case, mode)
    if mode=='nodal':
        m=M.solve(M.build(case,H,year=year), method)
        if m['status']!='Optimal': raise RuntimeError(f'dispatch: {m["status"]}')
        log(f'dispatch {year} hours {int(H[0])}-{int(H[-1])} {m["status"]}')
        out=M.extract(m); out['mode']=mode; return out
    if mode=='zonal':
        return zonal.run_zonal(case, H, year, method=method, log=log)
    return zonal.run_zonal_nodal(case, H, year, method=method, workers=workers, log=log)
