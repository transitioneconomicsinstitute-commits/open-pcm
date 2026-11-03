"""Thin sparse LP builder on top of HiGHS."""
import numpy as np, scipy.sparse as sp, highspy

class LP:
    """Collects variables, row bounds and coefficients, then solves with HiGHS."""
    def __init__(s):
        s.n=0; s.c=[]; s.lb=[]; s.ub=[]; s.m=0; s.rl=[]; s.ru=[]; s.I=[]; s.J=[]; s.V=[]
    def var(s,shape,cost=0.,lb=0.,ub=np.inf):
        k=int(np.prod(shape)); idx=np.arange(s.n,s.n+k).reshape(shape); s.n+=k
        for L,v in ((s.c,cost),(s.lb,lb),(s.ub,ub)): L.append(np.broadcast_to(np.asarray(v,float),shape).ravel())
        return idx
    def rows(s,shape,lo,hi):
        k=int(np.prod(shape)); idx=np.arange(s.m,s.m+k).reshape(shape); s.m+=k
        s.rl.append(np.broadcast_to(np.asarray(lo,float),shape).ravel()); s.ru.append(np.broadcast_to(np.asarray(hi,float),shape).ravel()); return idx
    def a(s,r,j,v=1.):
        r,j,v=np.broadcast_arrays(np.asarray(r),np.asarray(j),np.asarray(v,float)); s.I.append(r.ravel()); s.J.append(j.ravel()); s.V.append(v.ravel())
    def solve(s,method='ipm',tol=None,threads=None,time_limit=None,accept_gap=None):
        """method: HiGHS solver ('ipm'/'ipx', 'hipo' with highspy-extras, 'simplex', ...). Interior-point methods run without
        crossover and retry with crossover if not optimal. tol: ipm_optimality_tolerance (HiGHS default 1e-8); threads: HiGHS threads.
        time_limit is seconds (unset means no limit). accept_gap, when set, keeps a feasible time-limited solution whose
        relative gap or primal-dual objective error is within it, and does not retry crossover in that case.
        The gap report is left on s.report."""
        accept_gap=gap_limit(accept_gap)
        A=sp.csr_matrix((np.concatenate(s.V),(np.concatenate(s.I),np.concatenate(s.J))),shape=(s.m,s.n))
        lp=highspy.HighsLp(); lp.num_col_=s.n; lp.num_row_=s.m
        lp.col_cost_=np.concatenate(s.c); lp.col_lower_=np.concatenate(s.lb); lp.col_upper_=np.concatenate(s.ub)
        lp.row_lower_=np.concatenate(s.rl); lp.row_upper_=np.concatenate(s.ru)
        lp.a_matrix_.format_=highspy.MatrixFormat.kRowwise; lp.a_matrix_.start_=A.indptr; lp.a_matrix_.index_=A.indices; lp.a_matrix_.value_=A.data
        h=highspy.Highs(); h.setOptionValue('output_flag',False); h.setOptionValue('solver',method)
        ipm=method in ('ipm','ipx','hipo')
        if tol: h.setOptionValue('ipm_optimality_tolerance',float(tol))
        if threads: h.setOptionValue('threads',int(threads))
        if time_limit is not None:
            time_limit=float(time_limit)
            if time_limit<0: raise ValueError('time_limit must be >= 0')
            h.setOptionValue('time_limit',time_limit)
        h.setOptionValue('run_crossover','off' if ipm else 'on'); h.passModel(lp); h.run()
        st=h.modelStatusToString(h.getModelStatus()); rep=solver_report(h)
        # A close enough interior-point solution is the result. Otherwise retry with crossover, as before.
        if ipm and st!='Optimal' and not within_gap(st,rep,accept_gap):
            h.setOptionValue('run_crossover','on'); h.run()
            st=h.modelStatusToString(h.getModelStatus()); rep=solver_report(h)
        s.report=rep
        sol=h.getSolution()
        return st, np.array(sol.col_value), np.array(sol.row_dual)

def _finite(v):
    v=float(v)
    return v if np.isfinite(v) else None

def gap_limit(accept_gap):
    """None, or a finite gap tolerance. Raises if the value is not usable."""
    if accept_gap is None: return None
    v=float(accept_gap)
    if not np.isfinite(v) or v<0: raise ValueError('accept_gap must be finite and >= 0')
    return v

def reported_figure(gap, objective_error, feasible):
    """The number accept_gap is compared with.

    HiGHS puts the relative MIP gap in mip_gap and leaves it infinite for an LP.
    An LP reports the primal-dual objective error instead. A solution that is not
    primal feasible has no figure, even if the error field is zero.
    """
    g=None if gap is None else _finite(gap)
    e=None if objective_error is None else _finite(objective_error)
    if g is not None: return g
    if feasible and e is not None: return e
    return None

def solver_report(h):
    """Relative gap and primal-dual objective error from a finished HiGHS run."""
    info=h.getInfo()
    gap=_finite(info.mip_gap)
    err=_finite(info.primal_dual_objective_error)
    feasible=int(info.primal_solution_status)==int(highspy.kSolutionStatusFeasible)
    return dict(gap=gap, objective_error=err, feasible=feasible,
                reported=reported_figure(gap, err, feasible))

def within_gap(status, report, accept_gap):
    """True when a time-limited feasible solve is inside accept_gap."""
    limit=gap_limit(accept_gap)
    if limit is None or status!='Time limit reached' or not report.get('feasible'): return False
    rel=report.get('reported')
    return rel is not None and rel<=limit

def check_status(status, report, accept_gap, what):
    """Accept an optimal solve, or a time-limited one inside accept_gap. Otherwise raise.

    A kept time-limited solve is marked flagged. gap is the MIP relative gap when HiGHS
    reported one; objective_error is the primal-dual figure.
    """
    if report is None: report={}
    meta=dict(status=status, flagged=False, gap=report.get('gap'), objective_error=report.get('objective_error'))
    if status=='Optimal': return meta
    if within_gap(status, report, accept_gap):
        meta['flagged']=True
        return meta
    raise RuntimeError(f'{what}: {status}')
