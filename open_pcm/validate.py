"""Checks on a loaded case, so bad inputs fail early with a message that names the file and row."""
import numpy as np, pandas as pd

class CaseError(ValueError):
    """Raised when case files are inconsistent. Holds every problem found, not just the first."""
    def __init__(self, problems):
        self.problems=list(problems)
        super().__init__('invalid case:\n  - '+'\n  - '.join(self.problems))

REQUIRED={
    'nodes.csv':['node','peak_MW'],
    'generators.csv':['name','node','type','mw'],
    'branches.csv':['from','to','MVA','x_pu'],
    'profiles.csv':['load'],
}
GEN_TYPES=('thermal','solar','wind')

def _label(df,i,f):
    n=df['name'].iloc[i] if 'name' in df.columns else None
    return f'{f} row {i+2}'+(f' ({n})' if isinstance(n,str) and n else '')

def _fmt(v):
    return repr(v) if isinstance(v,str) else (f'{v:g}' if isinstance(v,(int,float,np.number)) else str(v))

def _numeric(df,col,f,out,min_value=None,positive=False,optional=False):
    if col not in df.columns: return
    v=pd.to_numeric(df[col],errors='coerce')
    for i in range(len(df)):
        raw=df[col].iloc[i]
        if pd.isna(raw):
            if not optional: out.append(f'{_label(df,i,f)}: {col} is empty')
            continue
        x=v.iloc[i]
        if pd.isna(x): out.append(f'{_label(df,i,f)}: {col}={_fmt(raw)} is not a number'); continue
        if positive and not x>0: out.append(f'{_label(df,i,f)}: {col} must be > 0, got {x:g}')
        elif min_value is not None and x<min_value: out.append(f'{_label(df,i,f)}: {col} must be >= {min_value:g}, got {x:g}')

def check_case(case):
    """Return a list of problems with `case`. An empty list means the case looks usable."""
    out=[]
    tables={'nodes.csv':case.nodes,'generators.csv':case.generators,'branches.csv':case.branches,'profiles.csv':case.profiles}
    for f,cols in REQUIRED.items():
        miss=[c for c in cols if c not in tables[f].columns]
        if miss: out.append(f'{f} is missing column(s): {", ".join(miss)}')
    if out: return out
    nodes=pd.to_numeric(case.nodes.node,errors='coerce')
    if nodes.isna().any(): out.append('nodes.csv: node ids must be integers')
    dup=sorted(set(nodes[nodes.duplicated()].dropna().astype(int)))
    if dup: out.append(f'nodes.csv: duplicate node id(s): {dup}')
    known=set(nodes.dropna().astype(int))
    _numeric(case.nodes,'peak_MW','nodes.csv',out,min_value=0)
    def ref(df,col,f):
        for i,v in enumerate(pd.to_numeric(df[col],errors='coerce')):
            if pd.isna(v) or int(v) not in known:
                out.append(f'{_label(df,i,f)}: {col}={_fmt(df[col].iloc[i])} is not a node in nodes.csv')
    g=case.generators
    ref(g,'node','generators.csv'); _numeric(g,'mw','generators.csv',out,min_value=0)
    for c in ('heat_rate','fuel_usd_mmbtu','vom'): _numeric(g,c,'generators.csv',out,min_value=0,optional=True)
    for i,t in enumerate(g['type']):
        if t not in GEN_TYPES:
            out.append(f'{_label(g,i,"generators.csv")}: type={t!r}, expected one of {", ".join(GEN_TYPES)}')
        elif t!='thermal':
            p=g['profile'].iloc[i] if 'profile' in g.columns else None
            if not isinstance(p,str) or not p:
                out.append(f'{_label(g,i,"generators.csv")}: {t} plant needs a profile column name')
            elif p not in case.profiles.columns:
                out.append(f'{_label(g,i,"generators.csv")}: profile {p!r} is not a column in profiles.csv')
    b=case.branches
    ref(b,'from','branches.csv'); ref(b,'to','branches.csv')
    _numeric(b,'MVA','branches.csv',out,min_value=0); _numeric(b,'x_pu','branches.csv',out,positive=True)
    _numeric(b,'loss_pct','branches.csv',out,min_value=0,optional=True)
    st=case.storage
    if st is not None and len(st):
        for c in ('name','node','mw','mwh'):
            if c not in st.columns: out.append(f'storage.csv is missing column {c}')
        if 'node' in st.columns: ref(st,'node','storage.csv')
        for c in ('mw','mwh'): _numeric(st,c,'storage.csv',out,min_value=0)
        _numeric(st,'efficiency','storage.csv',out,positive=True,optional=True)
        if 'efficiency' in st.columns:
            e=pd.to_numeric(st.efficiency,errors='coerce')
            for i in np.where(e>1)[0]: out.append(f'{_label(st,i,"storage.csv")}: efficiency must be <= 1, got {e.iloc[i]:g}')
    p=case.profiles
    if len(p)!=8760: out.append(f'profiles.csv must have 8760 rows, got {len(p)}')
    for c in p.columns:
        if c=='hour': continue
        v=pd.to_numeric(p[c],errors='coerce')
        if v.isna().any(): out.append(f'profiles.csv: column {c} has {int(v.isna().sum())} empty or non-numeric value(s)')
        elif c!='load' and ((v<0)|(v>1)).any(): out.append(f'profiles.csv: capacity factors in {c} must be between 0 and 1')
        elif c=='load' and (v<0).any(): out.append('profiles.csv: load must not be negative')
    rn=case.cfg.get('ref_node')
    if rn is not None and known and int(rn) not in known: out.append(f'case.json: ref_node={rn} is not a node in nodes.csv')
    return out
