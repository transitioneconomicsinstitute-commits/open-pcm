import shutil, pandas as pd, pytest
from pathlib import Path
import open_pcm as P

CASE=Path(__file__).resolve().parents[1]/'examples'/'synthetic_4node'

def copy(tmp_path):
    d=tmp_path/'case'; shutil.copytree(CASE,d); return d

def edit(d,f,fn):
    df=pd.read_csv(d/f); df=fn(df); df.to_csv(d/f,index=False)

def problems(d):
    with pytest.raises(P.CaseError) as e: P.Case(str(d))
    return e.value.problems

def test_example_is_clean():
    assert P.check_case(P.Case(str(CASE)))==[]

def test_unknown_generator_node(tmp_path):
    d=copy(tmp_path); edit(d,'generators.csv',lambda g: g.assign(node=g.node.where(g.index!=0,99)))
    p=problems(d)
    assert len(p)==1 and 'generators.csv row 2' in p[0] and 'node=99' in p[0]

def test_branch_problems_are_all_reported(tmp_path):
    d=copy(tmp_path)
    def bad(b):
        b.loc[0,'x_pu']=0; b.loc[1,'to']=42; return b
    edit(d,'branches.csv',bad)
    p=problems(d)
    assert any('x_pu must be > 0' in s for s in p)
    assert any('to=42' in s for s in p)

def test_missing_profile_column(tmp_path):
    d=copy(tmp_path)
    edit(d,'generators.csv',lambda g: g.assign(profile=g.profile.where(g.type=='thermal','sun')))
    assert any("profile 'sun'" in s for s in problems(d))

def test_unknown_type_and_missing_column(tmp_path):
    d=copy(tmp_path); edit(d,'generators.csv',lambda g: g.assign(type=g.type.replace('thermal','nuclear')))
    assert any("type='nuclear'" in s for s in problems(d))
    d2=tmp_path/'c2'; shutil.copytree(CASE,d2); edit(d2,'branches.csv',lambda b: b.drop(columns='MVA'))
    assert problems(d2)==['branches.csv is missing column(s): MVA']

def test_short_profiles_and_bad_cf(tmp_path):
    d=copy(tmp_path)
    edit(d,'profiles.csv',lambda p: p.iloc[:100])
    assert any('8760 rows, got 100' in s for s in problems(d))
    d2=tmp_path/'c2'; shutil.copytree(CASE,d2)
    edit(d2,'profiles.csv',lambda p: p.assign(solar=p.solar*5))
    assert any('solar must be between 0 and 1' in s for s in problems(d2))

def test_validate_can_be_skipped(tmp_path):
    d=copy(tmp_path); edit(d,'generators.csv',lambda g: g.assign(node=99))
    c=P.Case(str(d),validate=False)
    assert P.check_case(c)
