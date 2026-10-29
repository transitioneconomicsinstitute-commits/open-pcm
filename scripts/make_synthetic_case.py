"""Generates the SYNTHETIC example case. All numbers are invented for testing;
they describe no real power system."""
import os, json, numpy as np, pandas as pd
O=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','examples','synthetic_4node'); os.makedirs(O,exist_ok=True)
rng=np.random.default_rng(42); h=np.arange(8760); hod=h%24; doy=h//24
load=0.65+0.2*np.sin((hod-8)/24*2*np.pi).clip(0)+0.1*np.sin((hod-19)/24*2*np.pi).clip(0)+0.05*np.cos((doy-200)/365*2*np.pi)
load=load*(1+0.02*rng.standard_normal(8760)); load/=load.max()
sol=np.clip(np.sin((hod-6)/12*np.pi),0,None)*(0.85+0.15*rng.random(8760))*(0.9+0.1*np.cos((doy-172)/365*2*np.pi))
wind=np.clip(0.35+0.25*np.sin(doy/365*2*np.pi+1)+0.15*rng.standard_normal(8760).cumsum()/60,0.02,0.95)
pd.DataFrame(dict(hour=h,load=load.round(4),solar=sol.round(4),wind=wind.round(4))).to_csv(f'{O}/profiles.csv',index=False)
pd.DataFrame(dict(node=[1,2,3,4],name=['North','East','South','West'],peak_MW=[400,250,600,150],
                 zone=['NE','NE','SW','SW'])).to_csv(f'{O}/nodes.csv',index=False)
pd.DataFrame([
 dict(name='Gas CC A',node=3,type='thermal',mw=450,heat_rate=7.0,fuel_usd_mmbtu=6.0,vom=3,profile=''),
 dict(name='Gas peaker B',node=1,type='thermal',mw=200,heat_rate=10.5,fuel_usd_mmbtu=6.0,vom=6,profile=''),
 dict(name='Oil engines C',node=4,type='thermal',mw=120,heat_rate=9.8,fuel_usd_mmbtu=16.0,vom=9,profile=''),
 dict(name='Solar E',node=4,type='solar',mw=100,heat_rate=0,fuel_usd_mmbtu=0,vom=0,profile='solar'),
 dict(name='Wind F',node=1,type='wind',mw=80,heat_rate=0,fuel_usd_mmbtu=0,vom=0,profile='wind'),
]).to_csv(f'{O}/generators.csv',index=False)
pd.DataFrame([dict(name='L12',**{'from':1,'to':2},MVA=250,x_pu=0.10,loss_pct=1.5),
              dict(name='L23',**{'from':2,'to':3},MVA=300,x_pu=0.08,loss_pct=1.5),
              dict(name='L34',**{'from':3,'to':4},MVA=150,x_pu=0.12,loss_pct=2.0),
              dict(name='L14',**{'from':1,'to':4},MVA=100,x_pu=0.15,loss_pct=2.0)]).to_csv(f'{O}/branches.csv',index=False)
json.dump(dict(description='SYNTHETIC 4-node test system. Invented numbers; not a real power system.',
               base_year=2026,discount_rate=0.08,voll_usd_mwh=1500,ref_node=3,demand_growth=0.04),
          open(f'{O}/case.json','w'),indent=1)
open(f'{O}/README.md','w').write('# Synthetic 4-node example\n\nSYNTHETIC DATA. Every number in this folder was invented by `scripts/make_synthetic_case.py` (random seed 42) for testing and illustration. It does not describe any real power system.\n\nBuses 1 and 2 are zone NE. Buses 3 and 4 are zone SW. The default nodal solve ignores the zone column.\n')
print('wrote',O)
