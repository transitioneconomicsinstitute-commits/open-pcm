"""Command line: python -m open_pcm CASE_DIR OUT_DIR --year Y [--hours START END] [--blocks N ...] [--mode MODE]"""
import argparse, json, os
from . import Case, dispatch

def _csv(df, path, digits):
    df.round(digits).to_csv(path, index=False)

def main():
    a=argparse.ArgumentParser(prog='open-pcm')
    a.add_argument('case'); a.add_argument('out'); a.add_argument('--year',type=int,required=True)
    a.add_argument('--hours',type=int,nargs=2,metavar=('START','END'),help='inclusive hour-of-year range (default 0 167)')
    a.add_argument('--mode', choices=('nodal','zonal','zonal-nodal'),
                   help='nodal (default), zonal, or zonal-nodal; overrides case.json')
    a.add_argument('--blocks', type=int, nargs='+', metavar='N',
                   help='consecutive block lengths in hours; one N repeats to cover the horizon')
    a.add_argument('--time-limit', type=float, help='HiGHS time limit in seconds')
    a.add_argument('--accept-gap', type=float,
                   help='keep a time-limited solve whose relative gap or objective error is at most this')
    A=a.parse_args(); C=Case(A.case); os.makedirs(A.out,exist_ok=True)
    H=None if A.hours is None else range(A.hours[0],A.hours[1]+1)
    blocks=None if A.blocks is None else (A.blocks[0] if len(A.blocks)==1 else A.blocks)
    r=dispatch(C,A.year,H,mode=A.mode,blocks=blocks,time_limit=A.time_limit,accept_gap=A.accept_gap)
    y=A.year
    _csv(r['price'], os.path.join(A.out,f'prices_hourly_{y}.csv'), 2)
    gcols=['name','node','type']+(['zone'] if 'zone' in r['gen'].columns else [])
    r['gen'].groupby(gcols).MWh.sum().round(0).reset_index().to_csv(
        os.path.join(A.out,f'generation_{y}.csv'), index=False)
    _csv(r['use'], os.path.join(A.out,f'unserved_{y}.csv'), 0)
    _csv(r['flows'], os.path.join(A.out,f'flows_{y}.csv'), 0)
    if 'net_position' in r: _csv(r['net_position'], os.path.join(A.out,f'net_position_{y}.csv'), 2)
    if 'transfers' in r: _csv(r['transfers'], os.path.join(A.out,'transfers.csv'), 4)
    if 'imbalance' in r: _csv(r['imbalance'], os.path.join(A.out,f'imbalance_{y}.csv'), 2)
    if 'balancing_flows' in r: _csv(r['balancing_flows'], os.path.join(A.out,f'balancing_flows_{y}.csv'), 2)
    if 'zonal' in r:
        _csv(r['zonal']['price'], os.path.join(A.out,f'zonal_prices_{y}.csv'), 2)
        _csv(r['zonal']['flows'], os.path.join(A.out,f'zonal_flows_{y}.csv'), 2)
    if 'storage' in r: _csv(r['storage'], os.path.join(A.out,f'storage_{y}.csv'), 4)
    if 'meta' in r:
        with open(os.path.join(A.out,'meta.json'),'w') as f:
            json.dump(r['meta'], f, indent=1)
            f.write('\n')
    print('done')

if __name__=='__main__':
    main()
