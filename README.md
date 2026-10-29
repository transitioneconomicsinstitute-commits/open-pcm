# OPEN-PCM

OPEN-PCM is a small production cost model for power systems, written in Python. It solves hourly dispatch with DC power flow, line losses, thermal plant fuel costs, and wind and solar profiles, using the open-source HiGHS solver. The default is one nodal linear programme. The same case can also be cleared as a zonal market, or as a zonal market followed by a nodal solve inside each zone.

The core solves a contiguous block of hours. Unit commitment, storage, hydro budgets, full-year windowed runs and expansion are outside this core.

## Install

```
pip install -e .[test]
```

Requires Python 3.9 or later, numpy, pandas, scipy and highspy.

For large cases, HiGHS's parallel interior-point solver HiPO needs `pip install highspy-extras` (or `pip install open-pcm[hipo]`); pass `method='hipo'` to `LP.solve`, optionally with `tol=1e-6` and `threads=N`. Interior-point solves run without crossover. HiPO currently runs on x86-64; on some ARM servers the extras build stops with an illegal-instruction error, so use `method='ipm'` there.

## Quick start

```
python -m open_pcm examples/synthetic_4node out --year 2027
```

By default this solves the first 168 hours of the year as a nodal model and writes hourly prices, generation by unit, unserved energy and line flows. Pass `--hours 0 47` for a shorter block, or `--mode zonal` to clear the example's two zones (NE and SW) instead.

From Python:

```python
import open_pcm as gd
case = gd.Case('examples/synthetic_4node')
res = gd.dispatch(case, 2027)          # dict of DataFrames: price, gen, use, flows
res = gd.dispatch(case, 2027, range(48))
res = gd.dispatch(case, 2027, range(48), mode='zonal')
```

## Example data

`examples/synthetic_4node` is a **synthetic** four-node system made by `scripts/make_synthetic_case.py` (seed 42). The numbers are invented for testing and do not describe any real power system.

## Case format

A case is a folder of CSV files:

| File | Columns |
|---|---|
| `case.json` | `base_year`, `discount_rate`, `voll_usd_mwh`, `ref_node`, `demand_growth` (annual fraction applied to `peak_MW` from `base_year`, default 0), optional `line_loss_frac`, `flow_cost_usd_mwh`, `mode` (`nodal`, `zonal`, `zonal-nodal`), `zones` (bus id to zone name), `imbalance_penalty_usd_mwh` |
| `nodes.csv` | `node`, `name`, `peak_MW`, optional `zone` |
| `profiles.csv` | 8760 rows: `load` and any wind or solar capacity-factor columns |
| `generators.csv` | `name`, `node`, `type` (`thermal`, `solar`, `wind`), `mw`, `heat_rate`, `fuel_usd_mmbtu`, `vom`, `profile` (for solar/wind) |
| `branches.csv` | `name`, `from`, `to`, `MVA`, `x_pu`, `loss_pct` |

## Solve modes

Set `"mode"` in `case.json`, or pass `--mode`. The flag wins over the file. The default is `nodal`, and a nodal run matches a run that never sets a mode.

### nodal

One DC power-flow model over every bus and branch. Use this when congestion inside the network matters and the whole case still fits in one LP.

### zonal

Buses are grouped into zones, and the market clears on one bus per zone.

Put a `zone` column on `nodes.csv`, or a `zones` object in `case.json` such as `{"1": "NE", "2": "NE", "3": "SW"}`. A non-empty `zone` cell wins. The map fills buses left blank. Every bus needs a zone.

Branches inside a zone are dropped. Branches that join two zones become one transfer: the rating is the sum of the tie-line MVAs, and the loss percent is the average weighted by those ratings. The zonal LP has no DC angles, only these transfer limits, with the same plants, load and unserved energy as the nodal model.

Use zonal for a bidding-zone study, or for a fast look at inter-zone limits when prices inside a zone are not needed.

The price table has one column per zone, and generation rows carry a `zone` column. `net_position_<year>.csv` is hourly load, generation, unserved energy and net export (positive means the zone generates more than it consumes). `transfers.csv` lists the summed tie ratings.

### zonal-nodal

Three stages.

1. **Stage A** is the zonal solve. It fixes each zone's net position and the flow on each inter-zone transfer.
2. **Stage B** solves the nodal DC model of each zone on its own. Zones run in parallel. Tie flows from stage A are fixed injections at the border buses, split across the physical ties in proportion to rating. Plants are redispatched inside the zone, subject to internal line limits. Shortage and spill slacks exist so an impossible schedule still solves; they are priced above the value of lost load, so the model serves load and holds the border whenever it can.
3. **Stage C** balances what stage B could not. It moves the inter-zone flows, still within the summed ratings, to cover each zone's residual. Energy that still does not match is slack, priced at `imbalance_penalty_usd_mwh` (default: `voll_usd_mwh`). The run reports the stage B shortfall and spill, the stage C slack, and the penalty cost.

Use zonal-nodal when the day-ahead result is zonal but internal limits still have to be checked, and the zones are easier to solve separately than as one nodal LP of the whole system.

The main price, generation, unserved and flow files are the stage B nodal results. Stage A is also written as `zonal_prices_<year>.csv` and `zonal_flows_<year>.csv`. Border balancing is `imbalance_<year>.csv` and `balancing_flows_<year>.csv`.

## Tests

```
python -m pytest -q
```

## Licence

MIT. Copyright (c) 2026 Transition Economics Institute. See `LICENSE`. Please cite using `CITATION.cff`.
