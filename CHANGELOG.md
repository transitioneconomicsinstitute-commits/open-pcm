# Changelog

## 0.4.1 (2026-11-04)
- Cases are checked when loaded. Unknown node ids, missing columns, non-numeric or negative values, zero `x_pu`, unknown generator types, profile names that are not in `profiles.csv`, capacity factors outside 0-1, and a bad `ref_node` now raise `CaseError` listing every problem with its file and row, instead of a bare `KeyError` or a divide-by-zero deep in the LP build.
- `check_case(case)` returns the same list without raising. `Case(path, validate=False)` skips the check.

## 0.4.0 (2026-11-03)
- A horizon can be solved as consecutive blocks (`blocks` on `dispatch`, or `--blocks`). Storage levels cross each join, and the tables are stitched into one result.
- Optional `storage.csv`: power, energy, charge efficiency and the initial level. Cases without it are unchanged. Zonal-nodal does not schedule storage.
- `time_limit` on `LP.solve` and `dispatch`. A solve that hits the limit is kept when `accept_gap` is set and the relative gap or primal-dual objective error is within it. `meta['flagged']` is then true.

## 0.3.0 (2026-10-29)
- Solve mode `nodal` (default), `zonal`, or `zonal-nodal`, set in `case.json` or with `--mode`.
- Zonal mode groups buses by a `zone` column or a `zones` map. Inter-zone transfer limits are the summed tie-line ratings.
- Zonal-nodal runs the zonal market, then a nodal solve in each zone with the border flows fixed, then a border balancing step that reports slack and penalty cost.

## 0.2.0 (2026-10-23)
- `LP.solve` accepts `method='hipo'` (HiGHS's parallel interior-point solver; install the optional extra `open-pcm[hipo]`, i.e. `highspy-extras`), plus `tol` (IPM optimality tolerance) and `threads`. Interior-point methods (`ipm`, `ipx`, `hipo`) run without crossover and retry with crossover if the solve is not optimal.
- Demand growth from `base_year` is applied to node load peaks.
- `crf` handles a zero discount rate and validates its inputs.
- Dispatch rejects empty or out-of-range hour ranges.
- Project renamed to OPEN-PCM (package `open_pcm`, CLI `open-pcm`).

## 0.1.0 (2026-10-15)
- First release: hourly nodal production cost model with DC power flow, losses, thermal fuel costs and wind/solar profiles, solved as one sparse LP with HiGHS.
