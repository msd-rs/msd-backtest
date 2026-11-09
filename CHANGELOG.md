# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.17] - 2026-11-09

### Added

- portfolio account: support dynamic stock pool and history rebalance records

## [0.1.16] - 2026-11-09

### Refactor

- Combine skills to one `msd-strategy-guide`
- alpha lib context
- trading strategy now can filter financial fields to reduce memory usage

## [0.1.15] - 2026-11-07

### Add

- Improve by some feedback
- `fee.json` is optional now.

## [0.1.14] - 2026-11-07

### Add

- `Strategy.prepare(ctx)` pre-calculation lifecycle hook to compute static indicators in `O(N * T)` before the bar-by-bar loop.
- `DataProvider.to_panel()` and `Context.panel()` to unify `DataProvider` and 2D `Panel` views (enabling `panel.resample()` in backtest strategies), plus public `ctx.bar` (`ctx.step`), `ctx.bars`, and `ctx.n_objs` properties on `Context`.
- Shared-cash portfolio backtesting (`PortfolioStrategy`, `PortfolioAccount`, `PortfolioContext`, `PortfolioRunner`, `SelectorStrategy`) and `ctx.rebalance(weights)` API.
- `StaticSelector` to unify fixed stock code lists and `.py` selector strategies under `Selector` execution in `mbt.run.cli`.
- Support coexisting `Selector` and `Strategy` subclasses in a single `.py` file with automatic detection and signature-based `-a key=val` argument dispatch (supporting both `-a k1=v1 k2=v2` and repeated `-a k1=v1 -a k2=v2`).
- Markdown statistical summary report (`build_markdown_summary` with `benchmark_profit` and `Beat Benchmark` stats), `-S/--summary` and `.md` output in `mbt.run.cli`, and `summary` subcommand in `mbt.run.review`.
- Support `exec_mode="next_open"` and `check_limit=True` in `Runner` and `PortfolioRunner`.

### Change

- Unify `-a/--args` in `mbt.run.cli` to `key=value` format consistent with `mbt.select.cli`, and add `-p/--portfolio` and `-c/--cash` options.
- Extend lookback window by 400 days for `stock_shares` and `stock_financial` in `mbt.run.msd.load_data` so backward asof joins populate initial backtest bars with the prior year's latest financial reports and share counts.
- Separate `buy_percents` and `sell_percents` in `RunnerContext` using vectorized indexing so same-bar sell and buy signals do not overwrite each other (executing sells before buys).
- Forward-adjust `vwap` to match the scale of forward-adjusted `open/high/low/close`, while preserving `raw_open` and `raw_vwap`.
- Update `.agents/skills` (`write-strategy`, `write-selector`, and `optimize-strategy`) documentation with new APIs, valuation/financial caveats, and Markdown summary workflow.

### Fix

- Fix `Account.do_buy` tuple unpacking `TypeError` when `cash == 0` and recalculate fee after position downsizing (including `min_commission_buy` lot adjustment).
- Fix rights issue (`right_shares`) cash overdraft and suspended bar (`NaN` price) contamination in `Account.do_operation`.
- Calculate daily limit (`limited`) on unadjusted prices against corporate-action ex-dividend reference prices in `mbt.run.msd` and `mbt.select.msd`.
- Fix `Panel.set_ctx` to explicitly set `start=0, end=self.bars` (avoiding `py-alpha-lib` `end=0` empty window issue during `Strategy.prepare`), and reset `alpha-lib` context in `Runner.run` `finally` blocks with lock protection in `mbt.run.server`.
- Replace positional `pl.concat(..., how="align_left")` with `df.join(fin_df, on="obj", how="left")` in `MsdSelectorDataProvider.load_snapshot` to prevent row misalignment when stocks lack financial records.

## [0.1.13] - 2026-11-05

### Change

- load_financial use enum opt obj column performance
- rename symbol -> obj to standardize terminology.

## [0.1.12] - 2026-11-04

### Change

- selector.DataProvider.load_financial now return long data frame instead lot of small data frames

## [0.1.11] - 2026-11-01

### Add

- Lazy load stocks list
- Selector now can run at specified date, default to last trading date
- multi-period resampling in select panel

## [0.1.10] - 2026-11-01

### Add

- Improve skill docs to make agents work easy and correct.

## [0.1.9] - 2026-10-31

### Add

- `Panel` 2D view (`mbt.select.Panel`) for kline/factor data:
  - 2D views (`column`, `last`, `offset`, `timeline`) with bound `objs` metadata.
  - Automatic `alpha.set_ctx` integration, `panel.ctx()` context manager, and `panel.compute()`.
  - Factor table assembly via `cross_section()` and long-table conversion via `to_frame()`.
  - Full backward compatibility with `(data, objs, bars)` triple unpacking.
- Automatic time-axis alignment via `baseline` parameter in `load_kline` (e.g. `SH000001`), with baseline excluded from the result.
- Detailed layout contract and column semantics documentation in `SelectorDataProvider`.

### Changed

- `MsdSelectorDataProvider.load_kline` returns a `Panel` instance.
- Deduplicate and sort requested objs in `load_kline` for deterministic ordering.
- Widen table display (`tbl_width_chars=240`) when printing factors in selector CLI.

### Fix

- Fix `-a/--arg` key-value parsing bug in selector CLI (`arg.split('=', 1)`).
- Align `groups` and `limit_ratio` in `load_kline` to the returned obj block order.
- Remove accidental `uvicorn` dependency in `mbt/select/selector.py`.
- Allow missing docstrings on selector `stepXX` methods without raising errors.
- Support both `obj` and `obj` column names for factor sorting in selector CLI.
- Normalize `kind` string and generate `status`/`limit_ratio` columns in `normalize_stocks`.


## [0.1.8] - 2026-10-31

### Add

- Skill to install `msd`

## [0.1.7] - 2026-10-30

### Add

- Support loading stock list dynamically from MSD database (`stock_infos`) with local JSON fallback.
- Unit tests for MSD data loading (`tests/test_run_msd.py`).
- Dedicated reference documentation for strategy algorithms and financial indicators.

### Changed

- Switch to native `pymsd.MsdConnection` and SQL `query_join` in selector and strategy runner data providers for improved performance.
- Optimize snapshot queries with native `pymsd.MsdConnection.snapshot()`.
- Filter full-year financial data directly in SQL query (`AND ts.month=12`).
- Modularize AI skills and decouple reference materials.

### Fix

- Fix invalid `encoding` parameter when opening binary JSON report in CLI.
- Handle empty objs and zero bars safely in `DataProvider` and `load_data`.
- Fix single obj condition check in `get_limit_ratio`.


## [0.1.6] - 2026-08-04

### Improve

- Improve select strategy building experience

## [0.1.5] - 2026-07-27

### Fix

- Use utf8 encoding when read/write files
- Excel writer ignore INF/NAN, avoid exception

## [0.1.4] - 2026-07-22

### Add

- `SelectorDataProvider.load_snapshot`  to speed up data loading.


## [0.1.3] - 2026-07-17

### Add

- selector module to select stocks
- a new skill named `write-selector` to guide AI to write selector codes


### Changed

- Use polars instead of pandas for better performance


## [0.1.2] - 2026-07-14

### Fix

- add stock shares aware

## [0.1.1] - 2026-07-10

### Add

- More docs

### Fix

- some typo 

## [0.1.0] - 2026-07-10

Initial release.