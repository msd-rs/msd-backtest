# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [0.1.20] - 2026-11-11

- Accept some feedback

## [0.1.19] - 2026-11-10

- financial disclosure statutory fallback: `load_data` automatically shifts financial report period end dates (`report_ts`) to statutory disclosure deadlines (`ts`: 04-30, 08-31, 10-31), completely eliminating lookahead bias where annual/quarterly figures were matched to market bars prior to actual disclosure
- memory guard on financial fields: `fin_fields` defaults to `None` (does not query `stock_financial` by default), preventing accidental memory exhaustion from loading all 180+ fields; queries only when explicitly requested
- in-backtest financial screening: implemented `_HistoricalPanelSelectorDataProvider.load_financial`, enabling `SelectorStrategy` to filter by historical financial reports dynamically inside the backtest loop without lookahead
- rebalance and sell conflict resolution: `PortfolioRunner` prioritizes explicit `ctx.sell()` signals over target weights on the same bar, ensuring stop-loss orders are not silently discarded during rebalancing
- benchmark position decoupling: decoupled benchmark from hardcoded index 0 (`tradable_start=1`); `PortfolioAccount` and `Report` now match benchmark by symbol name and manage tradability via `tradable_mask`
- cli bugfix: added `from __future__ import annotations` and explicit `DataProvider` import in `mbt/run/cli.py`, fixing `NameError` on Python 3.12/3.13
- lookahead prevention default: calling `ctx.panel()` during `execute()` now defaults to `sliced=True` (strictly up to current bar), requiring explicit `allow_future=True` to access future bars
- docstring & nomenclature standardization: centralized system contracts, life cycles, and parameters directly into DocStrings; cleaned up naming artifacts (`("obj", "obj")`)

- dynamic universe pre-loaded pool clarity: documented Super Universe vs. Dynamic Active Whitelist semantics; `set_universe` and `run_selector` emit explicit warnings if requested stocks are not present in the pre-loaded data pool
- whitelist guard on rebalance: `rebalance()` and `PortfolioAccount.rebalance_step` now strictly enforce `ctx.universe` whitelist, capping target weights of non-whitelisted stocks at current holdings (`min(w[g], curr_w)`) and disallowing new purchases, aligning behavior with `ctx.buy()` while supporting natural liquidation (Runoff mode)
- argument typo detection: `instantiate_class` now detects unrecognized kwargs passed via `-a` and logs warnings with fuzzy-match suggestions (e.g. `pb_maxx` -> `did you mean 'pb_max' ?`) against class constructor signatures
- raw price unification: unified `raw_open`, `raw_high`, `raw_low`, `raw_close`, `raw_vwap` across `load_kline` (Panel), `load_snapshot` (DataFrame), and `load_data` (DataProvider). `close` remains forward-adjusted for technical indicators, while `raw_close` provides the unadjusted raw price for fundamental valuation (PB = raw_close / f004, market cap = raw_close * total_shares), preventing distorted valuation multiples from split-adjusted prices
- strategy lifecycle: added `finalize(self, ctx)` hook to `Strategy` and `PortfolioStrategy`, called after the backtest loop completes with settled accounts and full `alpha-lib` range `[0, bars)`. Any dict returned by `finalize()` is attached to `Report.diagnostics` and rendered in Markdown summaries (`## Strategy Diagnostics`) and JSON reports
- warmup bars: fixed `load_data` date binary search (`find_date_index`) to natively support `datetime64` (including `[us]`, `[ns]`, `[ms]`, `[s]`, `[D]`) and microseconds, preventing `TypeError: float() argument must be a string or a real number` and ensuring exact warmup bar calculation without trading start delay
- holdout guard: improved date lookup in `compute_holdout_split` to support `datetime64` and microsecond timestamps

## [0.1.18] - 2026-11-09

- financial alignment: backtest data supports `fin_only_year` to align annual report criteria between static universe selection and in-backtest dynamic selection
- lookahead prevention: calling `ctx.panel()` during `execute()` warns of future bar access; added `ctx.slice_panel()` and `ctx.panel(sliced=True)`
- portfolio report metrics: `trade_count` now reflects actual executed trade records, added `rebalance_count` for rebalance bars, and clarified benchmark vs pool buy-and-hold returns
- portfolio rebalance tolerance: added `tolerance` parameter to `rebalance()` and `PortfolioAccount` to suppress micro-trades within tolerance band
- warmup bars: support `--warmup-bars` (and `Strategy.warmup_bars`) to load N history bars before start date for indicator initialization without trading, reporting performance strictly from target start date
- multi-start evaluation: support specifying multiple `-b` / `--begin` flags or comma-separated start dates (e.g. `-b 2023-01-01 -b 2024-01-01` or `-b 2023-01-01,2024-01-01`, with `--starts` preserved as a backward-compatible alias) to run batch backtests across multiple start dates and generate comparison summary tables
- holdout guard: support `--holdout` to evaluate In-Sample vs Out-of-Sample metrics and overfit ratio
- stock attribution & trade cycles: structured `trades` and `attribution` persisted in JSON and Excel exports; added `## Stock Attribution` table in Markdown report
- review cli: added `trades` and `attribution` subcommands to `mbt.run.review` for easy trade history and stock contribution inspection
- win rate clarity & adaptive summary: disambiguated `daily_win_rate` (bar-level) from `trade_win_rate` (single holding period); removed 15-row `std=0` distribution table when `total == 1` in favor of a clean 2-column `Strategy Metrics` table
- dynamic fee override: support overriding fee parameters dynamically via `-a` (e.g. `commission=0.0002 min_commission=0 tax=0.001`) without needing temporary JSON files

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