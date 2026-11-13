# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from typing import TYPE_CHECKING, Tuple

import logging
import time

import polars as pl

if TYPE_CHECKING:
  from mbt.select.panel import Panel

logger = logging.getLogger("selector")


class SelectorDataProvider:
  """Abstract interface contract for providing market and financial data to a `Selector`.

  The data provider delivers stock-organized time-series and snapshot data to the
  selector, encapsulating critical alignment details (corporate actions and dividend
  adjustments, lookahead-free backward as-of joins based on statutory disclosure dates,
  suspension/missing value padding, and rectangular matrix layouts).

  Concrete Implementations:
    * `mbt.select.MsdSelectorDataProvider`: Primary implementation for standalone selection
      and CLI execution, loading data directly from an MSD database service.
    * `mbt.portfolio._HistoricalPanelSelectorDataProvider`: In-memory implementation used
      during backtest execution (`SelectorStrategy`), providing strictly lookahead-free
      slices of the preloaded historical panel at each bar.
    * Custom mock implementations in unit tests (e.g. `MockSelectorDataProvider`).

  Strategy developers should write selectors against this interface. In both CLI
  and backtest execution, the framework automatically instantiates and injects
  the appropriate concrete provider into `selector.execute(dp, ...)`.
  """

  mode: str = "standalone"
  """Execution mode: 'standalone' (initial screening outside backtest) or 'backtest_slice' (dynamic selection inside backtest loop)."""

  date: str | None = None
  """Selection date (YYYY-MM-DD); None indicates the latest available trading date."""

  @property
  def current_date(self) -> str:
    """Current selection date as YYYY-MM-DD string."""
    return str(self.date)[:10] if self.date else ""

  @property
  def is_backtest(self) -> bool:
    """True if executing inside backtest dynamic slice, False if standalone initial screening."""
    return self.mode == "backtest_slice"

  def load_kline(self, objs: list[str], lastN: int | list[int] = 100,
                 fin_fields: list[str] = [], fin_only_year: bool = True,
                 baseline: str | None = None) -> "Panel":
    """
    load `lastN` kline bars for `objs`, can also with asof joined some `fin_fields`.

    Layout contract
    ---------------
    The result is a *rectangular* matrix in obj-major order:

    ```python
    block i = data[i*bars : (i+1)*bars]   # belongs to returned objs[i]
    ```

    and `len(data) == len(objs) * bars` always holds. It is NOT ragged: missing
    bars are padded with NaN, so "an all-NaN row" means "suspended / not yet
    listed", not "no data".

    The time axis of the WHOLE matrix is defined by the asof **baseline**
    (default: the first obj of the request, or `baseline` when given):

    * If the baseline is suspended, delisted or newly listed, every other obj
      in the request is silently truncated or shifted. No error is raised.
    * Recommended: `baseline="SH000001"` (a full-calendar index). The baseline
      block is removed from the result, so `objs`/`bars` are already clean.
    * The returned `objs` may be REORDERED by the implementation. Never map
      results to your input list positionally; always use the returned order.

    Column semantics
    ----------------
    * `open/high/low/close`: forward-adjusted (split/bonus/dividend applied). Use for technical indicators.
    * `raw_open/raw_high/raw_low/raw_close` (and `price` alias for `raw_close`): UNADJUSTED raw prices.
      Always use `raw_close` when computing valuation multiples with per-share financials (e.g. PB = raw_close / f003)
      or total market capitalization (raw_close * total_shares).
    * `raw_vwap`: unadjusted volume-weighted average price (amount / volume).
    * `dividend`, `transfer_shares`, `right_shares`, `right_price`: RAW
      corporate-action values, NOT adjusted. Mixing them with adjusted prices
      (e.g. computing yield from an old DPS after a 10-for-10 transfer) silently
      overstates the result.
    * `dividend` is filled with 0 on non-event days (join mode "zero"), while
      price columns are NaN on suspension days -- two different kinds of "empty".
    * `total_shares` / `tradable_shares`: as-of carried backward from
      `stock_shares` (the last known value at or before that bar).
    * `limited`: 1/-1/0 = limit-up/limit-down/normal, computed on unadjusted prices with ex-dividend reference price.

    Args:
      objs: the list of objs; the first one is used as the asof baseline
        when `baseline` is None.
      lastN: number of bars.
        * `int`: bars of kline; dividend/shares events are derived as
          `max(lastN//50, 2)`, which is NOT enough for dividend factors.
        * `list[int]` of 3: `[kline_bars, dividend_events, shares_events]`,
          use this when a factor needs an explicit dividend history window,
          e.g. `[750, 40, 2]` for a 3-year trailing dividend sum.
      fin_fields: a list of fin field to take, joined backward via asof.
      fin_only_year: only annual (12-31) financial data is used when True.
      baseline: obj used as the asof baseline, removed from the result.

    Returns:
      A `Panel` exposing `frame`, `objs`, `bars`, plus 2D views
      (`column/last/offset`), `ctx()` for alpha-lib, `cross_section()` for the
      n-row factor table, and `to_frame(key="enum")` for debug/export.
      It still unpacks as the legacy triple `(data, objs, bars)`.

    Notes:
      * This method manages its calculations with `alpha.Context`; if you call alpha
        functions on the returned data, enter the context via `panel.ctx()`.
      * For performance, use py-alpha-lib on the whole block; avoid per-obj
        Python loops.
      * Standalone vs In-Backtest:
        - In standalone mode (e.g. MsdSelectorDataProvider), financial fields are queried
          on-the-fly from the MSD database.
        - In backtesting mode (e.g. _HistoricalPanelSelectorDataProvider), data is sliced
          from preloaded historical arrays. Any financial fields specified in `fin_fields`
          MUST be preloaded upfront by declaring `fin_fields` on your Selector or Strategy
          class, or via CLI `--fin-fields`. Requesting undeclared financial fields in a
          backtest will immediately raise a `ValueError` (Fail-Fast principle).
    """
    raise NotImplementedError(
      f"{self.__class__.__name__}.load_kline is an abstract interface method. "
      "Use a concrete implementation such as MsdSelectorDataProvider (mbt.select.MsdSelectorDataProvider) "
      "or the backtest execution context's historical slice provider."
    )

  def load_financial(self, objs: list[str], fields: list[str],
                     only_year: bool = True, lastN: int = 12) -> pl.DataFrame:
    """
    load `lastN` financial data for `objs`

    Financial data is provided on a quarterly basis, with dates of 03-31, 06-30,
    09-30, and 12-31. When `only_year` is True only the annual data (12-31) is
    selected. Values are in CNY (yuan) unless the field is a ratio, which is in
    percent (e.g. f205 ROE, f223 debt-to-asset).

    Returns:
      A pl.DataFrame of lastN financial data.
      DataFrame columns ['obj', 'ts', 'report_ts', *fields]; missing objs are absent.
      `ts` is the statutory disclosure deadline (used for lookahead-free as-of alignment),
      while `report_ts` is the fiscal period end date (e.g. 12-31).

    Notes:
      * Standalone vs In-Backtest:
        - In standalone mode (MsdSelectorDataProvider), financial statements are queried
          on-demand directly from the MSD database.
        - In backtesting mode (_HistoricalPanelSelectorDataProvider), statements are queried
          from the preloaded backtest financial frame. You MUST declare the required financial
          fields on your Strategy or Selector class via `fin_fields = ['f001', ...]`, or
          provide them via CLI `--fin-fields`. If requested fields are missing in backtest,
          a `ValueError` is raised immediately to prevent silent data loss.
    """
    raise NotImplementedError(
      f"{self.__class__.__name__}.load_financial is an abstract interface method. "
      "Use a concrete implementation such as MsdSelectorDataProvider (mbt.select.MsdSelectorDataProvider) "
      "or the backtest execution context's historical slice provider."
    )

  def load_snapshot(self, objs: list[str], fin_fields: list[str] = [],
                    fin_only_year: bool = True) -> pl.DataFrame:
    """
    Take a snapshot for the given objs at the latest trading day.

    This is the fast path for cross-sectional screening: one row per obj,
    no block layout to worry about.

    Args:
      objs: objs to snapshot
      fin_fields: fields to snapshot from financial data
      fin_only_year: whether to only snapshot annual financial data

    Returns:
      A DataFrame with one row per requested obj, columns combined from
      `load_kline` and `load_financial` (the financial 'ts' column is renamed to
      'f_ts'). objs without a financial report get NULL financial columns.
      Row order follows the implementation, not the input list; join on `obj`.

    Notes:
      * Standalone vs In-Backtest:
        - In standalone mode (MsdSelectorDataProvider), financial fields are queried from MSD.
        - In backtesting mode (_HistoricalPanelSelectorDataProvider), financial fields are taken
          from the preloaded panel cross-section. Any requested `fin_fields` must be declared
          via `fin_fields` on your Strategy/Selector or via CLI `--fin-fields`. Requesting
          undeclared financial fields will raise a `ValueError`.
    """
    raise NotImplementedError(
      f"{self.__class__.__name__}.load_snapshot is an abstract interface method. "
      "Use a concrete implementation such as MsdSelectorDataProvider (mbt.select.MsdSelectorDataProvider) "
      "or the backtest execution context's historical slice provider."
    )


class Selector:
  """Base class for stock selection pipelines.

  Subclasses define selection steps step00, step01, ... in sequence.
  Each step receives the output list of stocks from the previous step and returns a new list.

  Convention: Each step method should have an informative docstring describing its selection criteria.
  Intermediate factors can be accumulated in `self.factors` (pl.DataFrame with 'obj' column).

  Lifecycle & Context:
    * `self.is_backtest`: True when running dynamically inside backtest loop (`ctx.run_selector`),
      False when performing standalone initial screening outside backtest. Use this property to
      cleanly distinguish initial universe creation from in-backtest ranking without relying on
      heuristics like `if not stocks:`.
    * `self.current_date`: Current selection date (YYYY-MM-DD).
    * `self.mode`: Execution mode ('standalone' or 'backtest_slice').
    * `self.dp`: The underlying `SelectorDataProvider` instance.
  """

  dp: SelectorDataProvider

  @property
  def mode(self) -> str:
    """Current execution mode ('standalone' or 'backtest_slice')."""
    return getattr(self.dp, "mode", "standalone")

  @property
  def is_backtest(self) -> bool:
    """True if selector is executing inside a backtest dynamic loop, False if standalone initial screening."""
    return getattr(self.dp, "is_backtest", False)

  @property
  def current_date(self) -> str:
    """Current selection date as YYYY-MM-DD string."""
    return getattr(self.dp, "current_date", "")

  def __init__(self):
    pass

  def execute(self, dp: SelectorDataProvider, init_stocks: list[str]) -> list[str]:
    self.dp = dp
    stocks = init_stocks
    t0 = time.time()
    logger.info(f"start execute selector, {self.__class__.__name__}")
    for i in range(100):
      fn_name = f'step{i:02d}'
      fn = getattr(self, fn_name, None)
      if fn is None:
        break
      t1 = time.time()
      logger.info(f"start {fn_name} with {len(stocks)} stocks")
      # docstring 是约定的说明位, 缺失时不应中断选股流程
      doc = (fn.__doc__ or "").strip()
      if len(doc) > 0:
        logger.info(doc)
      stocks = fn(stocks)
      logger.info(f"done {fn_name}, used {time.time() - t1:0.3f}s, remain {len(stocks)}")
    logger.info(f"done execute selector {self.__class__.__name__}, "
                f"used {time.time() - t0:0.3f}s, remain {len(stocks)}")
    return stocks


class StaticSelector(Selector):
  """总是返回预设股票列表的选股策略，用于将固定股票代码列表统一抽象为 Selector。"""

  def __init__(self, objs: list[str] | None = None):
    super().__init__()
    self.fixed_objs = list(objs) if objs is not None else []

  def step00(self, stocks: list[str]) -> list[str]:
    """直接返回预设的股票列表"""
    if self.fixed_objs:
      return list(self.fixed_objs)
    return list(stocks)

