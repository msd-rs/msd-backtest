from typing import TYPE_CHECKING, Tuple

import logging
import time

import polars as pl

if TYPE_CHECKING:
  from mbt.select.panel import Panel

logger = logging.getLogger("selector")


class SelectorDataProvider:
  """Provide data to Selector

  数据提供者的职责是把"按股票组织的时序数据"交给选股器, 并把所有易错的对齐
  细节(复权、asof join、缺失填充、块布局)封装在内部。
  """

  date: str | None = None
  """选股日期(YYYY-MM-DD), None 表示最新交易日。 """

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
    * `open/high/low/close`: forward-adjusted (split/bonus/dividend applied).
    * `dividend`, `transfer_shares`, `right_shares`, `right_price`: RAW
      corporate-action values, NOT adjusted. Mixing them with adjusted prices
      (e.g. computing yield from an old DPS after a 10-for-10 transfer) silently
      overstates the result.
    * `dividend` is filled with 0 on non-event days (join mode "zero"), while
      price columns are NaN on suspension days -- two different kinds of "empty".
    * `total_shares` / `tradable_shares`: as-of carried backward from
      `stock_shares` (the last known value at or before that bar).
    * `limited`: 1/-1/0 = limit-up/limit-down/normal, computed on adjusted prices.

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
      * This method calls `alpha.set_ctx()` internally; if you call alpha
        functions afterwards, re-enter the context via `panel.ctx()`.
      * For performance, use py-alpha-lib on the whole block; avoid per-obj
        Python loops.
    """
    pass

  def load_financial(self, objs: list[str], fields: list[str],
                     only_year: bool = True, lastN: int = 12) -> pl.DataFrame:
    """
    load `lastN` financial data for `objs`

    Financial data is provided on a quarterly basis, with dates of 03-31, 06-30,
    09-30, and 12-31. When `only_year` is True only the annual data (12-31) is
    selected. Values are in CNY (yuan) unless the field is a ratio, which is in
    percent (e.g. f205 ROE, f223 debt-to-asset).

    Returns:
      A pl.DataFrame of lastN financial data,
      DataFrame columns ['obj', 'ts', *fields]; missing objs are absent.
    """
    pass

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
    """
    pass


class Selector:
  """选股器基类

  子类按 step00, step01, ... 顺序实现选股步骤, 每步接收上一步的结果并返回新
  的股票列表。约定: **每个 stepXX 必须有 docstring**(会被写入日志), 并把中间
  因子累积到 `self.factors`(pl.DataFrame, 股票代码列名为 `obj`)。
  """

  dp: SelectorDataProvider

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

