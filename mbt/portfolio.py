from __future__ import annotations

from typing import Callable
import numpy as np
import polars as pl
import alpha as al

from .account import (
  ACTION_BUY,
  ACTION_KEEP,
  ACTION_SELL,
  Fee,
  Operation,
  Position,
)
from .data_provider import DataProvider
from .select.panel import Panel
from .select.selector import Selector, SelectorDataProvider
from .strategy import Context, Strategy, PortfolioStrategy


class PortfolioAccount:
  """Shared-cash multi-asset portfolio account for backtesting."""

  def __init__(
    self,
    initial_cash: float,
    dp: DataProvider,
    /,
    fee: Fee | None = None,
    unit: int = 100,
    slippage: float = 0.0,
    tradable_start: int = 0,
  ):
    self.initial_cash = float(initial_cash)
    self.cash = float(initial_cash)
    self.dp = dp
    self.fee = fee if fee is not None else Fee()
    self.unit = unit
    self.slippage = slippage
    self.tradable_start = max(0, int(tradable_start))
    self.deducted = 0.0

    self.groups = dp.groups
    self.bars = dp.bars
    self.positions = [Position.zero() for _ in range(self.groups)]
    self.dates = (
      dp.dates[: self.bars].astype("datetime64[s]").astype(int)
      if self.bars > 0
      else np.array([], dtype=int)
    )

    self.portfolio_assets = np.full(self.bars, self.initial_cash, dtype=np.float64)
    self.portfolio_cash = np.full(self.bars, self.initial_cash, dtype=np.float64)
    self.portfolio_ror = np.zeros(self.bars, dtype=np.float64)

    self.actions = dp.all("actions", create_if_not_exist=np.int32)
    self.action_shares = dp.all("action_shares", create_if_not_exist=np.float64)
    self.actions.fill(ACTION_KEEP)
    self.action_shares.fill(0.0)

    self.dividends = (
      dp.all("dividend")
      if "dividend" in dp.data
      else np.zeros(len(dp.dates), dtype=np.float64)
    )
    self.transfer_shares = (
      dp.all("transfer_shares")
      if "transfer_shares" in dp.data
      else np.zeros(len(dp.dates), dtype=np.float64)
    )
    self.right_shares = (
      dp.all("right_shares")
      if "right_shares" in dp.data
      else np.zeros(len(dp.dates), dtype=np.float64)
    )
    self.right_prices = (
      dp.all("right_price")
      if "right_price" in dp.data
      else np.zeros(len(dp.dates), dtype=np.float64)
    )

  @property
  def market_value(self) -> float:
    return float(sum(p.value for p in self.positions))

  @property
  def asset(self) -> float:
    return self.cash + self.market_value

  def weights(self) -> np.ndarray:
    total = self.asset
    if total <= 0:
      return np.zeros(self.groups, dtype=np.float64)
    return np.array([p.value / total for p in self.positions], dtype=np.float64)

  def apply_corporate_actions(self, ts: int):
    for g in range(self.groups):
      pos = self.positions[g]
      if pos.shares <= 0:
        continue
      j = g * self.bars + ts
      div = self.dividends[j]
      trans = self.transfer_shares[j]
      r_share = self.right_shares[j]
      r_price = self.right_prices[j]

      if div > 0 and not np.isnan(div):
        self.cash += pos.shares * div
      if trans > 0 and not np.isnan(trans):
        pos.shares += pos.shares * trans
      if r_share > 0 and not np.isnan(r_share):
        max_r = pos.shares * r_share
        if r_price > 0 and not np.isnan(r_price):
          affordable = int(self.cash / r_price) if self.cash > 0 else 0
          actual_r = min(max_r, float(affordable))
          self.cash -= actual_r * r_price
        else:
          actual_r = max_r
        pos.shares += actual_r

  def _sell_group(self, g: int, ts: int, shares_to_sell: float, deal_price: float):
    pos = self.positions[g]
    if pos.shares <= 0 or shares_to_sell <= 0 or np.isnan(deal_price) or deal_price <= 0:
      return
    actual_shares = min(pos.shares, float(shares_to_sell))
    if actual_shares <= 0:
      return
    exec_price = deal_price * (1.0 - self.slippage)
    sell_pos = Position(actual_shares, exec_price)
    fee_item = self.fee.get(self.dates[ts])
    fee = fee_item.sell(sell_pos.value)
    self.cash += sell_pos.value - fee
    self.positions[g] = pos - sell_pos
    self.deducted += fee
    j = g * self.bars + ts
    self.actions[j] = ACTION_SELL
    self.action_shares[j] = actual_shares

  def _buy_group(self, g: int, ts: int, cash_budget: float, deal_price: float):
    if self.cash <= 0 or cash_budget <= 0 or np.isnan(deal_price) or deal_price <= 0:
      return
    budget = min(self.cash, float(cash_budget))
    exec_price = deal_price * (1.0 + self.slippage)
    fee_item = self.fee.get(self.dates[ts])
    buy_pos = Position.from_cash(budget, exec_price, self.unit)
    fee = fee_item.buy(buy_pos.value)
    while buy_pos.shares > 0 and buy_pos.value + fee > budget:
      buy_pos = Position(max(buy_pos.shares - self.unit, 0.0), exec_price)
      fee = fee_item.buy(buy_pos.value)

    if buy_pos.shares <= 0:
      return

    self.cash -= buy_pos.value + fee
    self.positions[g] = self.positions[g] + buy_pos
    self.deducted += fee
    j = g * self.bars + ts
    self.actions[j] = ACTION_BUY
    self.action_shares[j] = buy_pos.shares

  def update_mark_to_market(self, ts: int, mark_prices: np.ndarray):
    for g in range(self.groups):
      p = mark_prices[g]
      if not np.isnan(p) and p > 0:
        self.positions[g].price = float(p)
    self.portfolio_cash[ts] = self.cash
    self.portfolio_assets[ts] = self.asset
    self.portfolio_ror[ts] = (
      (self.asset / self.initial_cash - 1.0) if self.initial_cash > 0 else 0.0
    )

  def rebalance_step(
    self,
    ts: int,
    target_weights: np.ndarray,
    deal_prices: np.ndarray,
    mark_prices: np.ndarray,
    limited: np.ndarray | None = None,
  ):
    weights = np.maximum(np.nan_to_num(np.asarray(target_weights, dtype=np.float64), nan=0.0), 0.0)
    if self.tradable_start > 0:
      weights[: self.tradable_start] = 0.0
    w_sum = float(np.sum(weights))
    if w_sum > 1.0:
      weights = weights / w_sum

    # Update current valuation with deal_prices where valid before computing targets
    for g in range(self.tradable_start, self.groups):
      dp_g = deal_prices[g]
      if not np.isnan(dp_g) and dp_g > 0 and self.positions[g].shares > 0:
        self.positions[g].price = float(dp_g)

    total_equity = self.asset
    target_values = total_equity * weights

    # Pass 1: Sells first to free up cash
    for g in range(self.tradable_start, self.groups):
      lim = limited[g] if limited is not None else 0
      if lim == -1:
        continue
      dp_g = deal_prices[g]
      if np.isnan(dp_g) or dp_g <= 0:
        continue
      curr_val = self.positions[g].shares * dp_g
      tgt_val = target_values[g]
      if tgt_val == 0.0 and self.positions[g].shares > 0:
        self._sell_group(g, ts, self.positions[g].shares, dp_g)
      elif curr_val > tgt_val:
        excess_val = curr_val - tgt_val
        shares_to_sell = int(excess_val / dp_g)
        shares_to_sell = shares_to_sell - (shares_to_sell % self.unit)
        if shares_to_sell > 0:
          self._sell_group(g, ts, float(shares_to_sell), dp_g)

    # Pass 2: Buys using available cash
    buy_deficits = []
    for g in range(self.tradable_start, self.groups):
      lim = limited[g] if limited is not None else 0
      if lim == 1:
        continue
      dp_g = deal_prices[g]
      if np.isnan(dp_g) or dp_g <= 0:
        continue
      curr_val = self.positions[g].shares * dp_g
      tgt_val = target_values[g]
      if tgt_val > curr_val:
        buy_deficits.append((g, tgt_val - curr_val, dp_g))

    total_deficit = sum(d[1] for d in buy_deficits)
    if total_deficit > 0 and self.cash > 0:
      scale = min(1.0, self.cash / total_deficit)
      for g, deficit, dp_g in buy_deficits:
        self._buy_group(g, ts, deficit * scale, dp_g)

    self.update_mark_to_market(ts, mark_prices)


class PortfolioContext(Context):
  """Strategy context for shared-cash portfolio backtesting."""

  def __init__(self, account: PortfolioAccount):
    self.account = account
    self.dp = account.dp
    n = len(self.dp.dates)
    self.buy_percents = np.zeros(n, dtype=np.float64)
    self.sell_percents = np.zeros(n, dtype=np.float64)
    self.target_weights = np.full((self.dp.bars, self.dp.groups), np.nan, dtype=np.float64)
    self._group_offsets = np.arange(self.dp.groups, dtype=np.int64) * self.dp.bars

  def objs(self) -> list[str]:
    return self.dp.objs

  def panel(self) -> Panel:
    return self.dp.to_panel()

  def data(self, name: str, /, obj: str | None = None) -> np.ndarray:
    if name in self.dp.data:
      return self.dp.all(name, obj=obj)
    raise ValueError(f"data '{name}' not found")

  def _resolve_bar_flags(self, flags: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    idx = self._group_offsets + self.dp.i
    arr = np.asarray(flags, dtype=bool)
    if arr.ndim == 1 and len(arr) == self.dp.groups:
      return idx, arr
    return idx, arr[idx]

  def buy(self, flags: np.ndarray, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    active = idx[mask]
    if len(active) > 0:
      self.buy_percents[active] = percent

  def sell(self, flags: np.ndarray, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    active = idx[mask]
    if len(active) > 0:
      self.sell_percents[active] = percent

  def rebalance(self, weights: np.ndarray):
    """Set target portfolio weights for the current bar (shape: (groups,) or (groups * bars,))."""
    arr = np.asarray(weights, dtype=np.float64)
    ts_start = self.account.tradable_start
    if arr.ndim == 1 and len(arr) == self.dp.groups:
      w = arr.copy()
      if ts_start > 0:
        w[:ts_start] = 0.0
      self.target_weights[self.dp.i] = w
    elif ts_start > 0 and arr.ndim == 1 and len(arr) == self.dp.groups - ts_start:
      w = np.zeros(self.dp.groups, dtype=np.float64)
      w[ts_start:] = arr
      self.target_weights[self.dp.i] = w
    elif arr.ndim == 1 and len(arr) == self.dp.groups * self.dp.bars:
      idx = self._group_offsets + self.dp.i
      w = arr[idx].copy()
      if ts_start > 0:
        w[:ts_start] = 0.0
      self.target_weights[self.dp.i] = w
    else:
      raise ValueError(
        f"weights must have shape ({self.dp.groups},) or ({self.dp.groups * self.dp.bars},)"
      )


class PortfolioRunner:
  """Runner for multi-stock shared-cash portfolio strategies."""

  def __init__(
    self,
    account: PortfolioAccount,
    /,
    exec_mode: str = "close",
    check_limit: bool = True,
  ):
    if exec_mode not in ("close", "next_open"):
      raise ValueError(f"unsupported exec_mode: {exec_mode}, expected 'close' or 'next_open'")
    self.ctx = PortfolioContext(account)
    self.exec_mode = exec_mode
    self.check_limit = check_limit

  def run(self, strategy: Strategy):
    dp = self.ctx.dp
    acc = self.ctx.account
    if dp.groups == 0 or dp.bars == 0:
      return

    al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars)
    strategy.prepare(self.ctx)
    dp.reset()
    al.set_ctx(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)

    prices = dp.all("price")
    if self.exec_mode == "next_open":
      if "raw_open" in dp.data:
        deal_prices_all = dp.all("raw_open")
      elif "open" in dp.data:
        deal_prices_all = dp.all("open")
      else:
        deal_prices_all = prices
    else:
      deal_prices_all = prices

    limited_all = dp.all("limited") if (self.check_limit and "limited" in dp.data) else None
    offsets = self.ctx._group_offsets

    # Initialize bar 0 mark-to-market
    acc.update_mark_to_market(0, prices[offsets])

    try:
      for i in dp:
        strategy.execute(self.ctx)
        acc.apply_corporate_actions(i)

        sig_i = i - 1 if self.exec_mode == "next_open" else i
        idx_now = offsets + i
        mark_p = prices[idx_now]
        deal_p = deal_prices_all[idx_now]
        lim_now = limited_all[idx_now] if limited_all is not None else None

        if sig_i < 0:
          acc.update_mark_to_market(i, mark_p)
          continue

        # 1. Check if explicit target_weights were provided at sig_i
        tgt_w = self.ctx.target_weights[sig_i]
        if not np.all(np.isnan(tgt_w)):
          acc.rebalance_step(i, tgt_w, deal_p, mark_p, lim_now)
          continue

        # 2. Otherwise process buy/sell signals on shared cash pool
        idx_sig = offsets + sig_i
        sell_p = self.ctx.sell_percents[idx_sig]
        buy_p = self.ctx.buy_percents[idx_sig]

        # Execute sells first
        for g in range(acc.tradable_start, dp.groups):
          if sell_p[g] > 0:
            if lim_now is not None and lim_now[g] == -1:
              continue
            shares_to_sell = acc.positions[g].shares * sell_p[g]
            if sell_p[g] < 1.0:
              shares_to_sell = float(int(shares_to_sell) - (int(shares_to_sell) % acc.unit))
            acc._sell_group(g, i, shares_to_sell, deal_p[g])

        # Execute buys sharing available cash across active buy signals
        buy_groups = [
          g
          for g in range(acc.tradable_start, dp.groups)
          if buy_p[g] > 0 and (lim_now is None or lim_now[g] != 1)
        ]
        if len(buy_groups) > 0 and acc.cash > 0:
          total_req = sum(buy_p[g] for g in buy_groups)
          alloc_cash = acc.cash * min(1.0, total_req / len(buy_groups))
          for g in buy_groups:
            budget = alloc_cash * (buy_p[g] / total_req)
            acc._buy_group(g, i, budget, deal_p[g])

        acc.update_mark_to_market(i, mark_p)
    finally:
      al.set_ctx(groups=1, flags=al.FLAG_SKIP_NAN, start=0, end=0)


class _HistoricalPanelSelectorDataProvider(SelectorDataProvider):
  """In-memory SelectorDataProvider sliced at current backtest bar for SelectorStrategy."""

  def __init__(self, panel_slice: Panel):
    super().__init__()
    self._panel = panel_slice

  def load_kline(
    self,
    objs: list[str],
    lastN: int | list[int] = 100,
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
    baseline: str | None = None,
  ) -> Panel:
    n_bars = lastN if isinstance(lastN, int) else int(lastN[0])
    n_bars = min(n_bars, self._panel.bars)
    start_bar = self._panel.bars - n_bars

    target_set = set(objs) if objs else set(self._panel.objs)
    selected_indices = [
      idx for idx, s in enumerate(self._panel.objs) if s in target_set
    ]
    selected_objs = [self._panel.objs[idx] for idx in selected_indices]
    if not selected_objs:
      raise ValueError("no matching objs in historical panel")

    new_cols = {}
    for col in self._panel.frame.columns:
      arr_2d = self._panel.column(col)[selected_indices, start_bar:]
      new_cols[col] = arr_2d.ravel()
    return Panel(pl.DataFrame(new_cols), selected_objs, n_bars)

  def load_snapshot(
    self,
    objs: list[str],
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
  ) -> pl.DataFrame:
    p = self.load_kline(objs, lastN=1)
    cols = {col: p.last(col) for col in p.frame.columns}
    return p.cross_section(**cols)


class SelectorStrategy(PortfolioStrategy):
  """Adapter that runs a Selector (or selection callable) periodically during a portfolio backtest."""

  def __init__(
    self,
    selector: Selector | Callable[[Panel], list[str]],
    /,
    rebalance_period: int = 20,
    min_bars: int = 1,
    max_holdings: int | None = None,
  ):
    super().__init__()
    self.selector = selector
    self.rebalance_period = max(1, int(rebalance_period))
    self.min_bars = max(1, int(min_bars))
    self.max_holdings = max_holdings
    self._full_panel: Panel | None = None

  def prepare(self, ctx: Context):
    self._full_panel = ctx.panel()

  def _slice_panel(self, end_bar: int) -> Panel:
    assert self._full_panel is not None
    new_cols = {}
    for col in self._full_panel.frame.columns:
      arr_2d = self._full_panel.column(col)[:, :end_bar]
      new_cols[col] = arr_2d.ravel()
    return Panel(pl.DataFrame(new_cols), list(self._full_panel.objs), end_bar)

  def execute(self, ctx: PortfolioContext):  # type: ignore[override]
    i = ctx.dp.i
    if i < self.min_bars or (i - self.min_bars) % self.rebalance_period != 0:
      return

    sub_panel = self._slice_panel(i + 1)
    if isinstance(self.selector, Selector):
      dp_slice = _HistoricalPanelSelectorDataProvider(sub_panel)
      selected = self.selector.execute(dp_slice, list(sub_panel.objs))
    else:
      selected = self.selector(sub_panel)

    if self.max_holdings is not None and len(selected) > self.max_holdings:
      selected = selected[: self.max_holdings]

    weights = np.zeros(sub_panel.n_objs, dtype=np.float64)
    if len(selected) > 0:
      w = 1.0 / len(selected)
      for s in selected:
        if s in sub_panel._index:
          weights[sub_panel.pick(s)] = w

    ctx.rebalance(weights)
    # Restore alpha-lib context for Runner's current bar
    al.set_ctx(groups=ctx.dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=i + 1)
