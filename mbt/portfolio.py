from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable
import logging
import numpy as np
import polars as pl
import alpha as al

logger = logging.getLogger(__name__)

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


@dataclass
class PortfolioRecord:
  """Record of an executed trade/rebalance in a PortfolioAccount."""

  bar: int
  ts: int
  group: int
  obj: str
  action: str  # "BUY" or "SELL"
  shares: float
  deal_price: float
  exec_price: float
  amount: float
  fee: float
  cash: float
  pos_shares: float

  def __getitem__(self, item: str):
    return getattr(self, item)

  def to_dict(self) -> dict:
    return asdict(self)


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
    tolerance: float = 0.0,
    benchmark: str | None = None,
  ):
    """Shared-cash multi-asset portfolio account for backtesting.

    Args:
      initial_cash: Starting cash for the portfolio.
      dp: DataProvider supplying prices and corporate action data.
      fee: Fee configuration.
      unit: Lot size (shares per trading unit, default 100).
      slippage: Slippage fraction applied to executions.
      tradable_start: (Legacy) Index of the first tradable group. Defaults to 0.
      tolerance: Minimum weight deviation threshold required to trigger rebalancing.
      benchmark: Obj code of benchmark index (e.g. 'SH000300'). When provided, its
        corresponding group is marked untradable (`tradable_mask=False`) and used
        exclusively for benchmark tracking.
    """
    self.initial_cash = float(initial_cash)
    self.cash = float(initial_cash)
    self.dp = dp
    self.fee = fee if fee is not None else Fee()
    self.unit = unit
    self.slippage = slippage
    self.tolerance = float(tolerance)
    self.deducted = 0.0
    self.records: list[PortfolioRecord] = []

    self.groups = dp.groups
    self.bars = dp.bars
    self.positions = [Position.zero() for _ in range(self.groups)]
    self.entry_bars = np.full(self.groups, -1, dtype=np.int64)
    self._obj_to_group: dict[str, int] = (
      {s: i for i, s in enumerate(dp.objs)} if hasattr(dp, "objs") else {}
    )
    self.benchmark_obj = benchmark
    self.benchmark_idx: int | None = None
    if benchmark is not None and benchmark in self._obj_to_group:
      self.benchmark_idx = self._obj_to_group[benchmark]

    self.tradable_mask = np.ones(self.groups, dtype=bool)
    if self.benchmark_idx is not None:
      self.tradable_mask[self.benchmark_idx] = False
    elif tradable_start > 0:
      self.tradable_mask[: max(0, int(tradable_start))] = False
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
  def tradable_start(self) -> int:
    """Index of the first tradable group, maintained for backward compatibility."""
    for g in range(self.groups):
      if self.tradable_mask[g]:
        return g
    return 0

  @tradable_start.setter
  def tradable_start(self, val: int):
    self.tradable_mask = np.ones(self.groups, dtype=bool)
    if val > 0:
      self.tradable_mask[: max(0, int(val))] = False

  @property
  def market_value(self) -> float:
    return float(sum(p.value for g, p in enumerate(self.positions) if self.tradable_mask[g]))

  @property
  def asset(self) -> float:
    return self.cash + self.market_value

  def weights(self) -> np.ndarray:
    total = self.asset
    if total <= 0:
      return np.zeros(self.groups, dtype=np.float64)
    return np.array([p.value / total if self.tradable_mask[g] else 0.0 for g, p in enumerate(self.positions)], dtype=np.float64)

  @property
  def held_groups(self) -> list[int]:
    """List of group indices that currently have held shares (> 0)."""
    return [
      g
      for g in range(self.groups)
      if self.tradable_mask[g] and self.positions[g].shares > 0
    ]

  @property
  def held_objs(self) -> list[str]:
    """List of obj symbols that currently have held shares (> 0)."""
    objs = self.dp.objs
    return [objs[g] for g in self.held_groups if g < len(objs)]

  def has_position(self, obj: str | int) -> bool:
    """Check whether the account currently holds shares in obj or group."""
    if isinstance(obj, str):
      g = self._obj_to_group.get(obj)
      return g is not None and self.positions[g].shares > 0
    elif isinstance(obj, (int, np.integer)):
      return 0 <= obj < self.groups and self.positions[obj].shares > 0
    return False

  def position(self, obj: str | int, current_bar: int | None = None) -> Position:
    """Get the current Position object for an obj or group index."""
    if isinstance(obj, str):
      g = self._obj_to_group.get(obj)
      if g is not None and 0 <= g < len(self.positions):
        pos = self.positions[g]
        pos.holding_days = self.holding_days(g, current_bar=current_bar)
        if current_bar is not None and 0 <= current_bar < self.bars:
          j = g * self.bars + current_bar
          p = self.dp.all("price")[j]
          if not np.isnan(p) and p > 0:
            pos.raw_mark_price = float(p)
        return pos
      return Position.zero()
    elif isinstance(obj, (int, np.integer)):
      if 0 <= obj < len(self.positions):
        pos = self.positions[obj]
        pos.holding_days = self.holding_days(obj, current_bar=current_bar)
        if current_bar is not None and 0 <= current_bar < self.bars:
          j = int(obj) * self.bars + current_bar
          p = self.dp.all("price")[j]
          if not np.isnan(p) and p > 0:
            pos.raw_mark_price = float(p)
        return pos
      return Position.zero()
    return Position.zero()

  def runoff_objs(self, universe: set[str] | list[str] | None = None) -> list[str]:
    """Held stocks that are no longer in the active universe whitelist."""
    if universe is None:
      return []
    u_set = set(universe)
    return [s for s in self.held_objs if s not in u_set]

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
        pos.dividends += pos.shares * div
      if trans > 0 and not np.isnan(trans):
        old_shares = pos.shares
        pos.shares += pos.shares * trans
        if pos.shares > 0:
          ratio = old_shares / pos.shares
          pos.raw_cost_price *= ratio
          pos._adj_cost_price *= ratio
      if r_share > 0 and not np.isnan(r_share):
        max_r = pos.shares * r_share
        if r_price > 0 and not np.isnan(r_price):
          affordable = int(self.cash / r_price) if self.cash > 0 else 0
          actual_r = min(max_r, float(affordable))
          self.cash -= actual_r * r_price
          right_cost = actual_r * r_price
        else:
          actual_r = max_r
          right_cost = 0.0
        if actual_r > 0:
          old_cost = pos.cost_basis
          pos.shares += actual_r
          if pos.shares > 0:
            pos.raw_cost_price = (old_cost + right_cost) / pos.shares
            if pos.raw_cost_price > 0:
              pos._adj_cost_price = pos.raw_cost_price * (pos._adj_cost_price / pos.raw_cost_price)

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
    if self.positions[g].shares <= 0:
      self.entry_bars[g] = -1
    self.deducted += fee
    j = g * self.bars + ts
    self.actions[j] = ACTION_SELL
    self.action_shares[j] = actual_shares

    obj = self.dp.objs[g] if (hasattr(self.dp, "objs") and g < len(self.dp.objs)) else str(g)
    ts_val = int(self.dates[ts]) if (len(self.dates) > 0 and ts < len(self.dates)) else ts
    self.records.append(
      PortfolioRecord(
        bar=ts,
        ts=ts_val,
        group=g,
        obj=obj,
        action="SELL",
        shares=actual_shares,
        deal_price=float(deal_price),
        exec_price=float(exec_price),
        amount=float(sell_pos.value),
        fee=float(fee),
        cash=float(self.cash),
        pos_shares=float(self.positions[g].shares),
      )
    )

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

    # Calculate forward-adjustment ratio if available
    j = g * self.bars + ts
    adj_cost = exec_price
    if hasattr(self.dp, "data") and "close" in self.dp.data and "price" in self.dp.data:
      raw_p = self.dp.data["price"][j]
      fw_p = self.dp.data["close"][j]
      if raw_p > 0 and not np.isnan(raw_p) and not np.isnan(fw_p):
        adj_cost = exec_price * (fw_p / raw_p)
    buy_pos._adj_cost_price = float(adj_cost)

    if self.positions[g].shares <= 0:
      self.entry_bars[g] = ts

    self.cash -= buy_pos.value + fee
    self.positions[g] = self.positions[g] + buy_pos
    self.deducted += fee
    j = g * self.bars + ts
    self.actions[j] = ACTION_BUY
    self.action_shares[j] = buy_pos.shares

    obj = self.dp.objs[g] if (hasattr(self.dp, "objs") and g < len(self.dp.objs)) else str(g)
    ts_val = int(self.dates[ts]) if (len(self.dates) > 0 and ts < len(self.dates)) else ts
    self.records.append(
      PortfolioRecord(
        bar=ts,
        ts=ts_val,
        group=g,
        obj=obj,
        action="BUY",
        shares=float(buy_pos.shares),
        deal_price=float(deal_price),
        exec_price=float(exec_price),
        amount=float(buy_pos.value),
        fee=float(fee),
        cash=float(self.cash),
        pos_shares=float(self.positions[g].shares),
      )
    )

  def trades_df(self) -> pl.DataFrame:
    """Return trade and rebalance history as a Polars DataFrame."""
    if not self.records:
      return pl.DataFrame(
        schema={
          "bar": pl.Int64,
          "ts": pl.Int64,
          "group": pl.Int64,
          "obj": pl.Utf8,
          "action": pl.Utf8,
          "shares": pl.Float64,
          "deal_price": pl.Float64,
          "exec_price": pl.Float64,
          "amount": pl.Float64,
          "fee": pl.Float64,
          "cash": pl.Float64,
          "pos_shares": pl.Float64,
        }
      )
    return pl.DataFrame([r.to_dict() for r in self.records])

  def records_df(self) -> pl.DataFrame:
    """Alias for trades_df()."""
    return self.trades_df()

  def holding_ror(self, obj: str | int) -> float:
    """Return the current holding rate of return for obj/group as a float (e.g. -0.05 for -5%)."""
    return self.position(obj).holding_ror

  def holding_pnl(self, obj: str | int) -> float:
    """Return the current holding profit/loss amount for obj/group."""
    return self.position(obj).holding_pnl

  def cost_price(self, obj: str | int, adjusted: bool = False) -> float:
    """Return acquisition cost price.

    Args:
      obj: Symbol or group index.
      adjusted: If True, returns forward-adjusted equivalent cost (adj_cost_price),
        safe to compare with `ctx.close`. If False, returns raw execution cost price.
    """
    pos = self.position(obj)
    return pos.adj_cost_price if adjusted else pos.raw_cost_price

  def holding_days(self, obj: str | int, current_bar: int | None = None) -> int:
    """Return number of bars since position was first acquired."""
    g = self._resolve_group_idx(obj)
    if g is None or self.entry_bars[g] < 0:
      return 0
    cur = self.bars if current_bar is None else current_bar
    return max(0, cur - int(self.entry_bars[g]))

  def _resolve_group_idx(self, obj: str | int) -> int | None:
    if isinstance(obj, str):
      return self._obj_to_group.get(obj)
    elif isinstance(obj, (int, np.integer)):
      if 0 <= obj < self.groups:
        return int(obj)
    return None

  def update_mark_to_market(self, ts: int, mark_prices: np.ndarray):
    for g in range(self.groups):
      p = mark_prices[g]
      if not np.isnan(p) and p > 0:
        self.positions[g].raw_mark_price = float(p)
      if self.positions[g].shares > 0 and self.entry_bars[g] >= 0:
        self.positions[g].holding_days = max(0, ts - int(self.entry_bars[g]))
      else:
        self.positions[g].holding_days = 0
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
    tolerance: float | None = None,
    universe: set[str] | None = None,
  ):
    """Execute a single rebalancing step to align current holdings with target_weights.

    Execution Flow:
    1. Zero out weights for non-tradable assets (such as benchmark index).
    2. Normalize target weights if sum exceeds 1.0.
    3. Tolerance check: if deviation between current weight and target is within tolerance,
       maintain current position to avoid turnover costs (except for target 0 liquidation).
    4. Pass 1 (Sells): Liquidations and reductions executed first to release cash.
    5. Pass 2 (Buys): Buys scaled by available cash pool if total buy capital exceeds available cash.
    6. Mark-to-market valuation update.
    """
    tol = self.tolerance if tolerance is None else float(tolerance)
    weights = np.maximum(np.nan_to_num(np.asarray(target_weights, dtype=np.float64), nan=0.0), 0.0)
    weights[~self.tradable_mask] = 0.0
    w_sum = float(np.sum(weights))
    if w_sum > 1.0:
      weights = weights / w_sum

    # Update current valuation with deal_prices where valid before computing targets
    for g in range(self.groups):
      if not self.tradable_mask[g]:
        continue
      dp_g = deal_prices[g]
      if not np.isnan(dp_g) and dp_g > 0 and self.positions[g].shares > 0:
        self.positions[g].raw_mark_price = float(dp_g)

    total_equity = self.asset
    target_values = total_equity * weights

    # Apply tolerance band: keep current position value if deviation is within tolerance
    if tol > 0.0 and total_equity > 0.0:
      for g in range(self.groups):
        if not self.tradable_mask[g]:
          continue
        tgt_w = weights[g]
        # Explicit liquidation (target 0) is never suppressed
        if tgt_w == 0.0:
          continue
        dp_g = deal_prices[g]
        if np.isnan(dp_g) or dp_g <= 0:
          continue
        curr_val = self.positions[g].shares * dp_g
        curr_w = curr_val / total_equity
        if abs(tgt_w - curr_w) <= tol:
          target_values[g] = curr_val

    # Pass 1: Sells first to free up cash
    for g in range(self.groups):
      if not self.tradable_mask[g]:
        continue
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
    for g in range(self.groups):
      if not self.tradable_mask[g]:
        continue
      if universe is not None and self.dp.objs[g] not in universe:
        continue
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


class _HistoricalPanelSelectorDataProvider(SelectorDataProvider):
  """In-memory SelectorDataProvider sliced at current backtest bar for SelectorStrategy."""

  def __init__(
    self,
    panel_slice: Panel,
    fin_only_year: bool = False,
    financial_frame: pl.DataFrame | None = None,
  ):
    super().__init__()
    self.mode = "backtest_slice"
    self._panel = panel_slice
    self._fin_only_year = fin_only_year
    self._financial_frame = financial_frame
    if panel_slice.bars > 0 and "ts" in panel_slice.columns:
      self.date = str(panel_slice.timeline()[-1])[:10]

  @property
  def financial_frame(self) -> pl.DataFrame | None:
    """Preloaded historical financial reports frame (if available)."""
    return self._financial_frame

  def load_financial(
    self,
    objs: list[str],
    fields: list[str],
    only_year: bool = True,
    lastN: int = 12,
  ) -> pl.DataFrame:
    """Load up to `lastN` historical financial reports for `objs` without lookahead bias.

    Filters records strictly by `ts <= current_bar_date` where `ts` is the
    announcement / statutory disclosure date.

    Args:
      objs: List of stock codes to query.
      fields: Financial field names (e.g. ['f001', 'f205']).
      only_year: If True, only select annual reports (report_ts.month == 12).
      lastN: Maximum number of historical records per stock.

    Returns:
      pl.DataFrame with columns ['obj', 'ts', 'report_ts', *fields].
    """
    if not objs or not fields:
      return pl.DataFrame()

    if self._financial_frame is None or len(self._financial_frame) == 0:
      raise ValueError(
        f"In-backtest load_financial requested field(s) {fields}, but no financial data was preloaded in backtest. "
        f"Please declare fin_fields on your Strategy or Selector class (e.g. fin_fields={list(fields)}) or pass via CLI."
      )

    missing = [f for f in fields if f not in self._financial_frame.columns]
    if missing:
      raise ValueError(
        f"In-backtest load_financial requested field(s) {missing}, but only {list(self._financial_frame.columns)} were preloaded. "
        f"Please include {missing} in your Strategy or Selector fin_fields declaration."
      )

    current_date = self._panel.timeline()[-1] if self._panel.bars > 0 else 0
    target_set = set(objs)
    df = self._financial_frame.filter(
      (pl.col("ts") <= current_date) & (pl.col("obj").is_in(target_set))
    )

    if only_year:
      df = df.filter(pl.col("report_ts").dt.month() == 12)

    if len(df) == 0:
      return pl.DataFrame()

    available_fields = [f for f in fields if f in df.columns]
    keep_cols = ["obj", "ts", "report_ts"] + available_fields
    df = df.select([c for c in keep_cols if c in df.columns])

    df = (
      df.sort(["obj", "report_ts"])
      .group_by("obj", maintain_order=True)
      .tail(lastN)
    )
    return df

  def load_kline(
    self,
    objs: list[str],
    lastN: int | list[int] = 100,
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
    baseline: str | None = None,
  ) -> Panel:
    if fin_fields:
      missing = [f for f in fin_fields if f not in self._panel.columns]
      if missing:
        raise ValueError(
          f"In-backtest dynamic selection requested financial field(s) {missing}, but they were not preloaded in backtest data. "
          f"Please declare fin_fields on your Strategy or Selector class (e.g. fin_fields={list(fin_fields)}) or pass via CLI."
        )

    if len(fin_fields) > 0 and fin_only_year and not self._fin_only_year:
      logger.warning(
        "_HistoricalPanelSelectorDataProvider: load_snapshot/load_kline requested fin_only_year=True with fin_fields, "
        "but backtest data was not loaded with fin_only_year=True (contains quarterly as-of joins). "
        "To align data, set fin_only_year=True on your Strategy or via --fin-only-year."
      )
    n_bars = lastN if isinstance(lastN, int) else int(lastN[0])
    n_bars = min(n_bars, self._panel.bars)
    start_bar = self._panel.bars - n_bars
    return self._panel.slice(start_bar, self._panel.bars, objs=objs)

  def load_snapshot(
    self,
    objs: list[str],
    fin_fields: list[str] = [],
    fin_only_year: bool = True,
  ) -> pl.DataFrame:
    p = self.load_kline(
      objs,
      lastN=1,
      fin_fields=fin_fields,
      fin_only_year=fin_only_year,
    )
    cols = {col: p.last(col) for col in p.columns}
    if "raw_close" in cols:
      cols["close"] = cols["raw_close"]
    return p.cross_section(**cols)


class PortfolioContext(Context):
  """Strategy context for shared-cash portfolio backtesting."""

  def __init__(self, account: PortfolioAccount):
    self.account = account
    self.dp = account.dp
    self._obj_to_group = account._obj_to_group
    self.universe: set[str] | None = None
    self._full_panel: Panel | None = None
    self._in_execute = False
    n = len(self.dp.dates)
    self.buy_percents = np.zeros(n, dtype=np.float64)
    self.sell_percents = np.zeros(n, dtype=np.float64)
    self.target_weights = np.full((self.dp.bars, self.dp.groups), np.nan, dtype=np.float64)
    self.target_tolerances = np.full(self.dp.bars, np.nan, dtype=np.float64)
    self._group_offsets = np.arange(self.dp.groups, dtype=np.int64) * self.dp.bars

  def set_universe(self, objs: list[str] | set[str] | None):
    """Set the active universe whitelist for buying."""
    if objs is None:
      self.universe = None
    else:
      self.universe = set(objs)
      missing = [s for s in objs if s not in self._obj_to_group]
      if missing:
        logger.warning(
          f"set_universe: {len(missing)} obj(s) not in pre-loaded pool and will be ignored: "
          f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

  @property
  def universe_objs(self) -> list[str]:
    """List of objs in the active universe whitelist, or all objs if universe is not set."""
    if self.universe is not None:
      return [s for s in self.dp.objs if s in self.universe]
    return list(self.dp.objs)

  def slice_panel(self, lastN: int = 100) -> Panel:
    """Return a Panel sliced up to current bar (inclusive, no lookahead)."""
    if self._full_panel is None:
      self._full_panel = self.dp.to_panel()
    end_bar = self.dp.i + 1
    n_bars = min(end_bar, max(1, int(lastN)))
    start_bar = end_bar - n_bars
    return self._full_panel.slice(start_bar, end_bar)

  def run_selector(
    self,
    selector: Selector | Callable[[Panel], list[str]],
    lastN: int = 100,
  ) -> list[str]:
    """Execute a Selector on historical data up to current bar without lookahead."""
    sub_panel = self.slice_panel(lastN=lastN)
    with sub_panel.ctx():
      if isinstance(selector, Selector):
        fin_only_year = getattr(self.dp, "fin_only_year", False)
        financial_frame = getattr(self.dp, "financial_frame", None)
        dp_slice = _HistoricalPanelSelectorDataProvider(
          sub_panel, fin_only_year=fin_only_year, financial_frame=financial_frame
        )
        selected = selector.execute(dp_slice, list(sub_panel.objs))
      else:
        selected = selector(sub_panel)
    selected_list = list(selected)
    missing = [s for s in selected_list if s not in self._obj_to_group]
    if missing:
      logger.warning(
        f"run_selector: {len(missing)} selected obj(s) not in pre-loaded pool: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}. "
        "In-backtest dynamic selection operates only on the pre-loaded pool."
      )
    return selected_list

  def objs(self) -> list[str]:
    return self.dp.objs

  def index_of(self, obj: str) -> int:
    """Return group index of obj. Raises ValueError if not found."""
    if obj in self._obj_to_group:
      return self._obj_to_group[obj]
    raise ValueError(f"'{obj}' is not in context objs")

  @property
  def bar(self) -> int:
    return self.dp.i

  @property
  def date(self) -> str:
    """Current bar date as YYYY-MM-DD string."""
    if self.dp.bars > 0 and 0 <= self.dp.i < self.dp.bars:
      return str(self.dp.dates[self.dp.i])[:10]
    return ""

  @property
  def current_date(self) -> str:
    return self.date

  @property
  def step(self) -> int:
    return self.dp.i

  @property
  def warmup_bars(self) -> int:
    return int(getattr(self.dp, "warmup_bars", 0))

  @property
  def is_warmup(self) -> bool:
    return self.dp.i < self.warmup_bars

  @property
  def bars(self) -> int:
    return self.dp.bars

  @property
  def n_objs(self) -> int:
    return self.dp.groups

  def panel(self, allow_future: bool = False, sliced: bool | None = None) -> Panel:
    """Return a Panel view of the data provider.

    During strategy execute(), defaults to returning a strictly sliced Panel
    up to the current bar (`sliced=True`) to prevent lookahead bias.
    To explicitly obtain the full-range panel with future bars, pass `allow_future=True`.

    Args:
      allow_future: If True, allows returning full-range panel during execute().
      sliced: If True, returns sliced panel. If None, defaults to True during execute().
    """
    if sliced is True or (sliced is None and self._in_execute and not allow_future):
      return self.slice_panel(lastN=self.dp.i + 1)
    if self._in_execute and allow_future:
      logger.warning(
        "Calling ctx.panel(allow_future=True) during execute() returns the full-range panel with future bars. "
        "Lookahead risk present."
      )
    return self.dp.to_panel()

  def data(self, name: str, /, obj: str | None = None) -> np.ndarray:
    if name in self.dp.data:
      return self.dp.all(name, obj=obj)
    raise ValueError(f"data '{name}' not found")

  def position(self, obj: str | int) -> Position:
    """Get the current Position object for an obj or group index."""
    return self.account.position(obj, current_bar=self.dp.i)

  def holding_ror(self, obj: str | int) -> float:
    """Return current holding rate of return for obj/group as a float (e.g. -0.05 for -5%)."""
    return self.account.holding_ror(obj)

  def holding_pnl(self, obj: str | int) -> float:
    """Return current holding profit/loss amount for obj/group."""
    return self.account.holding_pnl(obj)

  def cost_price(self, obj: str | int, adjusted: bool = False) -> float:
    """Return acquisition cost price.

    Args:
      obj: Symbol or group index.
      adjusted: If True, returns forward-adjusted equivalent cost (adj_cost_price),
        safe to compare with `ctx.close`. If False, returns raw execution cost price.
    """
    return self.account.cost_price(obj, adjusted=adjusted)

  def holding_days(self, obj: str | int) -> int:
    """Return number of bars since position was first acquired."""
    return self.account.holding_days(obj, current_bar=self.dp.i)

  def _resolve_bar_flags(self, flags) -> tuple[np.ndarray, np.ndarray]:
    idx = self._group_offsets + self.dp.i
    if isinstance(flags, str):
      mask = np.zeros(self.dp.groups, dtype=bool)
      g = self._obj_to_group.get(flags)
      if g is not None:
        mask[g] = True
      return idx, mask
    if isinstance(flags, (int, np.integer)):
      mask = np.zeros(self.dp.groups, dtype=bool)
      if 0 <= flags < self.dp.groups:
        mask[flags] = True
      return idx, mask
    if isinstance(flags, (list, set, tuple)):
      mask = np.zeros(self.dp.groups, dtype=bool)
      for item in flags:
        if isinstance(item, str):
          g = self._obj_to_group.get(item)
          if g is not None:
            mask[g] = True
        elif isinstance(item, (int, np.integer)):
          if 0 <= item < self.dp.groups:
            mask[item] = True
      return idx, mask
    arr = np.asarray(flags, dtype=bool)
    if arr.ndim == 1 and len(arr) == self.dp.groups:
      return idx, arr
    return idx, arr[idx]

  def buy(self, flags, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    # Apply universe whitelist guard for buying
    if self.universe is not None:
      u_mask = np.array(
        [self.dp.objs[g] in self.universe for g in range(self.dp.groups)],
        dtype=bool,
      )
      mask = mask & u_mask
    active = idx[mask]
    if len(active) > 0:
      self.buy_percents[active] = percent

  def sell(self, flags, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    # Sell does NOT filter by universe (supports Runoff natural liquidation)
    active = idx[mask]
    if len(active) > 0:
      self.sell_percents[active] = percent

  def rebalance(self, weights: np.ndarray, tolerance: float | None = None):
    """Set target portfolio weights for the current bar (shape: (groups,) or (groups * bars,))."""
    if tolerance is not None:
      self.target_tolerances[self.dp.i] = float(tolerance)
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
    else:
      raise ValueError(
        f"weights must have shape ({self.dp.groups},) or ({self.dp.groups * self.dp.bars},)"
      )

    # Apply universe whitelist guard: stocks outside universe cannot be bought or increased
    if self.universe is not None:
      total_equity = self.account.asset
      for g in range(ts_start, self.dp.groups):
        if self.dp.objs[g] not in self.universe:
          curr_shares = self.account.positions[g].shares
          if curr_shares <= 0 or total_equity <= 0:
            w[g] = 0.0
          else:
            dp_g = self.account.positions[g].price
            curr_val = curr_shares * dp_g
            curr_w = curr_val / total_equity
            w[g] = min(w[g], curr_w)

    self.target_weights[self.dp.i] = w

  def rebalance_objs(self, target_weights: dict[str, float], tolerance: float | None = None):
    """Rebalance portfolio using a mapping of {symbol: weight}.

    Symbols not present in target_weights are assigned a target weight of 0.0.
    Symbols not in the pre-loaded pool are logged as warnings and ignored.

    Args:
      target_weights: dict mapping stock symbol to target weight (e.g. {"SH600000": 0.05}).
      tolerance: optional tolerance band (e.g. 0.01 for 1%).
    """
    weights = np.zeros(self.dp.groups, dtype=np.float64)
    missing = []
    for sym, weight in target_weights.items():
      g = self._obj_to_group.get(sym)
      if g is not None:
        weights[g] = max(0.0, float(weight))
      else:
        missing.append(sym)
    if missing:
      logger.warning(
        f"rebalance_objs: {len(missing)} obj(s) not in pre-loaded pool and ignored: "
        f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
      )
    self.rebalance(weights, tolerance=tolerance)


class PortfolioRunner:
  """Runner for multi-stock shared-cash portfolio strategies."""

  def __init__(
    self,
    account: PortfolioAccount,
    /,
    exec_mode: str = "close",
    check_limit: bool = True,
    tolerance: float = 0.0,
  ):
    if exec_mode not in ("close", "next_open"):
      raise ValueError(f"unsupported exec_mode: {exec_mode}, expected 'close' or 'next_open'")
    if tolerance > 0.0:
      account.tolerance = float(tolerance)
    account.exec_mode = exec_mode
    self.ctx = PortfolioContext(account)
    self.exec_mode = exec_mode
    self.check_limit = check_limit
    self.diagnostics: dict | None = None

  def run(self, strategy: Strategy):
    dp = self.ctx.dp
    acc = self.ctx.account
    if dp.groups == 0 or dp.bars == 0:
      return

    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      strategy.prepare(self.ctx)
    dp.reset()

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

    bar_ctx = al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
    self.ctx._in_execute = True
    try:
      with bar_ctx:
        for i in dp:
          bar_ctx.end = i + 1
          bar_ctx.apply()
          strategy.execute(self.ctx)
          acc.apply_corporate_actions(i)

          sig_i = i - 1 if self.exec_mode == "next_open" else i
          idx_now = offsets + i
          mark_p = prices[idx_now]
          deal_p = deal_prices_all[idx_now]
          lim_now = limited_all[idx_now] if limited_all is not None else None

          warmup_b = getattr(dp, "warmup_bars", 0)
          if sig_i < 0 or i < warmup_b:
            acc.update_mark_to_market(i, mark_p)
            continue

          # 1. Execute explicit sell signals (e.g. stop-loss / risk controls) first
          idx_sig = offsets + sig_i
          sell_p = self.ctx.sell_percents[idx_sig]
          buy_p = self.ctx.buy_percents[idx_sig]

          has_explicit_sells = False
          for g in range(dp.groups):
            if not acc.tradable_mask[g]:
              continue
            if sell_p[g] > 0:
              if lim_now is not None and lim_now[g] == -1:
                continue
              shares_to_sell = acc.positions[g].shares * sell_p[g]
              if sell_p[g] < 1.0:
                shares_to_sell = float(int(shares_to_sell) - (int(shares_to_sell) % acc.unit))
              if shares_to_sell > 0:
                acc._sell_group(g, i, shares_to_sell, deal_p[g])
                has_explicit_sells = True

          # 2. Check if explicit target_weights were provided at sig_i
          tgt_w = self.ctx.target_weights[sig_i]
          if not np.all(np.isnan(tgt_w)):
            step_tol = self.ctx.target_tolerances[sig_i]
            tol_val = None if np.isnan(step_tol) else float(step_tol)

            # If explicit sells occurred on group g, reduce/zero target weight accordingly
            if has_explicit_sells:
              tgt_w = tgt_w.copy()
              for g in range(dp.groups):
                if sell_p[g] > 0:
                  tgt_w[g] = max(0.0, tgt_w[g] * (1.0 - sell_p[g]))

            acc.rebalance_step(
              i, tgt_w, deal_p, mark_p, lim_now, tolerance=tol_val, universe=self.ctx.universe
            )
            continue

          # 3. Otherwise process buy signals sharing available cash across active signals
          buy_groups = [
            g
            for g in range(dp.groups)
            if acc.tradable_mask[g] and buy_p[g] > 0 and (lim_now is None or lim_now[g] != 1)
          ]
          if len(buy_groups) > 0 and acc.cash > 0:
            total_req = sum(buy_p[g] for g in buy_groups)
            alloc_cash = acc.cash * min(1.0, total_req)
            for g in buy_groups:
              budget = alloc_cash * (buy_p[g] / total_req)
              acc._buy_group(g, i, budget, deal_p[g])

          acc.update_mark_to_market(i, mark_p)
    finally:
      self.ctx._in_execute = False

    # Post-execution phase: finalize hook with full bar range
    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      diag = strategy.finalize(self.ctx)
      if diag is not None:
        self.diagnostics = diag
        setattr(strategy, "diagnostics", diag)
        setattr(acc, "diagnostics", diag)


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
    return self._full_panel.slice(0, end_bar)

  def execute(self, ctx: PortfolioContext):  # type: ignore[override]
    i = ctx.bar
    if i < self.min_bars or (i - self.min_bars) % self.rebalance_period != 0:
      return

    sub_panel = self._slice_panel(i + 1)
    with sub_panel.ctx():
      if isinstance(self.selector, Selector):
        fin_only_year = getattr(ctx.dp, "fin_only_year", False)
        financial_frame = getattr(ctx.dp, "financial_frame", None)
        dp_slice = _HistoricalPanelSelectorDataProvider(
          sub_panel, fin_only_year=fin_only_year, financial_frame=financial_frame
        )
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
