# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from typing import Any
import logging
from mbt.account import (
  ACTION_BUY,
  ACTION_BUY_NO_CASH,
  ACTION_BUY_PERCENT,
  ACTION_KEEP,
  ACTION_SELL,
  ACTION_SELL_NO_HOLD,
  ACTION_SELL_PERCENT,
  Account,
  Operation,
  Position,
  Record,
)
from mbt.strategy import Context, Strategy
import numpy as np
import alpha as al
from alpha.sim import AccountBatch, MarketData, SimConfig, Simulator

logger = logging.getLogger(__name__)


class RunnerContext(Context):
  """RunnerContext is a context for a strategy

  It is used to pass data and operations to a strategy.

  Data required internally are:
  - ts: timestamp
  - price: price of the stock
  Data optional internally are:
    - "dividend": dividend per share in dividend
    - "transfer_shares": transfer shares per share in dividend
    - "right_shares": right share per share in dividend
    - "right_price": right price per share in dividend
  Other data can be added by user for specific strategy

  The strategy can use the following methods to operate the account:
  - buy: buy a position
  - sell: sell a position
  and refer data by `ctx.data_name`
  """

  def __init__(self, account: Account):
    self.accounts = [account.clone(g) for g in range(account.dp.groups)]
    self.dp = account.dp
    self._in_execute = False
    self._full_panel = None
    self._obj_to_group: dict[str, int] = (
      {s: i for i, s in enumerate(self.dp.objs)} if hasattr(self.dp, "objs") else {}
    )
    self._batch: AccountBatch | None = None
    self._batch_rors_bar: int = -1
    self._batch_rors: np.ndarray | None = None
    self._batch_shares_bar: int = -1
    self._batch_shares: np.ndarray | None = None
    self._batch_holding_bars_bar: int = -1
    self._batch_holding_bars: np.ndarray | None = None
    n = len(self.dp.dates)
    self.actions = np.zeros(n, dtype=np.int32)
    self.action_percents = np.zeros(n, dtype=np.float64)
    self.buy_percents = np.zeros(n, dtype=np.float64)
    self.sell_percents = np.zeros(n, dtype=np.float64)
    self._group_offsets = np.arange(self.dp.groups, dtype=np.int64) * self.dp.bars

  def objs(self) -> list[str]:
    return self.dp.objs

  def index_of(self, obj: str) -> int:
    return self.dp.objs.index(obj)

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

  def slice_panel(self, lastN: int = 100):
    """Return a Panel sliced up to current bar (inclusive, no lookahead)."""
    if self._full_panel is None:
      self._full_panel = self.dp.to_panel()
    end_bar = self.dp.i + 1
    n_bars = min(end_bar, max(1, int(lastN)))
    start_bar = end_bar - n_bars
    return self._full_panel.slice(start_bar, end_bar)

  def panel(self, allow_future: bool = False, sliced: bool | None = None):
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
    # user may use it like ctx.open, ctx.high, ctx.low, ctx.close, ctx.volume
    if name in self.dp.data:
      return self.dp.all(name, obj=obj)
    raise ValueError(f"data '{name}' not found")

  def _resolve_bar_flags(self, flags: Any) -> tuple[np.ndarray, np.ndarray]:
    idx = self._group_offsets + self.dp.i
    if isinstance(flags, (bool, np.bool_)):
      mask = np.full(self.dp.groups, bool(flags), dtype=bool)
      return idx, mask
    if isinstance(flags, str):
      mask = np.zeros(self.dp.groups, dtype=bool)
      g = self._obj_to_group.get(flags)
      if g is not None:
        mask[g] = True
      else:
        raise ValueError(f"Symbol '{flags}' is not in context objs")
      return idx, mask
    if isinstance(flags, (int, np.integer)):
      mask = np.zeros(self.dp.groups, dtype=bool)
      if 0 <= flags < self.dp.groups:
        mask[flags] = True
      else:
        raise ValueError(f"Group index {flags} out of range [0, {self.dp.groups})")
      return idx, mask
    arr = np.asarray(flags, dtype=bool)
    if arr.ndim == 0:
      mask = np.full(self.dp.groups, bool(arr.item()), dtype=bool)
      return idx, mask
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
      self.actions[active] = ACTION_BUY_PERCENT
      self.action_percents[active] = percent

  def sell(self, flags: np.ndarray, percent: float = 1.0):
    if self.dp.groups == 0:
      return
    idx, mask = self._resolve_bar_flags(flags)
    active = idx[mask]
    if len(active) > 0:
      self.sell_percents[active] = percent
      self.actions[active] = ACTION_SELL_PERCENT
      self.action_percents[active] = percent

  def _get_account(self, obj: str | int | None = None) -> Account | None:
    if not self.accounts:
      return None
    if obj is None:
      return self.accounts[0]
    if isinstance(obj, str):
      g = self._obj_to_group.get(obj)
      if g is not None:
        return self.accounts[g]
      raise ValueError(f"Symbol '{obj}' is not in context objs")
    elif isinstance(obj, (int, np.integer)):
      if 0 <= obj < len(self.accounts):
        return self.accounts[int(obj)]
      raise ValueError(f"Group index {obj} out of range [0, {len(self.accounts)})")
    raise TypeError(f"Expected str, int, or None for obj, got {type(obj).__name__}")

  def position(self, obj: str | int | None = None) -> 'Position':
    """Get the current Position object."""
    acc = self._get_account(obj)
    if acc is not None:
      g = acc.group
      if self._batch is not None:
        cash, shares, raw_cost, raw_mark, adj_cost, dividends, h_bars = self._batch.state_of(g)
        acc.cash = cash
        acc._holding_bars = h_bars
        acc.position = Position(
          shares,
          raw_cost,
          raw_cost_price=raw_cost,
          raw_mark_price=raw_mark,
          dividends=dividends,
          adj_cost_price=adj_cost,
          holding_days=h_bars,
        )
      else:
        acc.position.holding_days = acc.holding_days
      i = self.dp.i
      if hasattr(self.dp, "data") and "price" in self.dp.data:
        p = self.dp.all("price")[g * self.dp.bars + i]
        if not np.isnan(p) and p > 0:
          acc.position.raw_mark_price = float(p)
      return acc.position
    return Position.zero()

  def holding_ror(self, obj: str | int | None = None) -> float:
    """Return holding rate of return as a decimal (e.g. -0.05 for -5%)."""
    if self._batch is not None:
      acc = self._get_account(obj)
      if acc is None:
        return 0.0
      cur_bar = self.dp.i
      if self._batch_rors_bar != cur_bar:
        self._batch_rors = self._batch.holding_rors(cur_bar)
        self._batch_rors_bar = cur_bar
      return float(self._batch_rors[acc.group])  # type: ignore[index]
    return self.position(obj).holding_ror

  def holding_pnl(self, obj: str | int | None = None) -> float:
    """Return holding profit/loss amount."""
    return self.position(obj).holding_pnl

  def cost_price(self, obj: str | int | None = None, adjusted: bool = False) -> float:
    """Return acquisition cost price.

    Args:
      obj: Symbol or group index (optional).
      adjusted: If True, returns forward-adjusted equivalent cost (adj_cost_price).
        If False, returns raw execution cost price.
    """
    pos = self.position(obj)
    return pos.adj_cost_price if adjusted else pos.raw_cost_price

  def holding_days(self, obj: str | int | None = None) -> int:
    """Return number of bars current position has been held."""
    if self._batch is not None:
      acc = self._get_account(obj)
      if acc is None:
        return 0
      cur_bar = self.dp.i
      if self._batch_holding_bars_bar != cur_bar:
        self._batch_holding_bars = self._batch.holding_bars()
        self._batch_holding_bars_bar = cur_bar
      return int(self._batch_holding_bars[acc.group])  # type: ignore[index]
    acc = self._get_account(obj)
    return acc.holding_days if acc is not None else 0

  def holding_shares(self) -> np.ndarray:
    """Return 1D float64 array of currently held shares across all objs (shape: `(n_objs,)`)."""
    if self._batch is not None:
      cur_bar = self.dp.i
      if self._batch_shares_bar != cur_bar or self._batch_shares is None:
        self._batch_shares = self._batch.shares()
        self._batch_shares_bar = cur_bar
      return self._batch_shares
    return np.array([acc.position.shares for acc in self.accounts], dtype=np.float64)

  def holding_rors(self) -> np.ndarray:
    """Return 1D float64 array of corporate-action-aware holding ROR across all objs (shape: `(n_objs,)`)."""
    if self._batch is not None:
      cur_bar = self.dp.i
      if self._batch_rors_bar != cur_bar or self._batch_rors is None:
        self._batch_rors = self._batch.holding_rors(cur_bar)
        self._batch_rors_bar = cur_bar
      return self._batch_rors
    return np.array([self.holding_ror(g) for g in range(len(self.accounts))], dtype=np.float64)


class Runner:
  """Runner is a runner of a strategy"""

  def __init__(
    self,
    account: Account,
    /,
    exec_mode: str = "close",
    check_limit: bool = True,
  ):
    if exec_mode not in ("close", "next_open"):
      raise ValueError(f"unsupported exec_mode: {exec_mode}, expected 'close' or 'next_open'")
    account.exec_mode = exec_mode
    self.ctx = RunnerContext(account)
    self.exec_mode = exec_mode
    self.check_limit = check_limit
    self.diagnostics: dict | None = None

  @staticmethod
  def _hydrate_accounts(
    accounts: list[Account],
    snapshot: dict[str, np.ndarray],
    trades: dict[str, np.ndarray],
  ) -> None:
    cash_arr = snapshot["cash"].tolist()
    shares_arr = snapshot["shares"].tolist()
    raw_cost_arr = snapshot["raw_cost_price"].tolist()
    raw_mark_arr = snapshot["raw_mark_price"].tolist()
    adj_cost_arr = snapshot["adj_cost_price"].tolist()
    div_arr = snapshot["dividends"].tolist()
    per_assets_arr = snapshot["per_assets"].tolist()
    deducted_arr = snapshot["deducted"].tolist()
    holding_bars_arr = snapshot["holding_bars"].tolist()

    for g, acc in enumerate(accounts):
      acc.records.clear()
      acc.cash = cash_arr[g]
      acc.deducted = deducted_arr[g]
      acc.per_assets = per_assets_arr[g]
      acc._holding_bars = holding_bars_arr[g]
      acc.position = Position(
        shares_arr[g],
        raw_cost_arr[g],
        raw_cost_price=raw_cost_arr[g],
        raw_mark_price=raw_mark_arr[g],
        dividends=div_arr[g],
        adj_cost_price=adj_cost_arr[g],
        holding_days=holding_bars_arr[g],
      )

    if trades and len(trades.get("bar", ())) > 0:
      t_groups = trades["group"].tolist()
      t_bars = trades["bar"].tolist()
      t_actions = trades["action"].tolist()
      t_shares = trades["shares"].tolist()
      t_prices = trades["deal_price"].tolist()
      t_fees = trades["fee"].tolist()
      t_cash = trades["cash_after"].tolist()
      t_pos_sh = trades["pos_shares"].tolist()
      t_cost = trades["raw_cost_price"].tolist()

      for k in range(len(t_bars)):
        g = t_groups[k]
        acc = accounts[g]
        act = t_actions[k]
        sh = t_shares[k]
        dp_val = t_prices[k]
        if act == ACTION_BUY:
          op = Operation.buy(Position(sh, dp_val))
        elif act == ACTION_SELL:
          op = Operation.sell(Position(sh, dp_val))
        elif act == ACTION_BUY_NO_CASH:
          op = Operation.buy_no_cash(Position(0.0, 0.0))
        elif act == ACTION_SELL_NO_HOLD:
          op = Operation.sell_no_hold(Position(0.0, 0.0))
        else:
          op = Operation(act, Position(sh, dp_val))
        pos_snap = Position(t_pos_sh[k], t_cost[k])
        acc.records.append(Record(t_bars[k], op, t_cash[k], 0.0, pos_snap, t_fees[k]))

  def run(self, strategy: Strategy):
    dp = self.ctx.dp
    if dp.groups == 0 or dp.bars == 0:
      return

    # 1. Pre-calculation phase: full bar range so static indicators run in O(N * T)
    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      strategy.prepare(self.ctx)
      sigs = strategy.signals(self.ctx)
    dp.reset()

    accounts = self.ctx.accounts
    base_acc = accounts[0]
    warmup_b = int(getattr(dp, "warmup_bars", 0))

    market = MarketData.from_dict(
      dp.data,
      groups=dp.groups,
      exec_mode=self.exec_mode,
      check_limit=self.check_limit,
    )
    cfg = SimConfig(
      initial_cash=float(base_acc.initial_cash),
      unit=int(base_acc.unit),
      slippage=float(base_acc.slippage),
      exec_mode=self.exec_mode,
      warmup_bars=warmup_b,
      start_bar=1,
      check_limit=self.check_limit,
      fees=base_acc.fee.config,
    )

    out_assets = dp.all("assets", create_if_not_exist=np.float64)
    out_ror = dp.all("ror", create_if_not_exist=np.float64)
    out_ror_hold = dp.all("ror_hold", create_if_not_exist=np.float64)
    out_actions = dp.all("actions", create_if_not_exist=np.int32)
    out_action_shares = dp.all("action_shares", create_if_not_exist=np.float64)

    if sigs is not None:
      # Fast path: full vectorized signals -> multi-threaded Rust simulation
      buy_sig, sell_sig = sigs
      res = Simulator(market, cfg).run_into(
        buy_sig,
        sell_sig,
        out_assets=out_assets,
        out_ror=out_ror,
        out_ror_hold=out_ror_hold,
        out_actions=out_actions,
        out_action_shares=out_action_shares,
      )
      self._hydrate_accounts(accounts, res.final_state, res.trades)
    else:
      # General bar-by-bar execute() path backed by Rust AccountBatch SoA state engine
      bars = dp.bars
      groups = dp.groups
      buy_percents = self.ctx.buy_percents
      sell_percents = self.ctx.sell_percents

      batch = AccountBatch(market, cfg)
      self.ctx._batch = batch
      bar_ctx = al.Context(groups=groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
      self.ctx._in_execute = True
      try:
        with bar_ctx:
          for i in dp:
            bar_ctx.end = i + 1
            bar_ctx.apply()
            batch.apply_corporate_actions_bar(i)
            strategy.execute(self.ctx)
            batch.step_bar(
              i,
              buy_percents,
              sell_percents,
              out_assets=out_assets,
              out_ror=out_ror,
              out_ror_hold=out_ror_hold,
              out_actions=out_actions,
              out_action_shares=out_action_shares,
            )
      finally:
        self.ctx._in_execute = False
        self.ctx._batch = None
        self.ctx._batch_rors_bar = -1
        self.ctx._batch_rors = None
        self.ctx._batch_shares_bar = -1
        self.ctx._batch_shares = None
        self.ctx._batch_holding_bars_bar = -1
        self.ctx._batch_holding_bars = None

      self._hydrate_accounts(accounts, batch.snapshot(), batch.trades())

    buy_cnt = int(np.count_nonzero(out_actions == ACTION_BUY))
    no_cash_cnt = int(np.count_nonzero(out_actions == ACTION_BUY_NO_CASH))
    if no_cash_cnt > 0 and (buy_cnt == 0 or no_cash_cnt > buy_cnt):
      logger.warning(
        "[BUY_NO_CASH GUARD] %d buy signals rejected due to insufficient cash for 1 lot "
        "(unit=%d, initial_cash=%.2f) vs %d filled buys. Increase -c/--cash or position sizing.",
        no_cash_cnt,
        base_acc.unit,
        base_acc.initial_cash,
        buy_cnt,
      )

    # 3. Post-execution phase: finalize hook with full bar range
    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      diag = strategy.finalize(self.ctx)
      if diag is not None:
        self.diagnostics = diag
        setattr(strategy, "diagnostics", diag)
