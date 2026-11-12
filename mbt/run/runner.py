from typing import Any
from mbt.account import Operation, ACTION_BUY_PERCENT, ACTION_SELL_PERCENT
from mbt.strategy import Context
from mbt.account import Account
from mbt.strategy import Strategy
import numpy as np
import alpha as al


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
      if hasattr(self.dp, "objs") and flags in self.dp.objs:
        mask[self.dp.objs.index(flags)] = True
      return idx, mask
    if isinstance(flags, (int, np.integer)):
      mask = np.zeros(self.dp.groups, dtype=bool)
      if 0 <= flags < self.dp.groups:
        mask[flags] = True
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
      if obj in self.dp.objs:
        return self.accounts[self.dp.objs.index(obj)]
      return None
    elif isinstance(obj, (int, np.integer)):
      if 0 <= obj < len(self.accounts):
        return self.accounts[int(obj)]
    return None

  def position(self, obj: str | int | None = None) -> 'Position':
    """Get the current Position object."""
    acc = self._get_account(obj)
    return acc.position if acc is not None else Position.zero()

  def holding_ror(self, obj: str | int | None = None) -> float:
    """Return holding rate of return as a decimal (e.g. -0.05 for -5%)."""
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
    acc = self._get_account(obj)
    return acc.holding_days if acc is not None else 0


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

  def run(self, strategy: Strategy):
    dp = self.ctx.dp
    if dp.groups == 0 or dp.bars == 0:
      return

    # 1. Pre-calculation phase: full bar range so static indicators run in O(N * T)
    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      strategy.prepare(self.ctx)
    dp.reset()

    prices = dp.all("price")
    if self.exec_mode == "next_open":
      if "raw_open" in dp.data:
        deal_prices = dp.all("raw_open")
      elif "open" in dp.data:
        deal_prices = dp.all("open")
      else:
        deal_prices = prices
    else:
      deal_prices = prices

    limited = dp.all("limited") if (self.check_limit and "limited" in dp.data) else None
    bars = dp.bars
    groups = dp.groups
    accounts = self.ctx.accounts
    buy_percents = self.ctx.buy_percents
    sell_percents = self.ctx.sell_percents

    bar_ctx = al.Context(groups=groups, flags=al.FLAG_SKIP_NAN, start=0, end=1)
    self.ctx._in_execute = True
    try:
      with bar_ctx:
        for i in dp:
          bar_ctx.end = i + 1
          bar_ctx.apply()
          strategy.execute(self.ctx)
          sig_offset = i - 1 if self.exec_mode == "next_open" else i
          warmup_b = getattr(dp, "warmup_bars", 0)
          if sig_offset < 0 or i < warmup_b:
            for g in range(groups):
              j = g * bars + i
              accounts[g].do_operation(i, Operation.keep(prices[j]))
            continue

          for g in range(groups):
            j = g * bars + i
            sig_j = g * bars + sig_offset
            sell_p = sell_percents[sig_j]
            buy_p = buy_percents[sig_j]
            acc = accounts[g]
            lim = limited[j] if limited is not None else 0

            can_sell = sell_p > 0 and lim != -1
            can_buy = buy_p > 0 and lim != 1

            if can_sell and can_buy:
              # Execute sell first to release cash, then buy in the same bar
              acc.do_operation(
                i,
                Operation.sell_percent(sell_p, deal_prices[j] * (1 - acc.slippage)),
              )
              acc.do_operation(
                i,
                Operation.buy_percent(buy_p, deal_prices[j] * (1 + acc.slippage)),
              )
            elif can_sell:
              acc.do_operation(
                i,
                Operation.sell_percent(sell_p, deal_prices[j] * (1 - acc.slippage)),
              )
            elif can_buy:
              acc.do_operation(
                i,
                Operation.buy_percent(buy_p, deal_prices[j] * (1 + acc.slippage)),
              )
            else:
              acc.do_operation(i, Operation.keep(prices[j]))

            # Ensure mark-to-market uses closing price when executed at open
            if self.exec_mode == "next_open" and acc.position.shares > 0 and not np.isnan(prices[j]) and prices[j] > 0:
              acc.position.price = prices[j]
              acc.assets[i] = acc.asset
              acc.ror[i] = acc.asset / acc.assets[0] - 1
              acc.ror_hold[i] = (acc.asset - acc.per_assets) / acc.per_assets if acc.per_assets > 0 else 0.0
    finally:
      self.ctx._in_execute = False

    # 3. Post-execution phase: finalize hook with full bar range
    with al.Context(groups=dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=dp.bars):
      diag = strategy.finalize(self.ctx)
      if diag is not None:
        self.diagnostics = diag
        setattr(strategy, "diagnostics", diag)
