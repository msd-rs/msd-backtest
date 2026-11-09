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

  @property
  def bar(self) -> int:
    return self.dp.i

  @property
  def step(self) -> int:
    return self.dp.i

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
    import polars as pl
    new_cols = {}
    for col in self._full_panel.frame.columns:
      arr_2d = self._full_panel.column(col)[:, start_bar:end_bar]
      new_cols[col] = arr_2d.ravel()
    from mbt.select.panel import Panel
    return Panel(pl.DataFrame(new_cols), list(self._full_panel.objs), n_bars)

  def panel(self, allow_future: bool = False, sliced: bool = False):
    if sliced:
      return self.slice_panel(lastN=self.dp.i + 1)
    if self._in_execute and not allow_future:
      logger.warning(
        "Calling ctx.panel() during execute() returns the full-range panel with future bars. "
        "To avoid lookahead bias, use ctx.slice_panel(...) or ctx.panel(sliced=True), "
        "or pass allow_future=True."
      )
    return self.dp.to_panel()

  def data(self, name: str, /, obj: str | None = None) -> np.ndarray:
    # user may use it like ctx.open, ctx.high, ctx.low, ctx.close, ctx.volume
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
    self.ctx = RunnerContext(account)
    self.exec_mode = exec_mode
    self.check_limit = check_limit

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
          if sig_offset < 0:
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
