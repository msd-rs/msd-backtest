from mbt.account import Operation, ACTION_BUY_PERCENT, ACTION_SELL_PERCENT
from mbt.strategy import Context
from mbt.account import Account
from mbt.strategy import Strategy
import numpy as np


class RunnerContext(Context):
  """RunnerContext is a context for a strategy

  It is used to pass data and operations to a strategy.

  Data required internally are:
  - ts: timestamp
  - price: price of the stock
  Data optional internally are:
    - "bonus": bonus per share in dividend
    - "transfers": transfers per share in dividend
    - "dividend": dividend per share in dividend
    - "rightShare": right share per share in dividend
    - "rightPrice": right price per share in dividend
  Other data can be added by user for specific strategy

  The strategy can use the following methods to operate the account:
  - buy: buy a position
  - sell: sell a position
  and refer data by `ctx.data_name`
  """

  def __init__(self, account: Account):
    self.accounts = [account.clone(g) for g in range(account.dp.groups)]
    self.dp = account.dp
    self.actions = np.zeros(len(self.dp.dates), dtype=np.int32)
    self.action_percents = np.zeros(len(self.dp.dates), dtype=np.float64)

  def data(self, name: str, /, symbol: str | None = None) -> np.ndarray:
    # user may use it like ctx.open, ctx.high, ctx.low, ctx.close, ctx.volume
    if name in self.dp.data:
      return self.dp.all(name, symbol=symbol)
    raise ValueError(f"data '{name}' not found")

  def buy(self, flags: np.ndarray, percent: float = 1.0):
    for g in range(self.dp.groups):
      j = g * self.dp.bars + self.dp.i
      if flags[j]:
        self.actions[j] = ACTION_BUY_PERCENT
        self.action_percents[j] = percent

  def sell(self, flags: np.ndarray, percent: float = 1.0):
    for g in range(self.dp.groups):
      j = g * self.dp.bars + self.dp.i
      if flags[j]:
        self.actions[j] = ACTION_SELL_PERCENT
        self.action_percents[j] = percent


class Runner:
  """Runner is a runner of a strategy"""

  def __init__(self, account: Account):
    self.ctx = RunnerContext(account)

  def run(self, strategy: Strategy):
    prices = self.ctx.dp.all("price")
    for i in self.ctx.dp:
      strategy.execute(self.ctx)
      for g in range(self.ctx.dp.groups):
        j = g * self.ctx.dp.bars + self.ctx.dp.i
        action = self.ctx.actions[j]
        percent = self.ctx.action_percents[j]
        if action == ACTION_BUY_PERCENT:
          self.ctx.accounts[g].do_operation(
            self.ctx.dp.i,
            Operation.buy_percent(
              percent, prices[j] * (1 + self.ctx.accounts[g].slippage)
            ),
          )
        elif action == ACTION_SELL_PERCENT:
          self.ctx.accounts[g].do_operation(
            self.ctx.dp.i,
            Operation.sell_percent(
              percent, prices[j] * (1 - self.ctx.accounts[g].slippage)
            ),
          )
        else:
          self.ctx.accounts[g].do_operation(self.ctx.dp.i, Operation.keep(prices[j]))
