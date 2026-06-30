from pathlib import Path
import json
from dataclasses import dataclass
from .data_provider import DataProvider
import numpy as np
import logging


ACTION_KEEP = 0

ACTION_BUY = 1
ACTION_BUY_PERCENT = 3
ACTION_BUY_NO_CASH = 5

ACTION_SELL = 2
ACTION_SELL_PERCENT = 4
ACTION_SELL_NO_HOLD = 6

logger = logging.getLogger(__name__)


class Position:
  """Position is a position of a stock"""

  def __init__(self, shares: float, price: float):
    self.shares = shares
    self.price = price

  @property
  def value(self) -> float:
    return self.shares * self.price

  @staticmethod
  def zero() -> "Position":
    return Position(0, 0)

  @staticmethod
  def from_cash(cash: float, price: float, unit: int = 100) -> "Position":
    if price == 0:
      return Position.zero()
    shares = int(cash / price)
    shares = shares - shares % unit  # round down to unit
    return Position(float(shares), price)

  def __add__(self, other: "Position") -> "Position":
    value = self.value + other.value
    shares = self.shares + other.shares
    if shares <= 0:
      return Position.zero()
    return Position(shares, value / shares)

  def __sub__(self, other: "Position") -> "Position":
    value = self.value - other.value
    shares = self.shares - other.shares
    if shares <= 0:
      return Position.zero()
    return Position(shares, value / shares)

  def __str__(self):
    return f"Position(shares={self.shares}, price={self.price})"


class Operation:
  """Operation is a operation in a day of a backtest account"""

  def __init__(self, action: int, position: Position | None):
    self.action = action
    self.position = position

  @staticmethod
  def keep(price: float) -> "Operation":
    return Operation(ACTION_KEEP, Position(0, price))

  @staticmethod
  def buy(position: Position) -> "Operation":
    return Operation(ACTION_BUY, position)

  @staticmethod
  def buy_percent(percent: float, price: float) -> "Operation":
    return Operation(ACTION_BUY_PERCENT, Position(percent, price))

  @staticmethod
  def sell(position: Position) -> "Operation":
    return Operation(ACTION_SELL, position)

  @staticmethod
  def sell_percent(percent: float, price: float) -> "Operation":
    return Operation(ACTION_SELL_PERCENT, Position(percent, price))

  @staticmethod
  def buy_no_cash(position: Position) -> "Operation":
    return Operation(ACTION_BUY_NO_CASH, position)

  @staticmethod
  def sell_no_hold(position: Position) -> "Operation":
    return Operation(ACTION_SELL_NO_HOLD, position)

  def __str__(self):
    kind = "None"
    if self.action == ACTION_KEEP:
      kind = "Keep"
    elif self.action == ACTION_BUY:
      kind = "Buy"
    elif self.action == ACTION_BUY_PERCENT:
      kind = "BuyPercent"
    elif self.action == ACTION_SELL:
      kind = "Sell"
    elif self.action == ACTION_SELL_PERCENT:
      kind = "SellPercent"
    elif self.action == ACTION_BUY_NO_CASH:
      kind = "BuyNoCash"
    elif self.action == ACTION_SELL_NO_HOLD:
      kind = "SellNoHold"
    if self.position is None:
      return f"{kind}"
    return f"{kind}({self.position})"


@dataclass
class FeeItem:
  ts: int = 0
  commission_buy: float = 0.0
  min_commission_buy: float = 0.0
  tax_buy: float = 0.0
  transaction_fee_buy: float = 0.0
  commission_sell: float = 0.0
  min_commission_sell: float = 0.0
  tax_sell: float = 0.0
  transaction_fee_sell: float = 0.0

  def buy(self, value: float) -> float:
    if value == 0:
      return 0.0

    commission = value * self.commission_buy
    min_commission = self.min_commission_buy
    tax = value * self.tax_buy
    transaction_fee = self.transaction_fee_buy
    return max(commission, min_commission) + tax + transaction_fee

  def sell(self, value: float) -> float:
    if value == 0:
      return 0.0
    commission = value * self.commission_sell
    min_commission = self.min_commission_sell
    tax = value * self.tax_sell
    transaction_fee = self.transaction_fee_sell
    return max(commission, min_commission) + tax + transaction_fee


class Fee:
  def __init__(self, config: list[FeeItem] | list[str] | str | bytes | None = None):
    def build_fee_item(x):
      return FeeItem(**x)

    if isinstance(config, list):
      if len(config) > 0 and isinstance(config[0], str):
        self.config = [json.loads(x, object_hook=build_fee_item) for x in config]  # type: ignore
        self.config.sort(key=lambda x: x.ts)
      else:
        self.config = config
    elif isinstance(config, str):
      p = Path(config)

      if p.is_file():
        self.config = json.load(p.open("r"), object_hook=build_fee_item)
      else:
        self.config = json.loads(config, object_hook=build_fee_item)
    elif isinstance(config, bytes):
      self.config = json.loads(config.decode("utf-8"), object_hook=build_fee_item)
    else:
      self.config = [FeeItem()]

    # add a default fee item at the beginning when no fee item is provided
    if len(self.config) == 0:
      self.config = [FeeItem()]

  def get(self, ts: int) -> FeeItem:
    """
    find the last fee item whose ts is less than or equal to the given ts
    Args:
      ts: timestamp
    Returns:
      fee item
    """
    for i in range(len(self.config)):
      if self.config[i].ts > ts:
        if i == 0:
          return self.config[0]
        return self.config[i - 1]
    return self.config[-1]


class Record:
  """Record is a record of a backtest account"""

  def __init__(
    self,
    ts: int,
    operation: Operation,
    cash: float,
    hold_profit: float,
    position: Position,
    deducted: float,
  ):
    """
    Args:
      ts: timestamp of the record
      operation: incoming operation of the record
      cash: cash after operation
      hold_profit: profit when position was bought
      position: position after operation
      deducted: deducted commission and other fees
    """

    """the date of the record"""
    self.ts = ts
    """incoming operation of the record"""
    self.operation = operation
    """cash after operation"""
    self.cash = cash
    """profit when position was bought"""
    self.hold_profit = hold_profit
    """position after operation"""
    self.position = position
    """value of position"""
    self.value = self.position.shares * self.position.price
    """total assets"""
    self.assets = self.cash + self.value
    """deducted commission and other fees"""
    self.deducted = deducted


class Account:
  """Account is a backtest account, with initial cash and data provider

  data_provider used to get the actions, price, dividend, commission, etc.
  """

  def __init__(
    self,
    initial_cash: float,
    dp: DataProvider,
    /,
    group: int = 0,
    fee: Fee | None = None,
    unit: int = 100,
    slippage: float = 0.0,
  ):
    """
    Args:
      initial_cash: initial cash of the account
      dp: data provider of the account, the data kinds will be used are:
        - "price": price of the stock, without dividend
        - "transfer_shares": transfers per share in dividend
        - "dividend": dividend per share in dividend
        - "right_shares": right share per share in dividend
        - "right_price": right price per share in dividend
      unit: trading unit, default is 100
      slippage: slippage of the buy and sell operations, for buy: deal_price = price * (1 + slippage), for sell: deal_price = price * (1 - slippage)
    """


    """initial cash of the account"""
    self.initial_cash = initial_cash
    self.cash = initial_cash
    """data provider of the account"""
    self.dp = dp
    """trading unit"""
    self.unit = unit
    """position of the account"""
    self.position = Position(0, 0)
    """assets after first buy operation"""
    self.per_assets = 0.0
    """deducted commission and other fees"""
    self.deducted = 0.0
    """records of the account"""
    self.records = []
    """fee calculator"""
    self.fee = fee if fee is not None else Fee()
    """slippage of the buy and sell operations"""
    self.slippage = slippage
    """group index"""
    self.group = group

    s = self.group * dp.bars
    e = s + dp.bars
    """dates of current group"""
    self.dates = dp.dates[s:e].astype('datetime64[s]').astype(int)
    """history of assets for current group"""
    self.assets = dp.all("assets", create_if_not_exist=np.float64)[s:e]
    """history of rate of return for current group"""
    self.ror = dp.all("ror", create_if_not_exist=np.float64)[s:e]
    """history of rate of return since last clear"""
    self.ror_hold = dp.all("ror_hold", create_if_not_exist=np.float64)[s:e]

    self.actions = dp.all("actions", create_if_not_exist=np.int32)[s:e]
    self.action_shares = dp.all("action_shares", create_if_not_exist=np.float64)[s:e]

    """corporate actions"""
    self.dividends = dp.all("dividend")[s:e]
    self.transfer_shares = dp.all("transfer_shares")[s:e]
    self.right_shares = dp.all("right_shares")[s:e]
    self.right_prices = dp.all("right_price")[s:e]

    self.assets.fill(initial_cash)
    self.ror.fill(0.0)
    self.actions.fill(ACTION_KEEP)
    self.action_shares.fill(0.0)

  def clone(self, group: int = 0) -> "Account":
    return Account(
      self.cash,
      self.dp,
      group=group,
      fee=self.fee,
      unit=self.unit,
      slippage=self.slippage,
    )

  @property
  def asset(self):
    """current asset"""
    return self.position.value + self.cash

  def do_operation(self, ts: int, operation: Operation):
    """
    perform operation, add record to records

    Args:
        ts: timestamp of the record
        operation: incoming operation of the record
    """

    if operation.position is None:
      raise ValueError(f"invalid operation: {operation}")

    fee = 0.0
    if self.position.shares > 0:
      dividend = self.dividends[ts]
      transfer_shares = self.transfer_shares[ts]
      right_share = self.right_shares[ts]
      right_price = self.right_prices[ts]

      if dividend > 0 or transfer_shares > 0 or right_share > 0:
        self.cash += self.position.shares * dividend
        self.cash -= self.position.shares * right_share * right_price
        self.position.shares += self.position.shares * (transfer_shares + right_share)

    if operation.action == ACTION_BUY:
      operation, fee = self.do_buy(ts, operation.position)
    elif operation.action == ACTION_BUY_PERCENT:
      operation, fee = self.do_buy(
        ts,
        Position.from_cash(
          self.cash * operation.position.shares, operation.position.price, self.unit
        ),
      )
    elif operation.action == ACTION_SELL:
      operation, fee = self.do_sell(ts, operation.position)
    elif operation.action == ACTION_SELL_PERCENT:
      operation, fee = self.do_sell(
        ts,
        Position(
          self.position.shares * operation.position.shares,
          operation.position.price,
        ),
      )
    elif operation.action == ACTION_KEEP:
      self.position.price = operation.position.price
    else:
      raise ValueError(f"invalid operation: {operation}")

    hold_profit = (
      (self.asset - self.per_assets) / self.per_assets if self.per_assets > 0 else 0.0
    )
    self.records.append(
      Record(ts, operation, self.cash, hold_profit, self.position, fee)
    )
    self.assets[ts] = self.asset
    self.ror[ts] = self.asset / self.assets[0] - 1
    self.ror_hold[ts] = hold_profit
    self.actions[ts] = operation.action
    self.action_shares[ts] = (
      operation.position.shares if operation.position is not None else 0.0
    )
    logger.debug(
      f"group: {self.group}, ts: {ts}, operation: {operation}, cash: {self.cash}, position: {self.position}, asset: {self.asset}, ror: {self.ror[ts]}"
    )

  def do_buy(self, ts: int, position: Position) -> tuple[Operation, float]:
    """
    perform buy operation

    Args:
        ts: timestamp of the record
        position: position to buy

    Returns:
        operation: because buy operation may be rejected or partially filled, return the actual operation
    """

    if self.cash == 0:
      return Operation.buy_no_cash(Position(0, 0))

    fee_item = self.fee.get(self.dates[ts])
    fee = fee_item.buy(position.value)
    if position.value + fee > self.cash:
      position = Position.from_cash(self.cash - fee, position.price, self.unit)

    if position.shares == 0:
      return Operation.buy_no_cash(Position(0, 0)), 0.0

    self.cash -= position.value + fee
    self.position += position
    fee = fee_item.buy(position.value)
    self.deducted += fee
    if self.per_assets == 0:
      self.per_assets = self.asset
    return Operation.buy(position), fee

  def do_sell(self, ts: int, position: Position) -> tuple[Operation, float]:
    """
    perform sell operation
    Args:
        ts: timestamp of the record
        position: position to sell

    Returns:
        operation: because sell operation may be rejected or partially filled, return the actual operation
    """

    if position.shares > self.position.shares:
      position.shares = self.position.shares

    if position.shares == 0:
      return Operation.sell_no_hold(Position(0, 0)), 0.0

    fee_item = self.fee.get(self.dates[ts])
    fee = fee_item.sell(position.value)
    self.cash += position.value - fee
    self.position -= position
    self.deducted += fee
    if self.position.shares == 0:
      self.per_assets = 0.0
    return Operation.sell(position), fee
