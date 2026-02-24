from .data_provider import DataProvider
import numpy as np


ACTION_KEEP = 0

ACTION_BUY = 1
ACTION_BUY_PERCENT = 3
ACTION_BUY_NO_CASH = 5

ACTION_SELL = 2
ACTION_SELL_PERCENT = 4
ACTION_SELL_NO_HOLD = 6


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
    shares = int(cash / price / unit)
    shares = shares - shares % unit  # round down to unit
    return Position(float(shares), price)

  def __add__(self, other: "Position") -> "Position":
    value = self.value + other.value
    shares = self.shares + other.shares
    return Position(shares, value / shares)

  def __sub__(self, other: "Position") -> "Position":
    value = self.value - other.value
    shares = self.shares - other.shares
    if shares <= 0:
      return Position.zero()
    return Position(shares, value / shares)


class Operation:
  """Operation is a operation in a day of a backtest account"""

  def __init__(self, action: int, position: Position | None):
    self.action = action
    self.position = position

  @staticmethod
  def keep() -> "Operation":
    return Operation(ACTION_KEEP, None)

  @staticmethod
  def buy(position: Position) -> "Operation":
    return Operation(ACTION_BUY, position)

  @staticmethod
  def sell(position: Position) -> "Operation":
    return Operation(ACTION_SELL, position)

  @staticmethod
  def buy_no_cash(position: Position) -> "Operation":
    return Operation(ACTION_BUY_NO_CASH, position)

  @staticmethod
  def sell_no_hold(position: Position) -> "Operation":
    return Operation(ACTION_SELL_NO_HOLD, position)


class Fee:
  def __init__(self, dp: DataProvider):
    self.dp = dp

  def buy(self, value: float) -> float:
    commission = value * self.dp.last("COMMISSION_BUY", 0.0)
    min_commission = self.dp.last("MIN_COMMISSION_BUY", 0.0)
    tax = value * self.dp.last("TAX_BUY", 0.0)
    transaction_fee = self.dp.last("TRANSACTION_FEE_BUY", 0.0)
    return max(commission, min_commission) + tax + transaction_fee

  def sell(self, value: float) -> float:
    commission = value * self.dp.last("COMMISSION_SELL", 0.0)
    min_commission = self.dp.last("MIN_COMMISSION_SELL", 0.0)
    tax = value * self.dp.last("TAX_SELL", 0.0)
    transaction_fee = self.dp.last("TRANSACTION_FEE_SELL", 0.0)
    return max(commission, min_commission) + tax + transaction_fee


class Record:
  """Record is a record of a backtest account"""

  def __init__(
    self,
    date: int,
    operation: Operation,
    cash: float,
    hold_profit: float,
    position: Position,
    deducted: float,
  ):
    """
    Args:
      date: date of the record
      operation: incoming operation of the record
      cash: cash after operation
      hold_profit: profit when position was bought
      position: position after operation
      deducted: deducted commission and other fees
    """

    """the date of the record"""
    self.date = date
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

  def __init__(self, initial_cash: float, dp: DataProvider):
    """
    Args:
      initial_cash: initial cash of the account
      dp: data provider of the account, the data kinds will be used are:
        - "PRICE": price of the stock, without dividend
        - "GCASH": given cash per share in dividend
        - "GSHARE": given share per share in dividend
    """

    """initial cash of the account"""
    self.cash = initial_cash
    """data provider of the account"""
    self.dp = dp
    """position of the account"""
    self.position = Position(0, 0)
    """assets after first buy operation"""
    self.per_assets = 0.0
    """deducted commission and other fees"""
    self.deducted = 0.0
    """records of the account"""
    self.records = []
    """fee calculator"""
    self.fee = Fee(dp)

  @property
  def assets(self):
    return self.position.value + self.cash

  def do_operation(self, date: int, operation: Operation):
    """
    Args:
      date: date of the record
      operation: incoming operation of the record
    """

    if operation.position is None:
      return

    if operation.action == ACTION_BUY:
      operation = self.do_buy(operation.position)
    elif operation.action == ACTION_BUY_PERCENT:
      operation = self.do_buy(
        Position.from_cash(
          self.cash * operation.position.shares, operation.position.price
        )
      )
    elif operation.action == ACTION_SELL:
      operation = self.do_sell(operation.position)
    elif operation.action == ACTION_SELL_PERCENT:
      operation = self.do_sell(
        Position(
          float(int(self.position.shares * operation.position.shares)),
          operation.position.price,
        )
      )
    hold_profit = (
      (self.assets - self.per_assets) / self.per_assets if self.per_assets > 0 else 0.0
    )
    self.records.append(
      Record(date, operation, self.cash, hold_profit, self.position, self.deducted)
    )

  def do_buy(self, position: Position) -> Operation:
    """
    Args:
      position: position to buy
    """

    if self.cash == 0:
      return Operation.buy_no_cash(Position(0, 0))

    fee = self.fee.buy(position.value)
    if position.value + fee > self.cash:
      position = Position.from_cash(self.cash - fee, position.price)

    self.cash -= position.value + fee
    self.position += position
    self.deducted = self.fee.buy(position.value)
    if self.per_assets == 0:
      self.per_assets = self.assets
    return position

  def do_sell(self, position: Position) -> Operation:
    """
    Args:
      position: position to sell
    """

    if position.shares > self.position.shares:
      position.shares = self.position.shares

    self.cash += position.value - self.fee.sell(position.value)
    self.position -= position
    self.deducted = self.fee.sell(position.value)
    if self.position.shares == 0:
      self.per_assets = 0.0
    return position
