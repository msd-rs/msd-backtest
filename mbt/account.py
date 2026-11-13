from pathlib import Path
import json
from dataclasses import dataclass
from typing import Any
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
  """Position represents a stock holding position in an account.

  Attributes:
    shares (float): Number of shares held (>= 0).
    raw_cost_price (float): Average unadjusted acquisition price per share (真实成交未复权加权买入均价).
    cost_basis (float): Total acquisition cost of currently held shares (shares * raw_cost_price).
    raw_mark_price (float): Latest unadjusted mark-to-market price per share (最新未复权收盘价/现价).
    adj_cost_price (float): Forward-adjusted equivalent cost price (前复权等效成本价),
      which can be directly compared with forward-adjusted prices like `ctx.close`.
    dividends (float): Accumulated cash dividends received during holding this position.
    holding_ror (float): Rate of return of current position as a decimal (e.g. 0.05 for +5%).
    holding_pnl (float): Net profit/loss amount of current position.
    ror (float): Alias for holding_ror.
    unrealized_pnl (float): Alias for holding_pnl.
    pnl (float): Alias for holding_pnl.
    holding_days (int): Number of bars/trading days this position has been held.
    holding_bars (int): Alias for holding_days.
    is_holding (bool): True if currently holding shares (shares > 0).
  """

  def __init__(
    self,
    shares: float,
    price: float = 0.0,
    raw_cost_price: float | None = None,
    raw_mark_price: float | None = None,
    dividends: float = 0.0,
    adj_cost_price: float | None = None,
    holding_days: int = 0,
  ):
    self.shares = float(shares)
    init_price = float(price) if price is not None else 0.0
    self.raw_cost_price = float(raw_cost_price) if raw_cost_price is not None else init_price
    self.raw_mark_price = float(raw_mark_price) if raw_mark_price is not None else init_price
    self.dividends = float(dividends)
    self._adj_cost_price = float(adj_cost_price) if adj_cost_price is not None else self.raw_cost_price
    self._holding_days = max(0, int(holding_days)) if holding_days is not None else 0

    if self.shares <= 0:
      self.shares = 0.0
      self.raw_cost_price = 0.0
      self.raw_mark_price = 0.0
      self.dividends = 0.0
      self._adj_cost_price = 0.0
      self._holding_days = 0

  @property
  def cost_price(self) -> float:
    """Alias for raw_cost_price."""
    return self.raw_cost_price

  @cost_price.setter
  def cost_price(self, val: float):
    self.raw_cost_price = float(val)

  @property
  def mark_price(self) -> float:
    """Alias for raw_mark_price."""
    return self.raw_mark_price

  @mark_price.setter
  def mark_price(self, val: float):
    self.raw_mark_price = float(val)

  @property
  def price(self) -> float:
    """[AMBIGUOUS] Backward-compatible price accessor.

    WARNING: Do NOT mix pos.price with forward-adjusted `ctx.close`!
    - For holding rate of return / stop-loss: use `pos.holding_ror` or `acc.holding_ror(obj)`.
    - For unadjusted acquisition cost: use `pos.raw_cost_price`.
    - For forward-adjusted equivalent cost: use `pos.adj_cost_price`.
    - For current unadjusted market price: use `pos.raw_mark_price`.
    """
    if self.raw_mark_price > 0:
      return self.raw_mark_price
    return self.raw_cost_price

  @price.setter
  def price(self, val: float):
    p = float(val)
    self.raw_mark_price = p
    if self.raw_cost_price == 0.0 and p > 0:
      self.raw_cost_price = p

  @property
  def cost_basis(self) -> float:
    """Total acquisition cash cost for currently held shares."""
    return self.shares * self.raw_cost_price

  @property
  def market_value(self) -> float:
    """Current market value based on latest mark price."""
    p = self.raw_mark_price if self.raw_mark_price > 0 else self.raw_cost_price
    return self.shares * p

  @property
  def value(self) -> float:
    """Alias for market_value."""
    return self.market_value

  @property
  def adj_cost_price(self) -> float:
    """Forward-adjusted equivalent cost price (前复权等效成本价).

    Safe to compare directly with forward-adjusted prices such as `ctx.close`.
    """
    return self._adj_cost_price

  @adj_cost_price.setter
  def adj_cost_price(self, val: float):
    self._adj_cost_price = float(val)

  @property
  def holding_pnl(self) -> float:
    """Holding profit/loss amount: (market_value + accumulated dividends) - cost_basis."""
    if self.shares <= 0 or self.cost_basis <= 0:
      return 0.0
    return (self.market_value + self.dividends) - self.cost_basis

  @property
  def holding_ror(self) -> float:
    """Holding rate of return as a decimal (e.g. -0.05 for -5% loss).

    Accurately accounts for bonus shares (splits), cash dividends, and dilution.
    """
    if self.shares <= 0 or self.cost_basis <= 0:
      return 0.0
    return self.holding_pnl / self.cost_basis

  @property
  def ror(self) -> float:
    """Holding rate of return as a decimal (alias for holding_ror, e.g. -0.05 for -5% loss)."""
    return self.holding_ror

  @property
  def unrealized_pnl(self) -> float:
    """Unrealized net profit/loss amount of currently held shares (alias for holding_pnl)."""
    return self.holding_pnl

  @property
  def pnl(self) -> float:
    """Alias for holding_pnl / unrealized_pnl."""
    return self.holding_pnl

  @property
  def holding_days(self) -> int:
    """Number of bars/trading days this position has been held."""
    return getattr(self, "_holding_days", 0)

  @holding_days.setter
  def holding_days(self, val: int):
    self._holding_days = max(0, int(val))

  @property
  def holding_bars(self) -> int:
    """Alias for holding_days."""
    return self.holding_days

  @property
  def is_holding(self) -> bool:
    """True if currently holding shares (shares > 0)."""
    return self.shares > 0

  @staticmethod
  def zero() -> "Position":
    return Position(0.0, 0.0)

  @staticmethod
  def from_cash(cash: float, price: float, unit: int = 100) -> "Position":
    if price <= 0 or np.isnan(price):
      return Position.zero()
    shares = int(cash / price)
    shares = shares - shares % unit  # round down to unit
    return Position(float(shares), float(price))

  def __add__(self, other: "Position") -> "Position":
    shares = self.shares + other.shares
    if shares <= 0:
      return Position.zero()
    cost_basis = self.cost_basis + other.cost_basis
    raw_cost_price = cost_basis / shares
    raw_mark_price = other.raw_mark_price if other.raw_mark_price > 0 else self.raw_mark_price
    dividends = self.dividends + other.dividends
    adj_cost_basis = (self.shares * self._adj_cost_price) + (other.shares * other._adj_cost_price)
    adj_cost_price = adj_cost_basis / shares if shares > 0 else 0.0
    # Retain holding duration from existing holding if present
    h_days = self._holding_days if self.shares > 0 else other._holding_days
    return Position(
      shares,
      raw_cost_price,
      raw_cost_price=raw_cost_price,
      raw_mark_price=raw_mark_price,
      dividends=dividends,
      adj_cost_price=adj_cost_price,
      holding_days=h_days,
    )

  def __sub__(self, other: "Position") -> "Position":
    shares = self.shares - other.shares
    if shares <= 0:
      return Position.zero()
    # Partial liquidation retains same per-share average cost
    raw_cost_price = self.raw_cost_price
    raw_mark_price = other.raw_mark_price if other.raw_mark_price > 0 else self.raw_mark_price
    ratio = shares / self.shares if self.shares > 0 else 0.0
    dividends = self.dividends * ratio
    adj_cost_price = self._adj_cost_price
    return Position(
      shares,
      raw_cost_price,
      raw_cost_price=raw_cost_price,
      raw_mark_price=raw_mark_price,
      dividends=dividends,
      adj_cost_price=adj_cost_price,
      holding_days=self._holding_days,
    )

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


DEFAULT_FEE_CONFIG: list[dict] = [
  {
    "ts": 4102416000,
    "commission_buy": 0.0015,
    "min_commission_buy": 5.0,
    "tax_buy": 0.001,
    "transaction_fee_buy": 0.0001,
    "commission_sell": 0.0015,
    "min_commission_sell": 5.0,
    "tax_sell": 0.0,
    "transaction_fee_sell": 0.0,
  }
]


class Fee:
  def __init__(
    self,
    config: list[FeeItem] | list[str] | list[dict] | str | Path | bytes | None = None,
  ):
    def build_fee_item(x):
      return FeeItem(**x)

    if isinstance(config, list):
      if len(config) > 0 and isinstance(config[0], str):
        self.config = [json.loads(x, object_hook=build_fee_item) for x in config]  # type: ignore
        self.config.sort(key=lambda x: x.ts)
      elif len(config) > 0 and isinstance(config[0], dict):
        self.config = [FeeItem(**x) for x in config]  # type: ignore
        self.config.sort(key=lambda x: x.ts)
      else:
        self.config = list(config)
    elif isinstance(config, (str, Path)):
      p = Path(config)

      if p.is_file():
        self.config = json.load(p.open("r", encoding="utf-8"), object_hook=build_fee_item)
      else:
        project_p = Path(__file__).resolve().parent.parent / p
        if project_p.is_file():
          self.config = json.load(project_p.open("r", encoding="utf-8"), object_hook=build_fee_item)
        else:
          try:
            self.config = json.loads(str(config), object_hook=build_fee_item)
          except Exception:
            logger.warning(
              "Fee config '%s' not found or invalid JSON, falling back to default fee configuration",
              config,
            )
            self.config = [FeeItem(**item) for item in DEFAULT_FEE_CONFIG]
    elif isinstance(config, bytes):
      self.config = json.loads(config.decode("utf-8"), object_hook=build_fee_item)
    else:
      self.config = [FeeItem()]

    # add a default fee item at the beginning when no fee item is provided
    if len(self.config) == 0:
      self.config = [FeeItem()]

  def apply_overrides(self, overrides: dict[str, Any] | None) -> None:
    """Apply fee parameter overrides from a dict (e.g. from CLI -a arguments)."""
    if not overrides:
      return

    applied: dict[str, float] = {}
    for item in self.config:
      if "commission" in overrides:
        c = float(overrides["commission"])
        item.commission_buy = c
        item.commission_sell = c
        applied["commission"] = c
      if "min_commission" in overrides:
        mc = float(overrides["min_commission"])
        item.min_commission_buy = mc
        item.min_commission_sell = mc
        applied["min_commission"] = mc
      if "tax" in overrides:
        t = float(overrides["tax"])
        if getattr(item, "tax_buy", 0.0) > 0 and getattr(item, "tax_sell", 0.0) == 0.0:
          item.tax_buy = t
        elif getattr(item, "tax_sell", 0.0) > 0 and getattr(item, "tax_buy", 0.0) == 0.0:
          item.tax_sell = t
        else:
          item.tax_sell = t
        applied["tax"] = t
      if "transaction_fee" in overrides:
        tf = float(overrides["transaction_fee"])
        item.transaction_fee_buy = tf
        item.transaction_fee_sell = tf
        applied["transaction_fee"] = tf

      for field in (
        "commission_buy",
        "commission_sell",
        "min_commission_buy",
        "min_commission_sell",
        "tax_buy",
        "tax_sell",
        "transaction_fee_buy",
        "transaction_fee_sell",
      ):
        if field in overrides:
          val = float(overrides[field])
          setattr(item, field, val)
          applied[field] = val

    if applied:
      logger.info("Fee configuration overridden by parameters: %s", applied)

  @classmethod
  def default(cls) -> "Fee":
    """
    Return a Fee instance initialized with default project fee config.
    Tries ./etc/fee.json and project_root/etc/fee.json, falling back to DEFAULT_FEE_CONFIG.
    """
    for candidate in [
      Path.cwd() / "etc" / "fee.json",
      Path(__file__).resolve().parent.parent / "etc" / "fee.json",
    ]:
      if candidate.is_file():
        try:
          return cls(candidate)
        except Exception as e:
          logger.warning("Failed to load fee config from %s: %s", candidate, e)
    return cls([FeeItem(**item) for item in DEFAULT_FEE_CONFIG])

  @classmethod
  def load(
    cls,
    config: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
  ) -> "Fee":
    """
    Load Fee from a given file path or JSON string, optionally applying overrides.
    If config is empty/None or file not found, fall back to default fee configuration.
    """
    fee: "Fee"
    if not config:
      fee = cls.default()
    else:
      p = Path(config)
      if p.is_file():
        try:
          fee = cls(p)
        except Exception as e:
          logger.warning(
            "Failed to load fee config from %s: %s, falling back to default fee",
            p,
            e,
          )
          fee = cls.default()
      else:
        project_p = Path(__file__).resolve().parent.parent / p
        if project_p.is_file():
          try:
            fee = cls(project_p)
          except Exception as e:
            logger.warning(
              "Failed to load fee config from %s: %s, falling back to default fee",
              project_p,
              e,
            )
            fee = cls.default()
        elif isinstance(config, str) and config.strip().startswith(("{", "[")):
          try:
            fee = cls(config)
          except Exception as e:
            logger.warning(
              "Failed to parse fee JSON config: %s, falling back to default fee",
              e,
            )
            fee = cls.default()
        else:
          logger.warning(
            "Fee config '%s' not found, falling back to default fee configuration",
            config,
          )
          fee = cls.default()

    if overrides:
      fee.apply_overrides(overrides)
    return fee

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
    self.value = self.position.market_value
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
    self._holding_bars = 0
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
        if dividend > 0 and not np.isnan(dividend):
          self.cash += self.position.shares * dividend
          self.position.dividends += self.position.shares * dividend
        if transfer_shares > 0 and not np.isnan(transfer_shares):
          old_shares = self.position.shares
          self.position.shares += self.position.shares * transfer_shares
          if self.position.shares > 0:
            ratio = old_shares / self.position.shares
            self.position.raw_cost_price *= ratio
            self.position._adj_cost_price *= ratio
        if right_share > 0 and not np.isnan(right_share):
          max_right_shares = self.position.shares * right_share
          if right_price > 0 and not np.isnan(right_price):
            affordable = int(self.cash / right_price) if self.cash > 0 else 0
            actual_right = min(max_right_shares, float(affordable))
            self.cash -= actual_right * right_price
            right_cost = actual_right * right_price
          else:
            actual_right = max_right_shares
            right_cost = 0.0
          if actual_right > 0:
            old_cost = self.position.cost_basis
            old_raw_cost = self.position.raw_cost_price
            self.position.shares += actual_right
            if self.position.shares > 0:
              self.position.raw_cost_price = (old_cost + right_cost) / self.position.shares
              if old_raw_cost > 0:
                self.position._adj_cost_price = self.position._adj_cost_price * (
                  self.position.raw_cost_price / old_raw_cost
                )

    op_price = operation.position.price
    is_valid_price = not np.isnan(op_price) and op_price > 0

    if operation.action == ACTION_BUY:
      operation, fee = self.do_buy(ts, operation.position)
    elif operation.action == ACTION_BUY_PERCENT:
      if not is_valid_price:
        operation, fee = Operation.buy_no_cash(Position(0, 0)), 0.0
      else:
        operation, fee = self.do_buy(
          ts,
          Position.from_cash(
            self.cash * operation.position.shares, operation.position.price, self.unit
          ),
        )
    elif operation.action == ACTION_SELL:
      if not is_valid_price:
        operation, fee = Operation.sell_no_hold(Position(0, 0)), 0.0
      else:
        operation, fee = self.do_sell(ts, operation.position)
    elif operation.action == ACTION_SELL_PERCENT:
      if not is_valid_price:
        operation, fee = Operation.sell_no_hold(Position(0, 0)), 0.0
      else:
        operation, fee = self.do_sell(
          ts,
          Position(
            self.position.shares * operation.position.shares,
            operation.position.price,
          ),
        )
    elif operation.action == ACTION_KEEP:
      if is_valid_price:
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
    if self.position.shares > 0:
      self._holding_bars += 1
      self.position.holding_days = self._holding_bars
    else:
      self._holding_bars = 0
      self.position.holding_days = 0
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

    if self.cash <= 0 or np.isnan(position.price) or position.price <= 0:
      return Operation.buy_no_cash(Position(0, 0)), 0.0

    fee_item = self.fee.get(self.dates[ts])
    fee = fee_item.buy(position.value)
    if position.value + fee > self.cash:
      position = Position.from_cash(max(self.cash - fee, 0.0), position.price, self.unit)
      fee = fee_item.buy(position.value)
      while position.shares > 0 and position.value + fee > self.cash:
        position = Position(max(position.shares - self.unit, 0.0), position.price)
        fee = fee_item.buy(position.value)

    if position.shares <= 0:
      return Operation.buy_no_cash(Position(0, 0)), 0.0

    # Initialize forward-adjusted equivalent price if data available
    j = self.group * self.dp.bars + ts
    adj_cost = position.price
    if hasattr(self.dp, "data") and "close" in self.dp.data and "price" in self.dp.data:
      raw_p = self.dp.data["price"][j]
      fw_p = self.dp.data["close"][j]
      if raw_p > 0 and not np.isnan(raw_p) and not np.isnan(fw_p):
        adj_cost = position.price * (fw_p / raw_p)
    position._adj_cost_price = float(adj_cost)

    self.cash -= position.value + fee
    self.position += position
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

  @property
  def holding_ror(self) -> float:
    """Holding rate of return of the current position as a decimal (e.g. -0.05 for -5%)."""
    return self.position.holding_ror

  @property
  def holding_pnl(self) -> float:
    """Holding profit/loss amount of the current position."""
    return self.position.holding_pnl

  def cost_price(self, adjusted: bool = False) -> float:
    """Return position acquisition cost price.

    Args:
      adjusted: If True, returns forward-adjusted equivalent cost (adj_cost_price),
        safe to compare with `ctx.close`. If False, returns unadjusted raw execution cost.
    """
    return self.position.adj_cost_price if adjusted else self.position.raw_cost_price

  @property
  def holding_days(self) -> int:
    """Number of bars current position has been held."""
    return getattr(self, "_holding_bars", 0)
