from mbt import (
  Account,
  DataProvider,
  Operation,
  Position,
)
from mbt.account import (
  ACTION_BUY_PERCENT,
  ACTION_SELL_PERCENT,
)
import numpy as np


def test_account_buy_sell():
  dp = DataProvider(
    {
      "ts": np.arange(20),
      "price": np.array([1.0] * 20),
      "dividend": np.zeros(20),
      "transfer_shares": np.zeros(20),
      "right_shares": np.zeros(20),
      "right_price": np.zeros(20),
    }
  )
  account = Account(1000.0, dp)

  prices = dp.all("price")

  # Verify initial state
  assert account.cash == 1000.0
  assert account.position.shares == 0.0
  assert account.assets[0] == 1000.0

  # 1. Buy 500 shares at bar 0
  buy_pos1 = Position.from_cash(500.0, prices[0])
  account.do_operation(0, Operation.buy(buy_pos1))

  assert account.cash == 500.0
  assert account.position.shares == 500.0
  assert account.assets[0] == 1000.0

  # 2. Buy with remaining cash (100%) at bar 5
  buy_pos_percent = Position(shares=1.0, price=prices[5])
  account.do_operation(5, Operation(ACTION_BUY_PERCENT, buy_pos_percent))

  assert account.cash == 0.0
  assert account.position.shares == 1000.0
  assert account.assets[5] == 1000.0

  # 3. Sell 50% at bar 10
  sell_pos_percent = Position(shares=0.5, price=prices[10])
  account.do_operation(10, Operation(ACTION_SELL_PERCENT, sell_pos_percent))

  assert account.cash == 500.0
  assert account.position.shares == 500.0
  assert account.assets[10] == 1000.0

  # 4. Sell remaining at bar 19
  sell_pos2 = Position(account.position.shares, prices[19])
  account.do_operation(19, Operation.sell(sell_pos2))

  assert account.cash == 1000.0
  assert account.position.shares == 0.0
  assert account.assets[19] == 1000.0

  # Verify records and that no fees were deducted
  assert len(account.records) == 4
  for record in account.records:
    assert record.deducted == 0.0


if __name__ == "__main__":
  test_account_buy_sell()
