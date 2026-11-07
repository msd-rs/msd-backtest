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


def test_account_zero_cash_and_fee_consistency():
  from mbt.account import Fee, FeeItem, ACTION_BUY_NO_CASH

  dp = DataProvider(
    {
      "ts": np.arange(5),
      "price": np.array([10.0, 10.0, np.nan, 10.0, 10.0]),
      "dividend": np.zeros(5),
      "transfer_shares": np.zeros(5),
      "right_shares": np.array([0.0, 0.2, 0.0, 0.0, 0.0]),
      "right_price": np.array([0.0, 8.0, 0.0, 0.0, 0.0]),
    }
  )
  fee = Fee([FeeItem(ts=0, commission_buy=0.001, min_commission_buy=5.0)])
  # Initial cash 1005.0 -> 100 shares @ 10.0 = 1000.0 + 5.0 min_commission = 1005.0 -> cash becomes 0.0
  account = Account(1005.0, dp, fee=fee)
  account.do_operation(0, Operation.buy_percent(1.0, 10.0))
  assert account.position.shares == 100.0
  assert np.isclose(account.cash, 0.0)
  assert np.isclose(account.deducted, 5.0)
  assert np.isclose(account.initial_cash - account.cash - account.position.value, account.deducted)

  # Bar 1: cash is 0.0, right_shares=0.2 @ 8.0 -> should not make cash negative; buy again should not raise TypeError
  account.do_operation(1, Operation.buy_percent(1.0, 10.0))
  assert account.cash >= 0.0
  assert account.actions[1] == ACTION_BUY_NO_CASH

  # Bar 2: suspended (price is NaN) -> keep previous valid position price (10.0), assets should not be NaN
  account.do_operation(2, Operation.keep(np.nan))
  assert not np.isnan(account.assets[2])
  assert np.isclose(account.assets[2], 1000.0)



if __name__ == "__main__":
  test_account_buy_sell()
