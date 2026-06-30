from mbt import (
  Runner,
  Context,
  Strategy,
  Account,
  DataProvider,
  ACTION_BUY,
  ACTION_BUY_NO_CASH,
  ACTION_SELL,
  ACTION_SELL_NO_HOLD,
  ACTION_KEEP,
)
import numpy as np


class SyntheticStrategy(Strategy):
  def execute(self, ctx: Context):
    # Buy when price is close to 10.33
    buy_signal = np.isclose(ctx.close, 10.33)
    # Sell when price is close to 20.45
    sell_signal = np.isclose(ctx.close, 20.45)

    ctx.buy(buy_signal, 1.0)
    ctx.sell(sell_signal, 1.0)


def test_strategy_actions_and_assets_with_unrounded_data():
  # Create synthetic data with 6 bars using unrounded prices
  prices = np.array([10.33, 10.33, 10.33, 20.45, 20.45, 10.33])
  dates = np.arange(len(prices))

  dp = DataProvider(
    {
      "ts": dates,
      "open": prices,
      "high": prices,
      "low": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "fw_price": prices,
      "dividend": np.zeros(len(prices)),
      "transfer_shares": np.zeros(len(prices)),
      "right_shares": np.zeros(len(prices)),
      "right_price": np.zeros(len(prices)),
    },
    symbols=["test_sym"],
  )

  # Initial cash 1500.0
  account = Account(1500.0, dp)
  runner = Runner(account)
  runner.run(SyntheticStrategy())

  # Expected actions:
  # Bar 0 (price 10.33): Skipped by Runner (iterator starts at i=1). Action: Keep
  # Bar 1 (price 10.33): Buy triggered. 1500 / 10.33 = 145.2. unit=100 -> 100 shares. Cost: 1033.0. cash=467.0
  # Bar 2 (price 10.33): Buy triggered again. 467 / 10.33 = 45.2. unit=100 -> 0 shares -> BuyNoCash
  # Bar 3 (price 20.45): Sell triggered. Sell 100 shares. Gained: 2045.0. cash=2512.0
  # Bar 4 (price 20.45): Sell triggered again, but shares=0 -> SellNoHold
  # Bar 5 (price 10.33): Buy triggered. 2512 / 10.33 = 243.1. unit=100 -> 200 shares. Cost: 2066.0. cash=446.0

  expected_actions = [
    ACTION_KEEP,
    ACTION_BUY,
    ACTION_BUY_NO_CASH,
    ACTION_SELL,
    ACTION_SELL_NO_HOLD,
    ACTION_BUY,
  ]
  actual_actions = account.actions.tolist()
  assert actual_actions == expected_actions, (
    f"Expected actions {expected_actions}, got {actual_actions}"
  )

  # Expected assets tracking cost basis:
  # Bar 0 (price 10.33): 1500 cash
  # Bar 1 (price 10.33): 100 shares @ 10.33 + 467.0 cash = 1033.0 + 467.0 = 1500.0
  # Bar 2 (price 10.33): 100 shares @ 10.33 + 467.0 cash = 1500.0
  # Bar 3 (price 20.45): 0 shares + 2512.0 cash = 2512.0
  # Bar 4 (price 20.45): 0 shares + 2512.0 cash = 2512.0
  # Bar 5 (price 10.33): 200 shares @ 10.33 + 446.0 cash = 2066.0 + 446.0 = 2512.0
  expected_assets = [1500.0, 1500.0, 1500.0, 2512.0, 2512.0, 2512.0]
  actual_assets = account.assets.tolist()
  assert np.allclose(actual_assets, expected_assets), (
    f"Expected assets {expected_assets}, got {actual_assets}"
  )


if __name__ == "__main__":
  test_strategy_actions_and_assets_with_unrounded_data()
