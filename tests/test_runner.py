from mbt.runner import Runner
from mbt.strategy import Context
from mbt.strategy import Strategy
from mbt.account import Account
from mbt.data_provider import DataProvider
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TestStrategy(Strategy):
  def execute(self, ctx: Context):
    buy = ctx.close == 2.0
    sell = ctx.close == 3.0
    ctx.buy(buy, 1.0)
    ctx.sell(sell, 1.0)


def test_runner():
  dp = DataProvider(
    {
      "ts": np.tile(np.arange(5), 2),
      "open": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
      "high": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
      "low": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
      "close": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
      "price": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
    },
    symbols=["a", "b"],
  )
  account = Account(1000.0, dp)

  runner = Runner(account)
  runner.run(TestStrategy())

  ror = dp.all("ror")
  ror_v = np.tile([0.0, 0.0, 0.5, 0.5, 0.5], 2)
  assert np.allclose(ror, ror_v)


if __name__ == "__main__":
  test_runner()
