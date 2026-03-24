from mbt import Runner, Context, Strategy, Account, DataProvider
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TestStrategy(Strategy):
  def execute(self, ctx: Context):
    buy = ctx.close == 2.0
    sell = ctx.close == 3.0
    ctx.buy(buy, 1.0)
    ctx.sell(sell, 1.0)


def run_strategy(dp: DataProvider):
  account = Account(1000.0, dp)
  runner = Runner(account)
  runner.run(TestStrategy())


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

  run_strategy(dp)

  ror = dp.all("ror")
  ror_v = np.tile([0.0, 0.0, 0.5, 0.5, 0.5], 2)
  assert np.allclose(ror, ror_v)


def test_bench_runner(benchmark):
  SYMBOLS = 300
  BARS = 10000
  dp = DataProvider(
    {
      "ts": np.tile(np.arange(BARS), SYMBOLS),
      "open": np.tile(np.arange(BARS, dtype=np.float64), SYMBOLS),
      "high": np.tile(np.arange(BARS, dtype=np.float64), SYMBOLS),
      "low": np.tile(np.arange(BARS, dtype=np.float64), SYMBOLS),
      "close": np.tile(np.arange(BARS, dtype=np.float64), SYMBOLS),
      "price": np.tile(np.arange(BARS, dtype=np.float64), SYMBOLS),
    },
    symbols=[f"a{i}" for i in range(SYMBOLS)],
  )
  benchmark(run_strategy, dp)


if __name__ == "__main__":
  test_runner()
