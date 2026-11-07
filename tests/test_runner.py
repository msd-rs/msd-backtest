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
      "bw_price": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
      "fw_price": np.tile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 2),
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["a", "b"],
  )

  run_strategy(dp)

  ror = dp.all("ror")
  ror_v = np.tile([0.0, 0.0, 0.5, 0.5, 0.5], 2)
  assert np.allclose(ror, ror_v)


def test_runner_prepare_and_panel():
  import alpha as al

  class PrecomputedStrategy(Strategy):
    def __init__(self):
      super().__init__()
      self.visited_bars = []

    def prepare(self, ctx: Context):
      panel = ctx.panel()
      assert panel.n_objs == 2
      assert panel.bars == 5
      assert ctx.n_objs == 2
      assert ctx.bars == 5
      # Verify panel.compute inside prepare() works and leaves full-range context intact
      ma2_2d = panel.compute(al.MA, "close", 2)
      assert not np.all(np.isnan(ma2_2d))
      self.ma2 = al.MA(ctx.close, 2)
      self.buy_sig = ctx.close > self.ma2
      self.sell_sig = ctx.close < self.ma2

    def execute(self, ctx: Context):
      assert ctx.bar == ctx.step
      self.visited_bars.append(ctx.bar)
      ctx.buy(self.buy_sig, 1.0)
      ctx.sell(self.sell_sig, 1.0)

  prices = np.tile(np.array([10.0, 12.0, 14.0, 11.0, 11.0]), 2)
  dp = DataProvider(
    {
      "ts": np.tile(np.arange(5), 2),
      "open": prices,
      "high": prices,
      "low": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["a", "b"],
  )
  account = Account(10000.0, dp)
  runner = Runner(account)
  strat = PrecomputedStrategy()
  runner.run(strat)
  assert strat.visited_bars == [1, 2, 3, 4]
  # Bought at bar 1 (price 12.0), sold at bar 3 (price 11.0)
  assert runner.ctx.accounts[0].position.shares == 0.0


def test_runner_limit_and_next_open():
  class BuyAtBar1SellAtBar2(Strategy):
    def execute(self, ctx: Context):
      ctx.buy(ctx.ts == 1, 1.0)
      ctx.sell(ctx.ts == 2, 1.0)

  # Case 1: Limit-up at bar 1 blocks buy, limit-down at bar 2 blocks sell
  dp_limit = DataProvider(
    {
      "ts": np.arange(4),
      "open": np.array([10.0, 11.0, 9.9, 10.0]),
      "close": np.array([10.0, 11.0, 9.9, 10.0]),
      "price": np.array([10.0, 11.0, 9.9, 10.0]),
      "limited": np.array([0, 1, -1, 0]),
      "dividend": np.zeros(4),
      "transfer_shares": np.zeros(4),
      "right_shares": np.zeros(4),
      "right_price": np.zeros(4),
    },
    objs=["a"],
  )
  acc1 = Account(10000.0, dp_limit)
  runner1 = Runner(acc1, check_limit=True)
  runner1.run(BuyAtBar1SellAtBar2())
  assert runner1.ctx.accounts[0].position.shares == 0.0
  assert runner1.ctx.accounts[0].cash == 10000.0

  # Case 2: exec_mode="next_open" executes bar 1 signal at bar 2 raw_open (10.0)
  dp_open = DataProvider(
    {
      "ts": np.arange(4),
      "raw_open": np.array([10.0, 10.0, 10.0, 20.0]),
      "open": np.array([10.0, 10.0, 10.0, 20.0]),
      "close": np.array([10.0, 12.0, 15.0, 20.0]),
      "price": np.array([10.0, 12.0, 15.0, 20.0]),
      "dividend": np.zeros(4),
      "transfer_shares": np.zeros(4),
      "right_shares": np.zeros(4),
      "right_price": np.zeros(4),
    },
    objs=["a"],
  )
  acc2 = Account(10000.0, dp_open)
  runner2 = Runner(acc2, exec_mode="next_open")
  runner2.run(BuyAtBar1SellAtBar2())
  # Signal at bar 1 bought at bar 2 open=10.0 (1000 shares), signal at bar 2 sold at bar 3 open=20.0 -> 20000 cash
  assert np.isclose(runner2.ctx.accounts[0].cash, 20000.0)
  assert runner2.ctx.accounts[0].position.shares == 0.0



def test_bench_runner(benchmark):
  objs = 300
  BARS = 10000
  dp = DataProvider(
    {
      "ts": np.tile(np.arange(BARS), objs),
      "open": np.tile(np.arange(BARS, dtype=np.float64), objs),
      "high": np.tile(np.arange(BARS, dtype=np.float64), objs),
      "low": np.tile(np.arange(BARS, dtype=np.float64), objs),
      "close": np.tile(np.arange(BARS, dtype=np.float64), objs),
      "price": np.tile(np.arange(BARS, dtype=np.float64), objs),
    },
    objs=[f"a{i}" for i in range(objs)],
  )
  benchmark(run_strategy, dp)


if __name__ == "__main__":
  test_runner()
