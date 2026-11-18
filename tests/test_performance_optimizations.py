import numpy as np
import alpha as al
from mbt import Account, Context, DataProvider, PortfolioAccount, PortfolioRunner, Runner, Strategy
from mbt.account import Fee, FeeItem


def test_data_provider_slice_time_window_preserves_data_and_warmup():
  n_objs = 2
  bars = 10
  dates = np.arange(
    np.datetime64("2024-01-01"),
    np.datetime64("2024-01-11"),
    dtype="datetime64[D]",
  ).astype("datetime64[us]")
  p0 = np.linspace(10.0, 19.0, bars)
  p1 = np.linspace(20.0, 29.0, bars)
  prices = np.concatenate([p0, p1])
  dp = DataProvider(
    {
      "ts": np.tile(dates, n_objs),
      "open": prices.copy(),
      "high": prices + 0.5,
      "low": prices - 0.5,
      "close": prices.copy(),
      "price": prices.copy(),
    },
    objs=["SH000300", "SH600000"],
  )
  dp.warmup_bars = 2
  dp.fin_only_year = True

  sub = dp.slice_time_window(4, warmup_bars=1)
  assert sub.bars == 6
  assert sub.groups == 2
  assert sub.warmup_bars == 1
  assert sub.fin_only_year is True
  assert sub.dates[0] == dates[4]
  assert np.allclose(sub.slice("close", "SH600000"), p1[4:])
  # Transient arrays must not carry stale buffers
  assert "assets" not in sub.data


def test_runner_run_sweep_fast_and_position_aware_match_single_run():
  n_objs = 2
  bars = 40
  dates = np.arange(
    np.datetime64("2024-01-01"),
    np.datetime64("2024-01-01") + np.timedelta64(bars, "D"),
    dtype="datetime64[D]",
  ).astype("datetime64[us]")
  t = np.arange(bars, dtype=np.float64)
  bench = 100.0 + 0.1 * t
  stock = 10.0 + np.sin(t / 3.0) * 2.0 + 0.05 * t
  prices = np.concatenate([bench, stock])

  dp = DataProvider(
    {
      "ts": np.tile(dates, n_objs),
      "open": prices.copy(),
      "high": prices + 0.2,
      "low": prices - 0.2,
      "close": prices.copy(),
      "price": prices.copy(),
      "bw_price": prices.copy(),
      "fw_price": prices.copy(),
      "dividend": np.zeros(n_objs * bars),
      "transfer_shares": np.zeros(n_objs * bars),
      "right_shares": np.zeros(n_objs * bars),
      "right_price": np.zeros(n_objs * bars),
    },
    objs=["SH000300", "SH600000"],
  )
  dp.warmup_bars = 5

  class SweepFastStrat(Strategy):
    exec_mode = "next_open"

    def __init__(self, fast: int = 3, slow: int = 8):
      super().__init__()
      self.fast = fast
      self.slow = slow

    def signals(self, ctx: Context):
      f = al.MA(ctx.close, self.fast)
      s = al.MA(ctx.close, self.slow)
      return al.CROSS(f, s), al.RCROSS(f, s)

    def finalize(self, ctx: Context):
      return {"fast": self.fast, "slow": self.slow}

  grid = [{"fast": 2, "slow": 6}, {"fast": 3, "slow": 10}]
  fee = Fee([FeeItem(ts=0, commission_buy=0.0003, commission_sell=0.0003, tax_sell=0.001)])
  sweep_res = Runner.run_sweep(
    dp,
    SweepFastStrat,
    grid,
    initial_cash=100_000.0,
    fee=fee,
    exec_mode="next_open",
  )
  assert len(sweep_res) == 2

  for idx, params in enumerate(grid):
    acc = Account(100_000.0, dp, fee=fee)
    runner = Runner(acc, exec_mode="next_open")
    runner.run(SweepFastStrat(**params))
    assets = dp.slice("assets", "SH600000")[dp.warmup_bars:]
    expected_profit = float(assets[-1] / assets[0] - 1.0)
    assert np.isclose(sweep_res[idx]["mean_profit"], expected_profit, atol=1e-6)
    assert sweep_res[idx]["diagnostics"] == params

  class SweepPosAwareStrat(Strategy):
    exec_mode = "next_open"

    def __init__(self, tp: float = 0.05):
      super().__init__()
      self.tp = tp

    def execute(self, ctx: Context):
      if ctx.is_warmup:
        return
      sh = ctx.holding_shares()
      rors = ctx.holding_rors()
      ctx.sell((sh > 0.0) & (rors >= self.tp))
      ctx.buy(sh <= 0.0)

  pos_res = Runner.run_sweep(
    dp,
    SweepPosAwareStrat,
    [{"tp": 0.03}, {"tp": 0.08}],
    initial_cash=100_000.0,
    fee=fee,
    exec_mode="next_open",
  )
  assert len(pos_res) == 2
  assert pos_res[0]["total_trades"] >= pos_res[1]["total_trades"]


def test_lazy_account_and_portfolio_records_materialization():
  bars = 6
  dates = np.arange(
    np.datetime64("2024-01-01"),
    np.datetime64("2024-01-07"),
    dtype="datetime64[D]",
  ).astype("datetime64[us]")
  p = np.array([10.0, 10.0, 11.0, 12.0, 11.0, 12.0])
  dp = DataProvider(
    {
      "ts": dates,
      "open": p,
      "close": p,
      "price": p,
      "dividend": np.zeros(bars),
      "transfer_shares": np.zeros(bars),
      "right_shares": np.zeros(bars),
      "right_price": np.zeros(bars),
    },
    objs=["SH600000"],
  )

  class BuySellFast(Strategy):
    def signals(self, ctx: Context):
      b = np.array([False, True, False, False, False, False])
      s = np.array([False, False, False, True, False, False])
      return b, s

  acc = Account(100_000.0, dp)
  runner = Runner(acc)
  runner.run(BuySellFast())
  sub_acc = runner.ctx.accounts[0]
  assert sub_acc._records is None
  recs = sub_acc.records
  assert len(recs) == 2
  assert recs[0].ts == 1
  assert recs[1].ts == 3

  # PortfolioAccount lazy records & trades_df
  class PortStrat(Strategy):
    def execute(self, ctx):
      if ctx.bar == 1:
        ctx.buy(0, percent=0.5)
      elif ctx.bar == 3:
        ctx.sell(0, percent=1.0)

  p_acc = PortfolioAccount(100_000.0, dp)
  PortfolioRunner(p_acc).run(PortStrat())
  df = p_acc.trades_df()
  assert len(df) == 2
  assert len(p_acc.records) == 2
