import time
import numpy as np
from mbt import Account, Context, DataProvider, PortfolioAccount, PortfolioRunner, Runner, Strategy
from mbt.account import Fee, FeeItem, Operation


def test_signals_fast_path_matches_execute_and_account_with_corp_actions():
  # 2 stocks x 8 bars with dividend, transfer_shares, right_shares, limit, NaN suspension, and min fee
  ts = np.tile(np.arange(8, dtype=np.int64) * 86400, 2)
  raw_open = np.array([
    10.0, 10.0, 11.0, 12.0, 6.0, 6.5, 7.0, 7.5,
    20.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0,
  ])
  price = np.array([
    10.0, 10.5, 11.5, 12.0, 6.2, np.nan, 7.2, 8.0,
    20.0, 20.5, 21.5, 22.0, 23.0, 24.0, 25.0, 26.0,
  ])
  dividend = np.array([
    0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
  ])
  transfer_shares = np.array([
    0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ])
  right_shares = np.array([
    0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ])
  right_price = np.array([
    0.0, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
  ])
  limited = np.array([
    0, 0, 0, 0, 0, 0, -1, 0,
    0, 1, 0, 0, 0, 0, 0, 0,
  ], dtype=np.int32)

  buy_sig = np.array([
    False, True, False, False, False, False, False, False,
    True, False, True, False, False, False, False, False,
  ])
  sell_sig = np.array([
    False, False, False, False, False, True, False, True,
    False, False, False, False, False, False, True, False,
  ])

  def make_dp():
    return DataProvider(
      {
        "ts": ts.copy(),
        "raw_open": raw_open.copy(),
        "open": raw_open.copy(),
        "close": price.copy(),
        "price": price.copy(),
        "dividend": dividend.copy(),
        "transfer_shares": transfer_shares.copy(),
        "right_shares": right_shares.copy(),
        "right_price": right_price.copy(),
        "limited": limited.copy(),
      },
      objs=["SH600000", "SZ000001"],
    )

  fee = Fee([FeeItem(ts=0, commission_buy=0.001, min_commission_buy=5.0, commission_sell=0.001, min_commission_sell=5.0, tax_sell=0.001)])

  class VectorizedStrat(Strategy):
    exec_mode = "next_open"
    warmup_bars = 1

    def signals(self, ctx: Context):
      return buy_sig, sell_sig

  class ExecuteStrat(Strategy):
    exec_mode = "next_open"
    warmup_bars = 1

    def execute(self, ctx: Context):
      ctx.buy(buy_sig, 1.0)
      ctx.sell(sell_sig, 1.0)

  dp1 = make_dp()
  acc1 = Account(10_000.0, dp1, fee=fee, slippage=0.001)
  r1 = Runner(acc1, exec_mode="next_open", check_limit=True)
  r1.run(VectorizedStrat())

  dp2 = make_dp()
  acc2 = Account(10_000.0, dp2, fee=fee, slippage=0.001)
  r2 = Runner(acc2, exec_mode="next_open", check_limit=True)
  r2.run(ExecuteStrat())

  assert np.allclose(dp1.all("assets"), dp2.all("assets"))
  assert np.allclose(dp1.all("ror"), dp2.all("ror"))
  assert np.array_equal(dp1.all("actions"), dp2.all("actions"))
  assert np.allclose(dp1.all("action_shares"), dp2.all("action_shares"))
  for g in range(2):
    assert np.isclose(r1.ctx.accounts[g].cash, r2.ctx.accounts[g].cash)
    assert np.isclose(r1.ctx.accounts[g].position.shares, r2.ctx.accounts[g].position.shares)
    assert np.isclose(r1.ctx.accounts[g].deducted, r2.ctx.accounts[g].deducted)
    assert len(r1.ctx.accounts[g].records) == len(r2.ctx.accounts[g].records)


def test_scale_5000x2000_fast_path_and_portfolio():
  groups = 5000
  bars = 2000
  total = groups * bars
  rng = np.random.default_rng(42)

  base_ts = np.arange(bars, dtype=np.int64) * 86400
  ts = np.tile(base_ts, groups)
  prices = np.full(total, 10.0, dtype=np.float64)
  prices += rng.uniform(-0.2, 0.2, size=total)

  # Sparse corporate actions (~0.1% of cells)
  dividend = np.zeros(total, dtype=np.float64)
  transfer_shares = np.zeros(total, dtype=np.float64)
  right_shares = np.zeros(total, dtype=np.float64)
  right_price = np.zeros(total, dtype=np.float64)
  corp_idx = rng.choice(total, size=total // 1000, replace=False)
  dividend[corp_idx[: len(corp_idx) // 2]] = 0.2
  transfer_shares[corp_idx[len(corp_idx) // 2 :]] = 0.5

  buy_sig = np.zeros(total, dtype=bool)
  sell_sig = np.zeros(total, dtype=bool)
  buy_sig[5::200] = True
  sell_sig[105::200] = True

  dp = DataProvider(
    {
      "ts": ts,
      "close": prices,
      "price": prices,
      "dividend": dividend,
      "transfer_shares": transfer_shares,
      "right_shares": right_shares,
      "right_price": right_price,
    },
    objs=[f"S{i:04d}" for i in range(groups)],
  )

  class FastStrat(Strategy):
    def signals(self, ctx: Context):
      return buy_sig, sell_sig

  t0 = time.perf_counter()
  acc = Account(100_000.0, dp)
  runner = Runner(acc)
  runner.run(FastStrat())
  elapsed = time.perf_counter() - t0

  # Sanity check that trades occurred and corporate actions ran cleanly
  assert np.any(dp.all("actions") == 1)
  assert np.any(dp.all("actions") == 2)
  # 5000 x 2000 should finish well under 5 seconds (typically < 0.5s in Rust)
  assert elapsed < 5.0, f"5000x2000 took too long: {elapsed:.3f}s"

  # Also verify PortfolioRunner on 5000 stocks x 200 bars
  p_bars = 200
  p_total = groups * p_bars
  p_dp = DataProvider(
    {
      "ts": np.tile(np.arange(p_bars, dtype=np.int64) * 86400, groups),
      "close": prices[:p_total],
      "price": prices[:p_total],
      "dividend": dividend[:p_total],
      "transfer_shares": transfer_shares[:p_total],
      "right_shares": right_shares[:p_total],
      "right_price": right_price[:p_total],
    },
    objs=[f"S{i:04d}" for i in range(groups)],
  )

  class SparseTop20Strat(Strategy):
    def execute(self, ctx):
      if ctx.bar in (1, 50, 100, 150):
        w = np.zeros(groups, dtype=np.float64)
        w[:20] = 0.05
        ctx.rebalance(w)

  t1 = time.perf_counter()
  p_acc = PortfolioAccount(1_000_000.0, p_dp)
  PortfolioRunner(p_acc).run(SparseTop20Strat())
  p_elapsed = time.perf_counter() - t1
  assert len(p_acc.held_groups) == 20
  assert p_elapsed < 3.0, f"Portfolio 5000x200 took too long: {p_elapsed:.3f}s"


def test_account_batch_execute_holding_ror_and_position_apis():
  # Verify AccountBatch-backed RunnerContext.position(), holding_ror(), holding_days()
  # handle 10-for-10 split and cash dividend accurately during bar-by-bar execute()
  ts = np.arange(5, dtype=np.int64) * 86400
  price = np.array([10.0, 10.0, 5.0, 5.5, 6.0])
  dividend = np.array([0.0, 0.0, 0.5, 0.0, 0.0])
  transfer_shares = np.array([0.0, 0.0, 1.0, 0.0, 0.0])
  zeros = np.zeros(5, dtype=np.float64)

  dp = DataProvider(
    {
      "ts": ts,
      "open": price,
      "close": price,
      "price": price,
      "dividend": dividend,
      "transfer_shares": transfer_shares,
      "right_shares": zeros,
      "right_price": zeros,
    },
    objs=["SH600000"],
  )

  rors = []
  hdays = []

  class InspectPositionStrat(Strategy):
    def execute(self, ctx: Context):
      if ctx.bar == 1:
        ctx.buy(0, 1.0)
      else:
        pos = ctx.position(0)
        rors.append((ctx.bar, ctx.holding_ror(0), pos.shares))
        hdays.append((ctx.bar, ctx.holding_days(0)))
        if ctx.holding_ror(0) >= 0.2:
          ctx.sell(0, 1.0)

  from mbt import Fee
  acc = Account(10_000.0, dp, fee=Fee.zero())
  runner = Runner(acc)
  runner.run(InspectPositionStrat())

  # Bought 1000@10.0 on bar 1.
  # Before bar 2 step, holding_days == 1; after bar 2 step (split 1->2, div=0.5),
  # before bar 3 step: 2000 shares @ 5.0 + 500 div = 10500 vs 10000 cost -> +5% at bar 3 price 5.5 -> (2000*5.5+500-10000)/10000 = +15%
  # Before bar 4 step: price 6.0 -> (2000*6.0+500-10000)/10000 = +25% >= 0.2 -> sells all!
  assert runner.ctx.accounts[0].position.shares == 0.0
  assert np.isclose(runner.ctx.accounts[0].cash, 12_500.0)
  assert hdays == [(2, 1), (3, 2), (4, 3)]
  assert np.isclose(rors[1][1], 0.15)
  assert np.isclose(rors[2][1], 0.25)


def test_double_avg_fast_matches_double_avg():
  from pathlib import Path
  from mbt.run.backend import RunRequest

  root = Path(__file__).resolve().parent.parent
  strat_slow = RunRequest.load_strategy(str(root / "strategy" / "double_avg.py"))
  strat_fast = RunRequest.load_strategy(str(root / "strategy" / "double_avg_fast.py"))

  bars = 120
  groups = 4
  rng = np.random.default_rng(7)
  ts = np.tile(np.arange(bars, dtype=np.int64) * 86400, groups)
  # Smooth oscillating price series to trigger multiple crossovers
  t = np.arange(bars, dtype=np.float64)
  close_series = [10.0 + 2.0 * np.sin(t / (4.0 + g)) + rng.normal(0.0, 0.05, bars) for g in range(groups)]
  close = np.concatenate(close_series)
  div = np.zeros(groups * bars, dtype=np.float64)
  div[45] = 0.25
  zeros = np.zeros(groups * bars, dtype=np.float64)

  def make_dp():
    return DataProvider(
      {
        "ts": ts.copy(),
        "open": close.copy(),
        "high": close.copy(),
        "low": close.copy(),
        "close": close.copy(),
        "price": close.copy(),
        "dividend": div.copy(),
        "transfer_shares": zeros.copy(),
        "right_shares": zeros.copy(),
        "right_price": zeros.copy(),
      },
      objs=[f"S{i}" for i in range(groups)],
    )

  dp_slow = make_dp()
  r_slow = Runner(Account(100_000.0, dp_slow))
  r_slow.run(strat_slow)

  dp_fast = make_dp()
  r_fast = Runner(Account(100_000.0, dp_fast))
  r_fast.run(strat_fast)

  assert np.allclose(dp_fast.all("assets"), dp_slow.all("assets"))
  assert np.allclose(dp_fast.all("ror"), dp_slow.all("ror"))
  assert np.array_equal(dp_fast.all("actions"), dp_slow.all("actions"))
  for g in range(groups):
    assert np.isclose(r_fast.ctx.accounts[g].cash, r_slow.ctx.accounts[g].cash)
    assert np.isclose(r_fast.ctx.accounts[g].position.shares, r_slow.ctx.accounts[g].position.shares)


def test_vectorized_build_report_matches_single_report_and_csv_output(tmp_path):
  import polars as pl
  from mbt.run.report import Report, build_report
  from mbt.run.cli import build_output

  bars = 60
  groups = 5
  rng = np.random.default_rng(123)
  ts = np.tile(np.arange(bars, dtype=np.int64) * 86400, groups)
  close_list = []
  for g in range(groups):
    c = 10.0 + np.cumsum(rng.normal(0.01, 0.2, bars))
    c = np.maximum(c, 1.0)
    close_list.append(c)
  close = np.concatenate(close_list)
  # Inject NaNs in one stock to verify _ffill_nan_zero_2d parity
  close[2 * bars + 0] = np.nan
  close[2 * bars + 10] = np.nan

  zeros = np.zeros(groups * bars, dtype=np.float64)
  buy_sig = np.zeros(groups * bars, dtype=bool)
  sell_sig = np.zeros(groups * bars, dtype=bool)
  buy_sig[5::20] = True
  sell_sig[15::20] = True

  dp = DataProvider(
    {
      "ts": ts,
      "open": close,
      "high": close,
      "low": close,
      "close": close,
      "price": close,
      "bw_price": close,
      "fw_price": close,
      "dividend": zeros,
      "transfer_shares": zeros,
      "right_shares": zeros,
      "right_price": zeros,
    },
    objs=[f"S{i}" for i in range(groups)],
  )
  dp.warmup_bars = 5

  class S(Strategy):
    def signals(self, ctx: Context):
      return buy_sig, sell_sig

  runner = Runner(Account(100_000.0, dp))
  runner.run(S())

  vec_reports = build_report(dp, runner.ctx.accounts)
  ref_reports = [Report(obj, dp, runner.ctx.accounts[i]) for i, obj in enumerate(dp.objs)]

  assert len(vec_reports) == len(ref_reports) == groups
  for vr, rr in zip(vec_reports, ref_reports):
    assert vr.metrics(True) == rr.metrics(True)
    assert isinstance(vr.sharpe, float)
    assert np.allclose(vr.ror, rr.ror)
    assert np.allclose(vr.ror_hold, rr.ror_hold)
    assert np.allclose(vr.ror_original, rr.ror_original)
    assert np.allclose(vr.benchmark_ror, rr.benchmark_ror)

  csv_path = tmp_path / "out.csv"
  build_output(vec_reports[1:], str(csv_path))
  df = pl.read_csv(csv_path)
  assert df.height == (groups - 1) * (bars - dp.warmup_bars)
  assert set(df["obj"].unique().to_list()) == {"S1", "S2", "S3", "S4"}


def test_scale_5227x1455_report_and_csv(tmp_path):
  from mbt.run.report import build_report
  from mbt.run.cli import build_output

  groups, bars = 5227, 1455
  total = groups * bars
  rng = np.random.default_rng(99)
  ts = np.tile(np.arange(bars, dtype=np.int64) * 86400, groups)
  prices = 10.0 + rng.uniform(-0.3, 0.3, size=total)
  zeros = np.zeros(total, dtype=np.float64)
  buy_sig = np.zeros(total, dtype=bool)
  sell_sig = np.zeros(total, dtype=bool)
  buy_sig[5::200] = True
  sell_sig[105::200] = True

  dp = DataProvider(
    {
      "ts": ts,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "fw_price": prices,
      "dividend": zeros,
      "transfer_shares": zeros,
      "right_shares": zeros,
      "right_price": zeros,
    },
    objs=[f"S{i:04d}" for i in range(groups)],
  )

  class S(Strategy):
    def signals(self, ctx: Context):
      return buy_sig, sell_sig

  runner = Runner(Account(10_000.0, dp))
  runner.run(S())

  t0 = time.perf_counter()
  reports = build_report(dp, runner.ctx.accounts)
  t_report = time.perf_counter() - t0

  out_csv = tmp_path / "f.csv"
  t1 = time.perf_counter()
  build_output(reports[1:], str(out_csv))
  t_csv = time.perf_counter() - t1

  assert len(reports) == groups
  assert out_csv.exists() and out_csv.stat().st_size > 1_000_000
  assert t_report < 2.0, f"build_report took too long: {t_report:.3f}s"
  assert t_csv < 4.0, f"build_output csv took too long: {t_csv:.3f}s"
