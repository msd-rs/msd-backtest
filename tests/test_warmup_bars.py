import numpy as np
import polars as pl
from mbt import DataProvider, PortfolioAccount, PortfolioRunner, Strategy, PortfolioContext
from mbt import Account, Runner
from mbt.run.report import build_portfolio_report, build_report


def test_portfolio_warmup_bars_suppresses_trading_and_slices_report():
  # 10 bars total: first 4 bars are warmup, 6 bars are actual backtest
  prices = np.array([10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0])
  dp = DataProvider(
    {
      "ts": np.arange(10),
      "open": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["SH600000"],
  )
  dp.warmup_bars = 4

  class AggressiveStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):
      # Try to rebalance on EVERY bar
      ctx.rebalance(np.array([1.0]))

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(AggressiveStrategy())

  # Check that NO trades happened in bars 0, 1, 2, 3
  warmup_trades = [r for r in acc.records if r.bar < 4]
  assert len(warmup_trades) == 0

  # First trade happened at bar 4
  assert any(r.bar == 4 for r in acc.records)

  # Check report
  reports = build_portfolio_report(dp, acc)
  rep = reports[0]
  # Report dates must start at bar 4
  assert len(rep.date) == 6
  assert rep.date[0] == 4
  assert rep.date[-1] == 9
  # Initial cash at bar 4 was full 10000.0, so ror starts from 0 at bar 4
  assert rep.ror[0] == 0.0


def test_single_runner_warmup_bars():
  prices = np.array([10.0, 10.0, 10.0, 10.0, 12.0, 15.0])
  dp = DataProvider(
    {
      "ts": np.arange(6),
      "open": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "dividend": np.zeros(6),
      "transfer_shares": np.zeros(6),
      "right_shares": np.zeros(6),
      "right_price": np.zeros(6),
    },
    objs=["SH600000"],
  )
  dp.warmup_bars = 3

  class BuyEveryBarStrategy(Strategy):
    def execute(self, ctx):
      ctx.buy(np.array([True]), percent=1.0)

  account = Account(10000.0, dp)
  runner = Runner(account)
  runner.run(BuyEveryBarStrategy())

  # Reports must be sliced to 6 - 3 = 3 bars
  reports = build_report(dp, runner.ctx.accounts)
  rep = reports[0]
  assert len(rep.date) == 3
  assert rep.date[0] == 3
  assert rep.date[-1] == 5


def test_find_date_index_all_types():
  import datetime
  from mbt.run.msd import find_date_index

  # 1. datetime64 across multiple precisions
  for unit in ["us", "ns", "ms", "s", "D"]:
    d_dt64 = np.array(["2022-12-30", "2023-01-03", "2023-01-04"], dtype=f"datetime64[{unit}]")
    assert find_date_index(d_dt64, "2023-01-01") == 1
    assert find_date_index(d_dt64, "2022-01-01") == 0
    assert find_date_index(d_dt64, "2025-01-01") == 3

  # 2. String array
  d_str = np.array(["2022-12-30", "2023-01-03", "2023-01-04"])
  assert find_date_index(d_str, "2023-01-01") == 1

  # 3. Bytes array
  d_bytes = np.array([b"2022-12-30", b"2023-01-03", b"2023-01-04"])
  assert find_date_index(d_bytes, "2023-01-01") == 1

  # 4. Integer YYYYMMDD
  d_int = np.array([20221230, 20230103, 20230104])
  assert find_date_index(d_int, "2023-01-01") == 1

  # 5. Timestamp seconds
  ts_target = int(datetime.datetime(2023, 1, 1).timestamp())
  d_ts_s = np.array([ts_target - 86400, ts_target + 86400])
  assert find_date_index(d_ts_s, "2023-01-01") == 1

  # 6. Timestamp microseconds
  ts_us = int(datetime.datetime(2023, 1, 1).timestamp() * 1e6)
  d_ts_us = np.array([ts_us - 86400000000, ts_us + 86400000000])
  assert find_date_index(d_ts_us, "2023-01-01") == 1

  # 7. datetime.datetime objects
  d_dt = np.array([
    datetime.datetime(2022, 12, 30),
    datetime.datetime(2023, 1, 3),
    datetime.datetime(2023, 1, 4),
  ], dtype=object)
  assert find_date_index(d_dt, "2023-01-01") == 1

  # 8. datetime.date objects
  d_d = np.array([
    datetime.date(2022, 12, 30),
    datetime.date(2023, 1, 3),
    datetime.date(2023, 1, 4),
  ], dtype=object)
  assert find_date_index(d_d, "2023-01-01") == 1

  # 9. Empty array
  assert find_date_index(np.array([]), "2023-01-01") == 0


def test_warmup_bars_datetime64_fewer_bars_than_requested():
  from mbt.run.msd import find_date_index

  # Construct 76 bars in 2022 and 50 bars in 2023 with datetime64[us]
  dates_2022 = np.array([f"2022-09-{i:02d}" for i in range(1, 31)] +
                         [f"2022-10-{i:02d}" for i in range(1, 32)] +
                         [f"2022-11-{i:02d}" for i in range(1, 16)], dtype="datetime64[us]")
  assert len(dates_2022) == 76

  dates_2023 = np.array([f"2023-01-{i:02d}" for i in range(3, 20)], dtype="datetime64[us]")
  all_dates = np.concatenate([dates_2022, dates_2023])

  # User requested start="2023-01-01" with warmup_bars=120
  target_start = "2023-01-01"
  warmup_bars = 120

  idx = find_date_index(all_dates, target_start)
  computed_warmup_bars = max(0, min(idx, len(all_dates)))

  # Must be 76 (the number of bars before 2023-01-01), NOT 120!
  assert computed_warmup_bars == 76
  # The first trade date is at index 76: 2023-01-03
  assert all_dates[computed_warmup_bars] == np.datetime64("2023-01-03", "us")


def test_compute_holdout_split_datetime64():
  from mbt.run.report import Report, compute_holdout_split

  dates = np.array([
    "2024-01-02", "2024-01-03", "2024-01-04",
    "2024-06-01", "2024-06-02", "2024-06-03",
  ], dtype="datetime64[us]")

  r = Report.from_dict(
    "PORTFOLIO",
    {
      "date": dates,
      "ror": np.linspace(0.0, 0.10, len(dates)),
      "profit": 0.10,
      "benchmark_profit": 0.05,
      "trade_count": 5,
      "rebalance_count": 2,
    },
  )

  # Split by ISO date string
  split_res = compute_holdout_split(r, "2024-06-01")
  assert split_res is not None
  # Split date must be 2024-06-01
  assert split_res["split_date"] == "2024-06-01"

