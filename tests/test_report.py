from mbt import Account
import numpy as np
from mbt import DataProvider
from mbt.run.report import Report



def test_report_initialization_and_metrics():
  # Set up a DataProvider with 2 objs, 5 bars each
  # obj 0: "a" (benchmark)
  # obj 1: "b"
  price = np.array([
    1.0, 1.02, 1.04, 1.03, 1.06,
    1.0, 1.05, 1.10, 1.08, 1.15,
  ])
  dp = DataProvider(
    {
      "ts": np.tile(np.arange(5), 2),
      "price": price,
      "close": price,
      "bw_price": price,
      "fw_price": price,
      "ror": np.array([0.0, 0.02, 0.04, -0.03, 0.03,    # obj "a" ror
                       0.0, 0.05, 0.10, 0.08, 0.15]),   # obj "b" ror
      "ror_hold": np.zeros(10),
      "actions": np.array([0, 1, 0, 2, 0,               # obj "a" actions
                           0, 1, 0, 2, 0]),             # obj "b" actions
      "action_shares": np.array([0.0, 100.0, 0.0, 100.0, 0.0,
                                 0.0, 200.0, 0.0, 200.0, 0.0]),
      "assets": np.array([1000.0, 990.0, 1020.0, 1015.0, 1030.0, # obj "a"
                          2000.0, 1980.0, 2080.0, 2060.0, 2120.0]),# obj "b"
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    objs=["a", "b"]
  )

  # Check initialization of obj "b"
  account = Account(1000, dp, group=1)
  report = Report("b", dp, account)
  assert report.obj == "b"
  assert np.array_equal(report.close, np.array([1.0, 1.05, 1.10, 1.08, 1.15]))
  assert np.array_equal(report.ror, np.array([0.0, 0.0, 0.0, 0.0, 0.0])) # Account had clear the ror
  assert np.array_equal(report.actions, np.array([0, 0, 0, 0, 0])) # Account had clear the actions

  # Calculate metrics
  report.calculate_metrics(dp, account)

  # Verify portfolio metrics are computed and are of correct float type
  assert isinstance(report.sharpe, float)
  assert isinstance(report.beta, float)
  assert isinstance(report.alpha, float)
  assert isinstance(report.ir, float)
  assert isinstance(report.sortino, float)
  assert isinstance(report.treynor, float)
  assert isinstance(report.max_drawdown, float)
  assert isinstance(report.calmar, float)

  # Verify trade metrics
  # For obj "b":
  # action_shares: [0, 200, 0, 200, 0]
  # actions: [0, 1 (BUY), 0, 2 (SELL), 0]
  # assets: [2000, 1980, 2080, 2060, 2120]
  # Shares held:
  # t=0: 0
  # t=1: 200 (holding starts, base_asset = assets[0] = 2000)
  # t=2: 200
  # t=3: 0 (holding ends/sell, final_asset = assets[3] = 2060)
  # t=4: 0
  # Single trade: profit = assets[3] - assets[0] = 2060 - 2000 = 60.
  # Return = 60 / 2000 = 0.03.
  # So:
  # win_rate = 1.0 (1 win, 0 losses)
  # avg_win = 0.03
  # avg_loss = 0.0
  # profit_factor = 60.0 / 0.0 = 0.0
  assert report.win_rate == 0.0
  assert np.allclose(report.avg_win, 0.0)
  assert report.avg_loss == 0.0
  assert report.profit_factor == 0.0


def test_report_edge_cases():
  # Zero/empty edge case
  dp = DataProvider(
    {
      "ts": np.array([0]),
      "close": np.array([1.0]),
      "price": np.array([1.0]),
      "bw_price": np.array([1.0]),
      "fw_price": np.array([1.0]),
      "ror": np.array([0.0]),
      "ror_hold": np.array([0.0]),
      "actions": np.array([0]),
      "action_shares": np.array([0.0]),
      "assets": np.array([1000.0]),
      "dividend": np.zeros(1),
      "transfer_shares": np.zeros(1),
      "right_shares": np.zeros(1),
      "right_price": np.zeros(1),
    },
    objs=["a"]
  )

  account = Account(1000, dp)
  report = Report("a", dp, account)
  report.calculate_metrics(dp, account)

  # Check that metrics default to 0.0 under insufficient data conditions
  assert report.sharpe == 0.0
  assert report.beta == 0.0
  assert report.alpha == 0.0
  assert report.ir == 0.0
  assert report.sortino == 0.0
  assert report.treynor == 0.0
  assert report.max_drawdown == 0.0
  assert report.calmar == 0.0
  assert report.win_rate == 0.0
  assert report.avg_win == 0.0
  assert report.avg_loss == 0.0
  assert report.profit_factor == 0.0


def test_build_markdown_summary():
  from mbt.run.report import build_markdown_summary

  r1 = Report.from_dict(
    "SH600000",
    {
      "profit": 0.25,
      "profit_original": 0.10,
      "alpha": 0.12,
      "sharpe": 1.5,
      "max_drawdown": -0.08,
      "win_rate": 0.6,
      "fee_rate": 0.005,
      "trade_count": 4,
    },
  )
  r2 = Report.from_dict(
    "SZ000001",
    {
      "profit": -0.05,
      "profit_original": 0.02,
      "alpha": -0.03,
      "sharpe": -0.4,
      "max_drawdown": -0.15,
      "win_rate": 0.3,
      "fee_rate": 0.004,
      "trade_count": 2,
    },
  )

  md = build_markdown_summary([r1, r2], top_k=2)
  assert "# Backtest Summary" in md
  assert "**Total Objs**: 2 | **Active (Traded)**: 2 (100.00%)" in md
  assert "**Positive Profit**: 1 (50.00%) | **Beat Buy&Hold**: 1 (50.00%)" in md
  assert "**Benchmark Return**: 0.0000 (0.00%) | **Beat Benchmark**: 1 (50.00%)" in md
  assert "## Metrics Statistics" in md
  assert "| profit | 0.1000 |" in md
  assert "## Top 2 Objs (by profit)" in md
  assert "## Bottom 2 Objs (by profit)" in md
  assert "| SH600000 | 0.2500 |" in md
  assert "| SZ000001 | -0.0500 |" in md


def test_portfolio_markdown_summary_with_trades():
  from mbt import PortfolioAccount, PortfolioContext, PortfolioRunner, Strategy
  from mbt.run.report import build_portfolio_report, build_markdown_summary, extract_portfolio_trades

  prices_a = np.array([10.0, 20.0, 20.0, 20.0])
  prices_b = np.array([10.0, 10.0, 10.0, 30.0])
  prices = np.concatenate([prices_a, prices_b])

  dp = DataProvider(
    {
      "ts": np.tile(np.arange(4), 2),
      "open": prices,
      "close": prices,
      "price": prices,
      "dividend": np.zeros(8),
      "transfer_shares": np.zeros(8),
      "right_shares": np.zeros(8),
      "right_price": np.zeros(8),
    },
    objs=["SH600000", "SZ000001"],
  )

  class RotateStrategy(Strategy):
    def execute(self, ctx: PortfolioContext):  # type: ignore[override]
      if ctx.dp.i == 1:
        ctx.rebalance(np.array([1.0, 0.0]))
      elif ctx.dp.i == 2:
        ctx.rebalance(np.array([0.0, 1.0]))

  acc = PortfolioAccount(10000.0, dp)
  runner = PortfolioRunner(acc)
  runner.run(RotateStrategy())

  trades = extract_portfolio_trades(account=acc)
  assert len(trades) == 2
  assert trades[0]["obj"] == "SH600000"
  assert trades[0]["entry_bar"] == 1
  assert trades[0]["exit_bar"] == 2
  assert np.isclose(trades[0]["profit_rate"], 0.0)

  assert trades[1]["obj"] == "SZ000001"
  assert trades[1]["entry_bar"] == 2
  assert trades[1]["exit_date"] == "Holding"
  assert np.isclose(trades[1]["profit_rate"], 2.0)
  assert np.isclose(trades[1]["profit_amount"], 20000.0)

  reports = build_portfolio_report(dp, acc)
  assert reports[0].trade_count == len(acc.records) == 3
  assert reports[0].rebalance_count == 2
  assert "rebalance_count" in reports[0].metrics()
  md = build_markdown_summary(reports)

  assert "## Rebalance Records" in md
  assert "| obj | Entry | Exit | Hold profit |" in md
  assert "| SH600000 | 1 | 2 | 0.00% (0.00) |" in md
  assert "| SZ000001 | 2 | Holding | +200.00% (+20,000.00) |" in md

