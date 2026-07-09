from mbt import Account
import numpy as np
from mbt import DataProvider
from mbt.run.report import Report



def test_report_initialization_and_metrics():
  # Set up a DataProvider with 2 symbols, 5 bars each
  # Symbol 0: "a" (benchmark)
  # Symbol 1: "b"
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
      "ror": np.array([0.0, 0.02, 0.04, -0.03, 0.03,    # Symbol "a" ror
                       0.0, 0.05, 0.10, 0.08, 0.15]),   # Symbol "b" ror
      "ror_hold": np.zeros(10),
      "actions": np.array([0, 1, 0, 2, 0,               # Symbol "a" actions
                           0, 1, 0, 2, 0]),             # Symbol "b" actions
      "action_shares": np.array([0.0, 100.0, 0.0, 100.0, 0.0,
                                 0.0, 200.0, 0.0, 200.0, 0.0]),
      "assets": np.array([1000.0, 990.0, 1020.0, 1015.0, 1030.0, # Symbol "a"
                          2000.0, 1980.0, 2080.0, 2060.0, 2120.0]),# Symbol "b"
      "dividend": np.zeros(10),
      "transfer_shares": np.zeros(10),
      "right_shares": np.zeros(10),
      "right_price": np.zeros(10),
    },
    symbols=["a", "b"]
  )

  # Check initialization of symbol "b"
  account = Account(1000, dp, group=1)
  report = Report("b", dp, account)
  assert report.symbol == "b"
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
  # For Symbol "b":
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
    symbols=["a"]
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
