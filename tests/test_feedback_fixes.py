from datetime import date
from unittest.mock import MagicMock
import numpy as np
import polars as pl
import pytest

import mbt
from mbt import Context, Strategy, PortfolioStrategy, Selector
from mbt.account import Account
from mbt.data_provider import DataProvider
from mbt.portfolio import (
  PortfolioAccount,
  PortfolioRunner,
  _HistoricalPanelSelectorDataProvider,
)
from mbt.run.cli import parse_args
from mbt.run.msd import load_data
from mbt.run.runner import Runner
from mbt.select import MsdSelectorDataProvider, Panel, SelectorDataProvider


def test_stock_shares_not_polluted_by_fin_only_year(monkeypatch):
  """Verify stock_shares query does NOT contain ts.month=12 even when fin_only_year=True."""
  recorded_sqls = []

  class FakeConnection:
    def __init__(self, host):
      pass

    def query_join(self, sql, primary, modes, baseline):
      recorded_sqls.append(sql)
      mock_table = MagicMock()
      mock_table.to_list.return_value = [
        ("ts", np.array([20240101])),
        ("close", np.array([10.0])),
        ("open", np.array([10.0])),
        ("high", np.array([10.0])),
        ("low", np.array([10.0])),
        ("price", np.array([10.0])),
        ("volume", np.array([1000.0])),
        ("amount", np.array([10000.0])),
      ]
      mock_table.rows = 1
      return mock_table, ["SH600000"], 1

    def query_concat(self, sql):
      return {}

    def query_map(self, sql, handler):
      pass

  monkeypatch.setattr("pymsd.MsdConnection", FakeConnection)
  monkeypatch.setattr("mbt.run.msd.get_limit_ratio", lambda objs, repeat: np.zeros(len(objs) * repeat))

  # Call load_data with fin_only_year=True and fin_fields
  load_data(
    "http://localhost:50510",
    ["SH600000"],
    start="2024-01-01",
    end="2024-12-31",
    fin_fields=["f001"],
    fin_only_year=True,
    fin_lookback_days=1500,
  )

  assert len(recorded_sqls) == 1
  sql = recorded_sqls[0]

  lines = [line.strip() for line in sql.strip().split("\n") if line.strip()]
  shares_line = next(line for line in lines if "stock_shares" in line)
  assert "ts.month=12" not in shares_line
  assert "ts >=" in shares_line


def test_context_warmup_properties():
  """Verify ctx.is_warmup and ctx.warmup_bars on RunnerContext and PortfolioContext."""
  prices = np.array([10.0, 11.0, 12.0])
  dp = DataProvider(
    {
      "ts": np.array([20240101, 20240102, 20240103]),
      "open": prices,
      "close": prices,
      "price": prices,
      "bw_price": prices,
      "dividend": np.zeros(3),
      "transfer_shares": np.zeros(3),
      "right_shares": np.zeros(3),
      "right_price": np.zeros(3),
    },
    objs=["SH600000"],
  )
  dp.warmup_bars = 2

  account = Account(10000.0, dp)
  runner = Runner(account)
  ctx = runner.ctx

  assert ctx.warmup_bars == 2
  assert ctx.is_warmup is True

  # advance bar to 2 (formal start date)
  dp.i = 2
  assert ctx.is_warmup is False

  # Portfolio Context
  p_account = PortfolioAccount(10000.0, dp)
  p_runner = PortfolioRunner(p_account)
  p_ctx = p_runner.ctx

  dp.i = 1
  assert p_ctx.warmup_bars == 2
  assert p_ctx.is_warmup is True
  dp.i = 2
  assert p_ctx.is_warmup is False


def test_exec_mode_cli_and_strategy(tmp_path, monkeypatch):
  """Verify exec_mode can be specified on Strategy or via CLI argument."""
  strat_file = tmp_path / "custom_exec_strat.py"
  strat_file.write_text(
    """
import mbt

class MyStrategy(mbt.Strategy):
  exec_mode = "next_open"
  warmup_bars = 5
  fin_lookback_days = 900

  def execute(self, ctx):
    pass
""",
    encoding="utf-8",
  )

  # Parse args without explicit --exec-mode -> inherits from Strategy
  monkeypatch.setattr(
    "sys.argv",
    ["cli.py", str(strat_file), "-s", "SH600000", "-b", "2024-01-01", "-e", "2024-06-01"],
  )
  req, _, _, _ = parse_args()
  assert req.exec_mode == "next_open"
  assert req.warmup_bars == 5
  assert req.fin_lookback_days == 900

  # Override via CLI flag --exec-mode close
  monkeypatch.setattr(
    "sys.argv",
    ["cli.py", str(strat_file), "-s", "SH600000", "-b", "2024-01-01", "--exec-mode", "close"],
  )
  req2, _, _, _ = parse_args()
  assert req2.exec_mode == "close"


def test_selector_lifecycle_and_context():
  """Verify Selector is_backtest, mode, and current_date without needing empty-stocks hacks."""
  # 1. Standalone / Initial screening
  msd_dp = MsdSelectorDataProvider("http://localhost:50510", date="2024-05-10")
  assert msd_dp.mode == "standalone"
  assert msd_dp.is_backtest is False
  assert msd_dp.current_date == "2024-05-10"

  class TrackingSelector(Selector):
    def __init__(self):
      super().__init__()
      self.observed_modes = []
      self.observed_dates = []

    def step00(self, stocks: list[str]) -> list[str]:
      self.observed_modes.append(self.is_backtest)
      self.observed_dates.append(self.current_date)
      if not self.is_backtest:
        return ["SH600000", "SZ000001"]
      return [stocks[0]]

  sel = TrackingSelector()
  out = sel.execute(msd_dp, ["SH600000"])
  assert out == ["SH600000", "SZ000001"]
  assert sel.observed_modes == [False]
  assert sel.observed_dates == ["2024-05-10"]

  # 2. Inside Backtest dynamic slice
  panel_df = pl.DataFrame({
    "ts": [20240510, 20240511],
    "close": [10.0, 11.0],
  })
  panel_slice = Panel(panel_df, ["SH600000"], 2)
  fin_frame = pl.DataFrame({"obj": ["SH600000"], "ts": [20240430], "f001": [1.5]})

  hist_dp = _HistoricalPanelSelectorDataProvider(
    panel_slice, fin_only_year=True, financial_frame=fin_frame
  )
  assert hist_dp.mode == "backtest_slice"
  assert hist_dp.is_backtest is True
  assert hist_dp.current_date == "20240511"
  assert hist_dp.financial_frame is not None

  sel_bt = TrackingSelector()
  out_bt = sel_bt.execute(hist_dp, ["SH600000", "SZ000001"])
  assert out_bt == ["SH600000"]
  assert sel_bt.observed_modes == [True]
  assert sel_bt.observed_dates == ["20240511"]


def test_load_financial_sql_date_where_only_year():
  """Verify load_financial includes ts.month = 12 when only_year=True and does not speculate fetch_n."""
  recorded_sqls = []

  class FakeClient:
    def query_concat(self, sql):
      recorded_sqls.append(sql)
      return {}

  dp = MsdSelectorDataProvider("http://localhost:50510", date="2024-05-10")
  dp._client = FakeClient()

  # Test only_year=True
  dp.load_financial(["SH600000"], fields=["f001"], only_year=True, lastN=3)
  assert len(recorded_sqls) == 1
  sql1 = recorded_sqls[0]
  assert 'ts <= "2024-05-10"' in sql1
  assert "ts.month = 12" in sql1
  assert "limit -3" in sql1
  assert "limit -7" not in sql1

  # Test only_year=False
  recorded_sqls.clear()
  dp.load_financial(["SH600000"], fields=["f001"], only_year=False, lastN=5)
  assert len(recorded_sqls) == 1
  sql2 = recorded_sqls[0]
  assert 'ts <= "2024-05-10"' in sql2
  assert "ts.month = 12" not in sql2
  assert "limit -5" in sql2

  # Test with date=None
  recorded_sqls.clear()
  dp_no_date = MsdSelectorDataProvider("http://localhost:50510", date=None)
  dp_no_date._client = FakeClient()
  dp_no_date.load_financial(["SH600000"], fields=["f001"], only_year=True, lastN=2)
  assert len(recorded_sqls) == 1
  sql3 = recorded_sqls[0]
  assert "ts.month = 12" in sql3
  assert "limit -2" in sql3

