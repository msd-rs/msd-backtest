import numpy as np
import polars as pl
from mbt.select.panel import Panel
from mbt.portfolio import _HistoricalPanelSelectorDataProvider


def test_historical_panel_selector_load_financial():
  """Verify that _HistoricalPanelSelectorDataProvider.load_financial slices
  financial records strictly up to current bar date without lookahead.
  """
  n_objs = 2
  bars = 5
  objs = ["SH600000", "SZ000001"]

  # Market dates from 2024-03-01 to 2024-05-01
  kline_dates = np.tile(
    np.array(
      [
        "2024-03-01T00:00:00.000000",
        "2024-03-15T00:00:00.000000",
        "2024-04-01T00:00:00.000000",
        "2024-04-30T00:00:00.000000",
        "2024-05-02T00:00:00.000000",
      ],
      dtype="datetime64[us]",
    ),
    n_objs,
  )

  frame = pl.DataFrame({
    "ts": kline_dates,
    "close": np.full(n_objs * bars, 10.0),
  })
  panel = Panel(frame, objs, bars)

  # Financial records:
  # 2023 Q3 disclosed by 2023-10-31
  # 2023 Annual disclosed by 2024-04-30
  financial_frame = pl.DataFrame({
    "obj": ["SH600000", "SH600000", "SZ000001", "SZ000001"],
    "report_ts": np.array(
      ["2023-09-30", "2023-12-31", "2023-09-30", "2023-12-31"],
      dtype="datetime64[us]",
    ),
    "ts": np.array(
      ["2023-10-31", "2024-04-30", "2023-10-31", "2024-04-30"],
      dtype="datetime64[us]",
    ),
    "f001": [4.32, 5.21, 3.10, 3.80],
  })

  # Slice at bar 2 (2024-04-01, before annual report disclosure on 04-30)
  sub_cols = {}
  for col in frame.columns:
    sub_cols[col] = panel.column(col)[:, :3].ravel()
  slice_bar2 = Panel(pl.DataFrame(sub_cols), objs, 3)

  dp_slice2 = _HistoricalPanelSelectorDataProvider(
    slice_bar2, financial_frame=financial_frame
  )
  fin_bar2 = dp_slice2.load_financial(objs, fields=["f001"], only_year=False, lastN=2)
  # At 2024-04-01, only 2023 Q3 is disclosed (ts=2023-10-31 <= 2024-04-01)
  # 2023 Annual report (ts=2024-04-30) must NOT be visible!
  assert len(fin_bar2) == 2
  assert all(fin_bar2["report_ts"].to_numpy() == np.datetime64("2023-09-30T00:00:00.000000", "us"))

  # Slice at bar 4 (2024-04-30, on annual report disclosure date)
  sub_cols4 = {}
  for col in frame.columns:
    sub_cols4[col] = panel.column(col)[:, :4].ravel()
  slice_bar4 = Panel(pl.DataFrame(sub_cols4), objs, 4)

  dp_slice4 = _HistoricalPanelSelectorDataProvider(
    slice_bar4, financial_frame=financial_frame
  )
  fin_bar4 = dp_slice4.load_financial(objs, fields=["f001"], only_year=True, lastN=1)
  # At 2024-04-30, annual report is now visible!
  assert len(fin_bar4) == 2
  assert all(fin_bar4["report_ts"].to_numpy() == np.datetime64("2023-12-31T00:00:00.000000", "us"))
  assert list(fin_bar4["f001"]) == [5.21, 3.80]


def test_historical_panel_selector_fail_fast_on_missing_fields():
  """Fail-fast: verify ValueError is raised when requested financial fields
  were not preloaded into backtest data.
  """
  import pytest

  objs = ["SH600000"]
  frame = pl.DataFrame({
    "ts": np.array(["2024-03-01T00:00:00.000000"], dtype="datetime64[us]"),
    "close": [10.0],
    "raw_close": [10.0],
  })
  panel = Panel(frame, objs, 1)

  # 1. No financial_frame preloaded -> load_financial raises ValueError
  dp_no_fin = _HistoricalPanelSelectorDataProvider(panel, financial_frame=None)
  with pytest.raises(ValueError, match="no financial data was preloaded in backtest"):
    dp_no_fin.load_financial(objs, fields=["f001"])

  # 2. Some financial fields missing from financial_frame -> raises ValueError
  partial_fin = pl.DataFrame({
    "obj": ["SH600000"],
    "ts": np.array(["2024-01-01T00:00:00.000000"], dtype="datetime64[us]"),
    "report_ts": np.array(["2023-12-31T00:00:00.000000"], dtype="datetime64[us]"),
    "f001": [5.0],
  })
  dp_partial = _HistoricalPanelSelectorDataProvider(panel, financial_frame=partial_fin)
  with pytest.raises(ValueError, match="requested field\\(s\\) \\['f205'\\]"):
    dp_partial.load_financial(objs, fields=["f001", "f205"])

  # 3. Missing fin_fields from panel for load_kline/load_snapshot -> raises ValueError
  with pytest.raises(ValueError, match="requested financial field\\(s\\) \\['f001'\\]"):
    dp_no_fin.load_kline(objs, fin_fields=["f001"])

  with pytest.raises(ValueError, match="requested financial field\\(s\\) \\['f001'\\]"):
    dp_no_fin.load_snapshot(objs, fin_fields=["f001"])


def test_cli_merges_selector_and_cli_fin_fields(monkeypatch, tmp_path):
  """Verify that CLI parse_args automatically merges Selector.fin_fields
  and CLI --fin-fields into req.fin_fields without manual Strategy duplication.
  """
  import sys
  from mbt.run.cli import parse_args

  # Create a strategy file with a Selector declaring fin_fields
  strat_file = tmp_path / "strat_with_selector.py"
  strat_file.write_text(
    """
from mbt.select.selector import Selector
from mbt.strategy import PortfolioStrategy

class MyTestSelector(Selector):
  fin_fields = ["f001", "f003"]
  fin_only_year = True

  def step00(self, stocks):
    return stocks

class MyTestStrategy(PortfolioStrategy):
  pass
"""
  )

  test_args = [
    "mbt.run.cli",
    str(strat_file),
    "-s", "SH600000",
    "-b", "2023-01-01",
    "-e", "2023-12-31",
    "--fin-fields", "f205,f001",
  ]
  monkeypatch.setattr(sys, "argv", test_args)

  req, msd_host, output, fee = parse_args()
  # f001, f205 from CLI + f001, f003 from Selector -> ['f205', 'f001', 'f003']
  assert req.fin_fields is not None
  assert set(req.fin_fields) == {"f001", "f003", "f205"}
  assert req.fin_only_year is True

