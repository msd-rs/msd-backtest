import numpy as np
from mbt.run.report import Report, compute_holdout_split, build_multi_start_summary, build_markdown_summary


def test_compute_holdout_split_and_markdown():
  # Create synthetic Report with 100 days
  dates = np.arange(100, dtype=np.int64)
  # First 50 days: +10% return, next 50 days: +5% return
  ror = np.concatenate([np.linspace(0.0, 0.10, 50), np.linspace(0.10, 0.15, 50)])

  r = Report.from_dict(
    "PORTFOLIO",
    {
      "date": dates,
      "ror": ror,
      "profit": 0.15,
      "benchmark_profit": 0.08,
      "trade_count": 10,
      "rebalance_count": 5,
    },
  )

  # Split by bar index / date 50
  split_res = compute_holdout_split(r, 50)
  assert split_res is not None
  assert split_res["split_date"] == "50"
  assert np.isclose(split_res["is_profit"], 0.10, atol=1e-3)
  assert np.isclose(split_res["oos_profit"], 0.05 / 1.10, atol=1e-3)
  assert split_res["is_sharpe"] > 0
  assert split_res["oos_sharpe"] > 0

  # Markdown summary should display Holdout Guard section
  md = build_markdown_summary([r], holdout=50)
  assert "## Sample Split Validation (Holdout Guard)" in md
  assert "Holdout Split Date" in md
  assert "In-Sample (IS)" in md
  assert "Out-of-Sample (OOS)" in md
  assert "Overfit Ratio" in md


def test_build_multi_start_summary():
  r1 = Report.from_dict(
    "PORTFOLIO",
    {
      "profit": 0.25,
      "benchmark_profit": 0.15,
      "max_drawdown": -0.05,
      "sharpe": 1.8,
      "win_rate": 0.60,
      "trade_count": 50,
      "rebalance_count": 10,
    },
  )
  r2 = Report.from_dict(
    "PORTFOLIO",
    {
      "profit": 0.05,
      "benchmark_profit": 0.10,
      "max_drawdown": -0.12,
      "sharpe": 0.4,
      "win_rate": 0.45,
      "trade_count": 30,
      "rebalance_count": 8,
    },
  )

  runs = [
    {"start": "2024-01-01", "end": "2025-12-31", "report": r1},
    {"start": "2025-01-01", "end": "2025-12-31", "report": r2},
  ]

  md = build_multi_start_summary(runs)
  assert "# Multi-Start Backtest Comparison Summary" in md
  assert "| Start Date | End Date | Profit | Benchmark | Beat Bench |" in md
  assert "| 2024-01-01 | 2025-12-31 | +25.00% | +15.00% | YES |" in md
  assert "| 2025-01-01 | 2025-12-31 | +5.00% | +10.00% | NO |" in md
  assert "**Evaluated Starts**: 2" in md
  assert "**Beat Benchmark**: 1/2" in md


def test_cli_parse_args_begin_options(tmp_path, monkeypatch):
  from mbt.run.cli import parse_args
  dummy_strat = tmp_path / "dummy.py"
  dummy_strat.write_text("import mbt\nclass S(mbt.Strategy):\n  pass\n")

  # 1. Multiple -b flags
  monkeypatch.setattr(
    "sys.argv",
    ["cli.py", str(dummy_strat), "-s", "SH600000", "-b", "2023-01-01", "-b", "2024-01-01"],
  )
  req1, _, _, _ = parse_args()
  assert req1.start == "2023-01-01"
  assert req1.starts == ["2023-01-01", "2024-01-01"]

  # 2. Comma-separated -b
  monkeypatch.setattr(
    "sys.argv",
    ["cli.py", str(dummy_strat), "-s", "SH600000", "-b", "2023-01-01,2024-01-01,2025-01-01"],
  )
  req2, _, _, _ = parse_args()
  assert req2.start == "2023-01-01"
  assert req2.starts == ["2023-01-01", "2024-01-01", "2025-01-01"]

  # 3. Single -b
  monkeypatch.setattr(
    "sys.argv",
    ["cli.py", str(dummy_strat), "-s", "SH600000", "-b", "2023-01-01"],
  )
  req3, _, _, _ = parse_args()
  assert req3.start == "2023-01-01"
  assert req3.starts is None

  # 4. Backward-compatible --starts
  monkeypatch.setattr(
    "sys.argv",
    ["cli.py", str(dummy_strat), "-s", "SH600000", "--starts", "2023-06-01,2024-06-01"],
  )
  req4, _, _, _ = parse_args()
  assert req4.start == "2023-06-01"
  assert req4.starts == ["2023-06-01", "2024-06-01"]

