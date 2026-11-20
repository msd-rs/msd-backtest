# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""End-to-end CLI, report export (JSON/MD/CSV/XLSX/PNG), and demo factor tests."""

from __future__ import annotations

import json
import numpy as np
import pytest

import mbt
from mbt.data_provider import DataProvider
from mbt.factor import FactorReport, MultiFactorReport, evaluate_panel, export_report
from mbt.factor.cli import main as factor_cli_main
from mbt.factor.factor import load_factor
from mbt.select.panel import Panel


def _make_dp_with_benchmark(n_stocks: int = 15, bars: int = 120, seed: int = 11) -> tuple[DataProvider, list[str]]:
  rng = np.random.default_rng(seed)
  objs = ["SH000300"] + [f"SH60{i:04d}" for i in range(n_stocks)]
  total_objs = len(objs)
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + np.timedelta64(bars, "D"))

  rets = rng.normal(0.0003, 0.018, size=(total_objs, bars))
  close = 12.0 * np.cumprod(1.0 + rets, axis=1)
  open_ = close * (1.0 + rng.normal(0.0, 0.003, size=(total_objs, bars)))
  high = np.maximum(open_, close) * 1.01
  low = np.minimum(open_, close) * 0.99
  vol = rng.uniform(2e5, 2e6, size=(total_objs, bars))
  amount = vol * close
  data = {
    "ts": np.tile(dates, total_objs),
    "open": open_.reshape(-1),
    "high": high.reshape(-1),
    "low": low.reshape(-1),
    "close": close.reshape(-1),
    "raw_open": open_.reshape(-1),
    "raw_high": high.reshape(-1),
    "raw_low": low.reshape(-1),
    "raw_close": close.reshape(-1),
    "price": close.reshape(-1),
    "volume": vol.reshape(-1),
    "amount": amount.reshape(-1),
    "total_shares": np.full(total_objs * bars, 1e8, dtype=np.float64),
    "tradable_shares": np.full(total_objs * bars, 8e7, dtype=np.float64),
    "limited": np.zeros(total_objs * bars, dtype=np.int32),
    "f003": rng.uniform(2.0, 10.0, size=total_objs * bars),
  }
  dp = DataProvider(data, objs=objs)
  dp.warmup_bars = 20
  return dp, objs


def test_demo_factor_and_all_report_exporters(tmp_path):
  dp, _ = _make_dp_with_benchmark(n_stocks=15, bars=120)
  full_panel = dp.to_panel()
  stock_panel = full_panel.slice(0, full_panel.bars, objs=full_panel.objs[1:])
  bench_close = full_panel.column("close")[0, :]
  bench_open = full_panel.column("open")[0, :]

  fac = load_factor("strategy/demo_factor.py", {"rev_window": 10, "mom_window": 20})
  rep = evaluate_panel(
    fac,
    stock_panel,
    periods=(1, 5, 10),
    primary_period=5,
    quantiles=5,
    warmup_bars=25,
    holdout="30d",
    benchmark_name="SH000300",
    benchmark_close=bench_close,
    benchmark_open=bench_open,
  )
  assert isinstance(rep, MultiFactorReport)
  md = rep.to_markdown()
  assert "Factor Leaderboard" in md
  assert "Pairwise Cross-Sectional Rank Correlation Matrix" in md

  for ext in ("json", "md", "csv", "xlsx", "png"):
    out_file = tmp_path / f"report.{ext}"
    export_report(rep, str(out_file))
    assert out_file.is_file() and out_file.stat().st_size > 0

  # Verify PNG magic bytes and valid JSON parse
  assert (tmp_path / "report.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
  parsed = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
  assert len(parsed["leaderboard"]) == 3

  # Also test single-factor exports
  single_rep = rep.reports[0]
  assert isinstance(single_rep, FactorReport)
  export_report(single_rep, str(tmp_path / "single.csv"))
  export_report(single_rep, str(tmp_path / "single.png"))
  assert (tmp_path / "single.png").read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_factor_cli_list_ops_and_mocked_e2e(monkeypatch, tmp_path, capsys):
  # 1. --list-ops
  rc = factor_cli_main(["--list-ops"])
  assert rc == 0
  captured = capsys.readouterr().out
  assert "ts_mean" in captured and "raw_close" in captured

  # 2. Mock mbt.run.msd.load_data for CLI end-to-end runs
  dp, objs = _make_dp_with_benchmark(n_stocks=15, bars=120)
  import mbt.run.msd as run_msd_mod

  monkeypatch.setattr(run_msd_mod, "load_data", lambda *args, **kwargs: (dp, objs))

  # 2a. Python file mode with -S and --overview
  png_path = tmp_path / "cli_overview.png"
  rc = factor_cli_main([
    "strategy/demo_factor.py",
    "-s", *objs[1:],
    "-b", "2024-01-21",
    "-e", "2024-04-30",
    "-a", "rev_window=10", "mom_window=20",
    "--periods", "1,5",
    "-S",
    "--overview", str(png_path),
  ])
  assert rc == 0
  out_md = capsys.readouterr().out
  assert "rev_10d" in out_md and "value_bp" in out_md
  assert png_path.is_file() and png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"

  # 2b. Inline -E expression mode (WQ101 + valuation)
  rc = factor_cli_main([
    "-E", "bp=f003 / raw_close",
    "-E", "wq7=((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 20)) * sign(delta(close, 7))) : -1)",
    "-s", *objs[1:],
    "-b", "2024-01-21",
    "-e", "2024-04-30",
    "--periods", "1,5",
    "-S",
  ])
  assert rc == 0
  out_expr = capsys.readouterr().out
  assert "bp" in out_expr and "wq7" in out_expr

  # 2c. Batch expression .txt file mode
  batch_txt = tmp_path / "batch.txt"
  batch_txt.write_text("mom5: ts_delta(close, 5) / ts_delay(close, 5)\nbp: f003 / raw_close\n", encoding="utf-8")
  rc = factor_cli_main([
    str(batch_txt),
    "-s", *objs[1:],
    "-b", "2024-01-21",
    "-e", "2024-04-30",
    "--periods", "1,5",
    "-S",
  ])
  assert rc == 0
  assert "mom5" in capsys.readouterr().out

  # 3. Fail-Fast CLI checks
  with pytest.raises(ValueError, match="No candidate stocks provided"):
    factor_cli_main(["strategy/demo_factor.py", "-S"])

  with pytest.raises(ValueError, match="Unrecognized argument"):
    factor_cli_main(["strategy/demo_factor.py", "-s", *objs[1:], "-a", "unknown_param=1"])

  # Verify top-level mbt export
  assert hasattr(mbt, "Factor") and hasattr(mbt, "ExpressionFactor") and hasattr(mbt, "factor")
