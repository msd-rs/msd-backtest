# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for the mbt.gui package."""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from mbt.data_provider import DataProvider
from mbt.gui.config import (
  GuiConfig,
  RUN_MODE_PORTFOLIO,
  RUN_MODE_SELECT,
  RUN_MODE_TIMING,
  ShortcutConfig,
  StrategyConfig,
  get_config_dir,
  load_config,
  save_config,
)
from mbt.gui.downsample import minmax_downsample
from mbt.gui.dpi import apply_ui_scale, detect_ui_scale, scaled
from mbt.gui.inspect_strategy import inspect_strategy_file
from mbt.gui.worker import build_kv_kwargs, execute_spec, split_objs_text

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_get_config_dir_respects_env_and_xdg(monkeypatch, tmp_path):
  custom_dir = tmp_path / "custom_mbt"
  monkeypatch.setenv("MBT_CONFIG_DIR", str(custom_dir))
  assert get_config_dir() == custom_dir

  monkeypatch.delenv("MBT_CONFIG_DIR", raising=False)
  if sys.platform.startswith("linux"):
    xdg_dir = tmp_path / "xdg_cfg"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg_dir))
    assert get_config_dir() == xdg_dir / "mbt"


def test_config_roundtrip_and_crud(tmp_path):
  cfg_path = tmp_path / "mbt" / "gui.json"
  sc1 = ShortcutConfig(
    name="沪深300测试",
    mode=RUN_MODE_TIMING,
    objs="SH600000 SZ000001",
    begin="2023-01-01",
    end="2023-12-31",
    strategy_args={"channel_period": "20"},
    extra_args="warmup_bars=40",
  )
  sc2 = ShortcutConfig(
    name="全市场组合",
    mode=RUN_MODE_PORTFOLIO,
    objs="A_STOCKS_EXCLUDE_ST",
    use_internal_selector=True,
    begin="2023-01-01",
    end="2024-06-30",
    cash="1000000",
  )
  strat = StrategyConfig(
    name="cta_momentum",
    path=str(REPO_ROOT / "mbt" / "strategy_templates" / "cta_momentum.py"),
    default_mode=RUN_MODE_TIMING,
    shortcuts=[sc1, sc2],
  )
  cfg = GuiConfig(strategies=[strat], last_dir=str(REPO_ROOT))
  save_config(cfg, path=cfg_path)
  assert cfg_path.exists()

  loaded = load_config(path=cfg_path)
  assert len(loaded.strategies) == 1
  assert loaded.strategies[0].name == "cta_momentum"
  assert loaded.strategies[0].shortcuts[0].strategy_args == {"channel_period": "20"}
  assert loaded.strategies[0].shortcuts[1].mode == RUN_MODE_PORTFOLIO
  assert loaded.strategies[0].shortcuts[1].use_internal_selector is True

  # Rename & update shortcut
  loaded.strategies[0].name = "海龟趋势"
  loaded.strategies[0].shortcuts.pop(0)
  save_config(loaded, path=cfg_path)
  reloaded = load_config(path=cfg_path)
  assert reloaded.strategies[0].name == "海龟趋势"
  assert len(reloaded.strategies[0].shortcuts) == 1


def test_config_rejects_invalid_mode():
  with pytest.raises(ValueError, match="Unsupported shortcut mode"):
    ShortcutConfig.from_dict({"name": "bad", "mode": "unknown"})


def test_inspect_strategy_files():
  meta_ft = inspect_strategy_file(
    REPO_ROOT / "mbt" / "strategy_templates" / "fundamental_timing.py"
  )
  assert meta_ft.has_selector is True
  assert meta_ft.has_portfolio_strategy is True
  assert meta_ft.default_mode == RUN_MODE_PORTFOLIO
  assert RUN_MODE_SELECT in meta_ft.available_modes()
  assert any(p.name == "fast_period" for p in meta_ft.params_for_mode(RUN_MODE_PORTFOLIO))
  assert any(p.name == "min_roe" for p in meta_ft.params_for_mode(RUN_MODE_SELECT))

  meta_cta = inspect_strategy_file(
    REPO_ROOT / "mbt" / "strategy_templates" / "cta_momentum.py"
  )
  assert meta_cta.has_strategy is True
  assert meta_cta.has_portfolio_strategy is False
  assert meta_cta.default_mode == RUN_MODE_TIMING
  assert RUN_MODE_PORTFOLIO in meta_cta.available_modes()

  meta_sel = inspect_strategy_file(
    REPO_ROOT / "strategy" / "growth_large_cap_ma15_selector.py"
  )
  assert meta_sel.has_selector is True
  assert meta_sel.has_strategy is False
  assert meta_sel.default_mode == RUN_MODE_SELECT
  assert any(p.name == "large_cap_percentile" for p in meta_sel.selector_params)


def test_inspect_missing_file_raises():
  with pytest.raises(FileNotFoundError):
    inspect_strategy_file("/nonexistent/path/to/strategy.py")


def test_minmax_downsample_small_and_large():
  small = np.linspace(0.0, 1.0, 100)
  xs, ys = minmax_downsample(small, 0, 100, max_points=3000)
  assert len(xs) == 100
  assert np.array_equal(ys, small)

  # Large curve with sharp spike & deep drawdown
  n = 200_000
  large = np.sin(np.linspace(0, 50, n)) * 0.1
  large[12345] = 2.5
  large[98765] = -1.8
  large[50000] = np.nan

  xs_l, ys_l = minmax_downsample(large, 0, n, max_points=3000)
  assert 100 < len(xs_l) <= 3005
  assert np.all(np.diff(xs_l) > 0)
  assert xs_l[0] == 0
  assert xs_l[-1] == n - 1
  assert 2.5 in ys_l
  assert -1.8 in ys_l

  # Zoomed slice
  xs_z, ys_z = minmax_downsample(large, 12000, 13000, max_points=3000)
  assert len(xs_z) == 1000
  assert xs_z[0] == 12000
  assert xs_z[-1] == 12999
  assert ys_z[345] == 2.5


def test_worker_arg_helpers():
  assert split_objs_text("SH600000, SZ000001\nSH600519") == [
    "SH600000",
    "SZ000001",
    "SH600519",
  ]
  kw = build_kv_kwargs({"fast_period": "5", "purge_outdated": "false"}, "max_pe=25.5")
  assert kw == {"fast_period": 5, "purge_outdated": False, "max_pe": 25.5}


def _make_mock_dp(objs: list[str], bars: int = 40) -> DataProvider:
  dates = np.array([20230101 + i for i in range(bars)], dtype=np.int64)
  n_objs = len(objs)
  all_dates = np.tile(dates, n_objs)
  close = np.zeros(n_objs * bars, dtype=np.float64)
  for g in range(n_objs):
    trend = np.linspace(10.0, 14.0 + g, bars) + np.sin(np.arange(bars) * 0.6) * 1.2
    close[g * bars : (g + 1) * bars] = trend

  return DataProvider(
    {
      "ts": all_dates,
      "open": close.copy(),
      "high": close * 1.01,
      "low": close * 0.99,
      "close": close.copy(),
      "volume": np.full(n_objs * bars, 1_000_000.0),
      "price": close.copy(),
      "fw_price": close.copy(),
      "bw_price": close.copy(),
      "raw_close": close.copy(),
      "raw_open": close.copy(),
      "limit_ratio": np.full(n_objs * bars, 0.1),
      "dividend": np.zeros(n_objs * bars, dtype=np.float64),
      "transfer_shares": np.zeros(n_objs * bars, dtype=np.float64),
      "right_shares": np.zeros(n_objs * bars, dtype=np.float64),
      "right_price": np.zeros(n_objs * bars, dtype=np.float64),
    },
    objs=objs,
  )


def test_worker_select_timing_and_portfolio(tmp_path):
  strat_file = tmp_path / "sample_strat.py"
  strat_file.write_text(
    """
import numpy as np
import polars as pl
import mbt

class SampleSelector(mbt.Selector):
  def __init__(self, min_score: float = 0.5):
    super().__init__()
    self.min_score = float(min_score)
    self.factors = pl.DataFrame()

  def step00(self, stocks):
    picks = stocks or ["SH600000", "SZ000001"]
    self.factors = pl.DataFrame({
      "obj": picks,
      "score": [0.9 - i * 0.1 for i in range(len(picks))],
    })
    return [p for i, p in enumerate(picks) if 0.9 - i * 0.1 >= self.min_score]

class SampleTiming(mbt.Strategy):
  def __init__(self, hold_bars: int = 3):
    super().__init__()
    self.hold_bars = int(hold_bars)

  def execute(self, ctx):
    if ctx.bar == 2:
      ctx.buy(np.ones(ctx.n_objs, dtype=bool))
    elif ctx.bar == 2 + self.hold_bars:
      ctx.sell(np.ones(ctx.n_objs, dtype=bool))
""",
    encoding="utf-8",
  )

  # 1. Select mode via subprocess
  spec_sel = tmp_path / "spec_sel.json"
  out_sel = tmp_path / "out_sel.pkl"
  spec_sel.write_text(
    json.dumps({
      "strategy_path": str(strat_file),
      "mode": RUN_MODE_SELECT,
      "objs": "SH600000 SZ000001",
      "select_date": "2023-06-01",
      "strategy_args": {"min_score": "0.5"},
    }),
    encoding="utf-8",
  )
  # Patch MsdSelectorDataProvider inside execute_spec for unit test
  with patch("mbt.gui.worker.MsdSelectorDataProvider", return_value=object()):
    res_sel = execute_spec(json.loads(spec_sel.read_text()))
  assert res_sel["mode"] == RUN_MODE_SELECT
  assert res_sel["stocks"] == ["SH600000", "SZ000001"]
  assert res_sel["columns"] == ["obj", "score"]

  # 2. Timing mode
  dp_timing = _make_mock_dp(["SH000300", "SH600000", "SZ000001"], bars=20)
  with (
    patch("mbt.gui.worker.MsdSelectorDataProvider", return_value=object()),
    patch("mbt.run.cli.load_data", return_value=(dp_timing, dp_timing.objs)),
  ):
    res_timing = execute_spec({
      "strategy_path": str(strat_file),
      "mode": RUN_MODE_TIMING,
      "objs": "SH600000 SZ000001",
      "begin": "2023-01-01",
      "end": "2023-02-01",
      "benchmark": "SH000300",
      "strategy_args": {"hold_bars": "4"},
    })
  assert res_timing["mode"] == RUN_MODE_TIMING
  assert len(res_timing["reports"]) == 2
  assert res_timing["summary_curve"] is not None

  # 3. Portfolio mode
  dp_port = _make_mock_dp(["SH000300", "SH600000", "SZ000001"], bars=20)
  with (
    patch("mbt.gui.worker.MsdSelectorDataProvider", return_value=object()),
    patch("mbt.run.cli.load_data", return_value=(dp_port, dp_port.objs)),
  ):
    res_port = execute_spec({
      "strategy_path": str(strat_file),
      "mode": RUN_MODE_PORTFOLIO,
      "objs": "SH600000 SZ000001",
      "begin": "2023-01-01",
      "end": "2023-02-01",
      "benchmark": "SH000300",
      "cash": "500000",
      "strategy_args": {"hold_bars": "4"},
    })
  assert res_port["mode"] == RUN_MODE_PORTFOLIO
  assert len(res_port["rebalance_records"]) > 0
  assert len(res_port["trades"]) > 0
  assert res_port["metrics"]["initial_cash"] == 500000.0

  # 4. Portfolio mode using embedded internal Selector as stock pool (objs="")
  dp_port_sel = _make_mock_dp(["SH000300", "SH600000"], bars=20)
  with (
    patch("mbt.gui.worker.MsdSelectorDataProvider", return_value=object()),
    patch("mbt.run.cli.load_data", return_value=(dp_port_sel, dp_port_sel.objs)) as mock_load,
  ):
    res_port_sel = execute_spec({
      "strategy_path": str(strat_file),
      "mode": RUN_MODE_PORTFOLIO,
      "objs": "",
      "use_internal_selector": True,
      "begin": "2023-01-01",
      "end": "2023-02-01",
      "benchmark": "SH000300",
      "strategy_args": {"min_score": "0.85", "hold_bars": "4"},
    })
  assert res_port_sel["mode"] == RUN_MODE_PORTFOLIO
  # Embedded SampleSelector with min_score=0.85 picks only SH600000 from default ["SH600000", "SZ000001"]
  assert mock_load.call_args.args[1] == ["SH000300", "SH600000"]

  # 5. Portfolio mode without objs and without use_internal_selector still fails fast
  with pytest.raises(ValueError, match="显式指定候选股票池"):
    execute_spec({
      "strategy_path": str(strat_file),
      "mode": RUN_MODE_PORTFOLIO,
      "objs": "",
      "use_internal_selector": False,
      "begin": "2023-01-01",
      "end": "2023-02-01",
    })

  # 6. Fail-fast when use_internal_selector=True on a file without Selector
  no_sel_file = tmp_path / "no_sel.py"
  no_sel_file.write_text(
    "import mbt\nclass OnlyPort(mbt.PortfolioStrategy):\n  def execute(self, ctx): pass\n",
    encoding="utf-8",
  )
  with pytest.raises(ValueError, match="未定义 Selector 子类"):
    execute_spec({
      "strategy_path": str(no_sel_file),
      "mode": RUN_MODE_PORTFOLIO,
      "objs": "",
      "use_internal_selector": True,
      "begin": "2023-01-01",
      "end": "2023-02-01",
    })

  # 7. Fail-fast on unknown -a argument
  with pytest.raises(ValueError, match="Unrecognized argument"):
    execute_spec({
      "strategy_path": str(strat_file),
      "mode": RUN_MODE_SELECT,
      "objs": "SH600000",
      "strategy_args": {"nonexistent_param": "123"},
    })

  # 5. Subprocess CLI invocation smoke check (--help)
  proc = subprocess.run(
    [sys.executable, "-m", "mbt.gui.worker", "--help"],
    capture_output=True,
    text=True,
    check=True,
  )
  assert "--spec" in proc.stdout


def test_detect_ui_scale_env_override(monkeypatch):
  monkeypatch.setenv("MBT_GUI_SCALE", "2.0")
  assert detect_ui_scale() == 2.0


def test_gui_tk_smoke_if_display_available(monkeypatch, tmp_path):
  if not os.environ.get("DISPLAY") and sys.platform.startswith("linux"):
    pytest.skip("No X DISPLAY available in headless environment")
  monkeypatch.setenv("MBT_CONFIG_DIR", str(tmp_path / "mbt_cfg"))
  import tkinter as tk
  import tkinter.font as tkfont
  from mbt.gui.app import MbtGuiApp

  root = tk.Tk()
  root.withdraw()
  try:
    app = MbtGuiApp(root, scale=2.0)
    assert app._scale == 2.0
    assert scaled(100, root) == 200
    assert tkfont.nametofont("TkDefaultFont", root=root).metrics("linespace") >= 24

    # Exercise ShortcutDialog with internal selector metadata
    from mbt.gui.dialogs import ShortcutDialog
    meta_ft = inspect_strategy_file(
      REPO_ROOT / "mbt" / "strategy_templates" / "fundamental_timing.py"
    )
    dlg = ShortcutDialog(
      root,
      ShortcutConfig(name="测试", mode=RUN_MODE_PORTFOLIO, use_internal_selector=True),
      meta_ft,
    )
    assert dlg._use_internal_sel_var.get() is True
    dlg.destroy()

    # Exercise log drawer toggle
    app._toggle_log_drawer(force=True)
    app._append_log("hello test")
    app._toggle_log_drawer(force=False)

    # Exercise all 3 result views including 100k-point chart rendering
    app._select_view.show_result({
      "mode": RUN_MODE_SELECT,
      "strategy_name": "DemoSelector",
      "select_date": "2024-05-10",
      "stocks": ["SH600000"],
      "columns": ["obj", "roe"],
      "rows": [["SH600000", 15.2]],
      "elapsed": 0.12,
    })

    n = 100_000
    dates = [f"2024-01-{i % 28 + 1:02d}" for i in range(n)]
    ror = np.linspace(0.0, 0.45, n)
    orig = np.linspace(0.0, 0.20, n)
    bench = np.linspace(0.0, 0.15, n)

    app._timing_view.show_result({
      "mode": RUN_MODE_TIMING,
      "strategy_name": "DemoTiming",
      "elapsed": 0.5,
      "reports": [{
        "obj": "SH600000",
        "dates": dates,
        "ror": ror,
        "ror_original": orig,
        "benchmark_ror": bench,
        "benchmark_obj": "SH000300",
        "metrics": {
          "obj": "SH600000",
          "profit": 0.45,
          "profit_original": 0.20,
          "benchmark_profit": 0.15,
          "alpha": 0.12,
          "beta": 0.85,
          "sharpe": 1.6,
          "sortino": 2.1,
          "max_drawdown": -0.08,
          "calmar": 2.5,
          "win_rate": 0.6,
          "profit_factor": 1.9,
          "fee_rate": 0.002,
          "trade_count": 12,
        },
      }],
      "summary_curve": None,
    })

    app._portfolio_view.show_result({
      "mode": RUN_MODE_PORTFOLIO,
      "strategy_name": "DemoPortfolio",
      "dates": dates,
      "ror": ror,
      "ror_original": orig,
      "benchmark_ror": bench,
      "benchmark_obj": "SH000300",
      "metrics": {"profit": 0.45, "benchmark_profit": 0.15, "sharpe": 1.6, "max_drawdown": -0.08, "win_rate": 0.6},
      "rebalance_records": [{
        "bar": 1,
        "date": "2024-01-02",
        "obj": "SH600000",
        "action": "BUY",
        "shares": 1000,
        "deal_price": 10.5,
        "amount": 10500.0,
        "fee": 5.0,
        "pos_shares": 1000,
        "cash": 989495.0,
      }],
      "trades": [{
        "obj": "SH600000",
        "entry_date": "2024-01-02",
        "exit_date": "2024-02-01",
        "profit_rate": 0.1,
        "profit_amount": 1050.0,
      }],
      "attribution": [{
        "obj": "SH600000",
        "trade_count": 1,
        "win_rate": 1.0,
        "total_profit": 1050.0,
        "capital_contribution": 0.001,
        "avg_return": 0.1,
      }],
      "diagnostics": {"rebalance_count": 1},
      "elapsed": 0.4,
    })
    root.update_idletasks()
  finally:
    root.destroy()
