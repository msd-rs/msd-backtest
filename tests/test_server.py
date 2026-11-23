# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Comprehensive unit and integration tests for ``mbt.server``."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import socket
import sys
import threading
import time
from unittest.mock import patch
import zipfile

from fastapi.testclient import TestClient
import numpy as np
import polars as pl
import pytest

from mbt.data_provider import DataProvider
from mbt.server import get_version
from mbt.server.__main__ import check_existing_server
from mbt.server.app import create_app
from mbt.server.config import (
  ServerConfig,
  SystemDefaults,
  get_config_dir,
  get_data_dir,
  load_server_config,
)
from mbt.server.storage import export_run_data, query_curve, query_table, read_manifest
from mbt.server.worker import execute_and_store
from mbt.shared.inspect_strategy import (
  RUN_MODE_FACTOR,
  RUN_MODE_PORTFOLIO,
  RUN_MODE_SELECT,
  RUN_MODE_TIMING,
  inspect_strategy_file,
)


def _make_mock_dp(objs: list[str], bars: int = 40) -> DataProvider:
  dates = np.array([20230101 + i for i in range(bars)], dtype=np.int64)
  n_objs = len(objs)
  all_dates = np.tile(dates, n_objs)
  close = np.zeros(n_objs * bars, dtype=np.float64)
  total_shares = np.zeros(n_objs * bars, dtype=np.float64)
  for g in range(n_objs):
    trend = np.linspace(10.0, 14.0 + g, bars) + np.sin(np.arange(bars) * 0.5) * 0.9
    close[g * bars : (g + 1) * bars] = trend
    total_shares[g * bars : (g + 1) * bars] = 1e8 * (1 + g)

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
      "total_shares": total_shares,
      "limited": np.zeros(n_objs * bars, dtype=np.float64),
      "limit_ratio": np.full(n_objs * bars, 0.1),
      "dividend": np.zeros(n_objs * bars, dtype=np.float64),
      "transfer_shares": np.zeros(n_objs * bars, dtype=np.float64),
      "right_shares": np.zeros(n_objs * bars, dtype=np.float64),
      "right_price": np.zeros(n_objs * bars, dtype=np.float64),
    },
    objs=objs,
  )


def _write_sample_strategy(path: Path) -> Path:
  path.write_text(
    """
import numpy as np
import polars as pl
import mbt

M_AUTHOR = "TestAgent"

class DemoSelector(mbt.Selector):
  def __init__(self, min_score: float = 0.5):
    super().__init__()
    self.min_score = float(min_score)
    self.factors = pl.DataFrame()

  def step00(self, stocks):
    picks = stocks or ["SH600000", "SZ000001"]
    self.factors = pl.DataFrame({
      "obj": picks,
      "score": [0.9 - i * 0.1 for i in range(len(picks))],
      "note": [f"item[0]_{p}" for p in picks],
    })
    return [p for i, p in enumerate(picks) if 0.9 - i * 0.1 >= self.min_score]

class DemoStrategy(mbt.Strategy):
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
  return path


def _write_sample_factor(path: Path) -> Path:
  path.write_text(
    """
import numpy as np
import mbt

M_AUTHOR = "FactorAgent"

class MomentumMixFactor(mbt.Factor):
  name = "mom_mix"
  direction = 1
  warmup_bars = 2

  def __init__(self, lookback: int = 3):
    self.lookback = int(lookback)

  def compute(self, panel):
    close = panel.column("close")
    ret = np.zeros_like(close)
    if close.shape[1] > self.lookback:
      ret[:, self.lookback :] = close[:, self.lookback :] / close[:, :-self.lookback] - 1.0
    return {
      "mom_a": ret,
      "mom_b": -ret * 0.5 + 0.01 * np.arange(close.shape[0])[:, None],
    }
""",
    encoding="utf-8",
  )
  return path


def test_os_dirs_and_env_overrides(monkeypatch, tmp_path):
  cfg_override = tmp_path / "cfg_override"
  data_override = tmp_path / "data_override"
  monkeypatch.setenv("MBT_SERVER_CONFIG_DIR", str(cfg_override))
  monkeypatch.setenv("MBT_SERVER_DATA_DIR", str(data_override))
  assert get_config_dir() == cfg_override
  assert get_data_dir() == data_override

  monkeypatch.delenv("MBT_SERVER_CONFIG_DIR", raising=False)
  monkeypatch.delenv("MBT_SERVER_DATA_DIR", raising=False)
  if sys.platform.startswith("linux"):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg_cfg"))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg_data"))
    assert get_config_dir() == tmp_path / "xdg_cfg" / "mbt"
    assert get_data_dir() == tmp_path / "xdg_data" / "mbt" / "server"


def test_check_existing_server_states():
  # Free port
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
  assert check_existing_server("127.0.0.1", free_port, timeout=0.3) is None

  # Healthy mbt server mock
  class _OkHandler(BaseHTTPRequestHandler):
    def do_GET(self):
      self.send_response(200)
      self.end_headers()
      self.wfile.write(f"OK, msd-backtest {get_version()}".encode("utf-8"))

    def log_message(self, *_args):
      pass

  srv_ok = ThreadingHTTPServer(("127.0.0.1", 0), _OkHandler)
  t_ok = threading.Thread(target=srv_ok.serve_forever, daemon=True)
  t_ok.start()
  try:
    assert check_existing_server("0.0.0.0", srv_ok.server_port, timeout=1.0) == 0
  finally:
    srv_ok.shutdown()
    srv_ok.server_close()

  # Unrelated HTTP service on port
  class _OtherHandler(BaseHTTPRequestHandler):
    def do_GET(self):
      self.send_response(200)
      self.end_headers()
      self.wfile.write(b"NOT_MBT_SERVICE")

    def log_message(self, *_args):
      pass

  srv_other = ThreadingHTTPServer(("127.0.0.1", 0), _OtherHandler)
  t_other = threading.Thread(target=srv_other.serve_forever, daemon=True)
  t_other.start()
  try:
    assert check_existing_server("127.0.0.1", srv_other.server_port, timeout=1.0) == 1
  finally:
    srv_other.shutdown()
    srv_other.server_close()


def test_inspect_factor_py_and_txt(tmp_path):
  f_py = _write_sample_factor(tmp_path / "my_factor.py")
  meta_py = inspect_strategy_file(f_py)
  assert meta_py.has_factor is True
  assert meta_py.default_mode == RUN_MODE_FACTOR
  assert meta_py.factor_name == "MomentumMixFactor"
  assert meta_py.author == "FactorAgent"
  assert [p.name for p in meta_py.factor_params] == ["lookback"]

  f_txt = tmp_path / "exprs.txt"
  f_txt.write_text("alpha1: close / open - 1\nalpha2 = volume / mean(volume, 5)\n", encoding="utf-8")
  meta_txt = inspect_strategy_file(f_txt)
  assert meta_txt.has_factor is True
  assert meta_txt.is_expression_file is True
  assert meta_txt.expression_names == ["alpha1", "alpha2"]
  assert meta_txt.default_mode == RUN_MODE_FACTOR


def test_health_openapi_config_and_crud(tmp_path):
  cfg_dir = tmp_path / "cfg"
  data_dir = tmp_path / "data"
  app = create_app(config_dir=cfg_dir, data_dir=data_dir)
  strat_file = _write_sample_strategy(tmp_path / "demo_strat.py")
  symlink_path = tmp_path / "demo_strat_link.py"
  os.symlink(strat_file, symlink_path)

  with TestClient(app) as client:
    # Health & OpenAPI
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.text == f"OK, msd-backtest {get_version()}"

    r_openapi = client.get("/api/openapi.json")
    assert r_openapi.status_code == 200
    spec = r_openapi.json()
    assert "/api/health" in spec["paths"]
    assert "/api/strategies" in spec["paths"]
    assert all(p.startswith("/api/") for p in spec["paths"])

    # System config GET / PUT
    r_cfg = client.get("/api/config")
    assert r_cfg.status_code == 200
    cfg_payload = r_cfg.json()
    assert cfg_payload["defaults"]["cash"] == 10000000.0
    assert (cfg_dir / "server.json").is_file()

    new_defaults = dict(cfg_payload["defaults"])
    new_defaults["begin"] = "2022-02-02"
    new_defaults["cash"] = 2000000.0
    r_put = client.put("/api/config", json={"defaults": new_defaults})
    assert r_put.status_code == 200
    assert r_put.json()["defaults"]["begin"] == "2022-02-02"
    assert load_server_config(cfg_dir).defaults.begin == "2022-02-02"

    # Strategy register & idempotent dedup via symlink
    r_add = client.post("/api/strategies", json={"path": str(strat_file), "notes": "v1"})
    assert r_add.status_code == 200
    s1 = r_add.json()
    assert s1["created"] is True
    assert s1["default_mode"] == RUN_MODE_TIMING
    sid = s1["id"]

    r_dup = client.post("/api/strategies", json={"path": str(symlink_path)})
    assert r_dup.status_code == 200
    s2 = r_dup.json()
    assert s2["created"] is False
    assert s2["id"] == sid

    # Missing file fails fast
    r_missing = client.post("/api/strategies", json={"path": str(tmp_path / "nope.py")})
    assert r_missing.status_code == 404

    # Shortcut CRUD
    r_sc = client.post(
      f"/api/strategies/{sid}/shortcuts",
      json={
        "name": "沪深测试",
        "mode": "portfolio",
        "params": {"objs": "SH600000 SZ000001", "strategy_args": {"hold_bars": "4"}},
      },
    )
    assert r_sc.status_code == 201
    sc1 = r_sc.json()
    sc_id = sc1["id"]

    r_sc_upd = client.put(
      f"/api/shortcuts/{sc_id}",
      json={"name": "沪深测试-改", "params": {"objs": "SH600000", "cash": "500000"}},
    )
    assert r_sc_upd.status_code == 200
    assert r_sc_upd.json()["name"] == "沪深测试-改"

    r_sc_dup = client.post(f"/api/shortcuts/{sc_id}/duplicate", json={})
    assert r_sc_dup.status_code == 201
    sc2_id = r_sc_dup.json()["id"]
    assert sc2_id != sc_id

    # Verify RunManager inherits SystemDefaults when shortcut leaves begin empty
    rm = app.state.run_manager
    resolved = rm.resolve_spec(
      SystemDefaults(begin="2022-02-02", cash=2000000.0),
      s2,
      shortcut=r_sc_upd.json(),
    )
    assert resolved["begin"] == "2022-02-02"
    assert resolved["cash"] == "500000"

    r_del_sc = client.delete(f"/api/shortcuts/{sc2_id}")
    assert r_del_sc.status_code == 200
    assert len(client.get(f"/api/strategies/{sid}/shortcuts").json()) == 1

    # Unknown API route returns JSON 404; root serves HTML
    assert client.get("/api/does-not-exist").status_code == 404
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "text/html" in r_root.headers["content-type"]

    assert client.delete(f"/api/strategies/{sid}").status_code == 200
    assert client.get("/api/strategies").json() == []


def test_worker_parquet_storage_and_queries_all_modes(tmp_path):
  strat_file = _write_sample_strategy(tmp_path / "strat.py")
  factor_file = _write_sample_factor(tmp_path / "factor.py")

  # 1. Select mode
  sel_dir = tmp_path / "run_select"
  with patch("mbt.server.worker.MsdSelectorDataProvider", return_value=object()):
    m_sel = execute_and_store(
      {
        "strategy_path": str(strat_file),
        "mode": RUN_MODE_SELECT,
        "objs": "SH600000 SZ000001",
        "select_date": "2023-06-01",
        "strategy_args": {"min_score": "0.5"},
      },
      sel_dir,
    )
  assert m_sel["mode"] == RUN_MODE_SELECT
  assert (sel_dir / "select_factors.parquet").is_file()
  # Substring search with regex special chars `[0]` must work safely via literal=True
  tbl_sel = query_table(sel_dir, "select_factors", q="[0]_SH600000")
  assert tbl_sel["total_rows"] == 1
  assert tbl_sel["rows"][0]["obj"] == "SH600000"

  # 2. Timing mode
  tim_dir = tmp_path / "run_timing"
  dp_tim = _make_mock_dp(["SH000300", "SH600000", "SZ000001"], bars=30)
  with (
    patch("mbt.server.worker.MsdSelectorDataProvider", return_value=object()),
    patch("mbt.run.cli.load_data", return_value=(dp_tim, dp_tim.objs)),
  ):
    m_tim = execute_and_store(
      {
        "strategy_path": str(strat_file),
        "mode": RUN_MODE_TIMING,
        "objs": "SH600000 SZ000001",
        "begin": "2023-01-01",
        "end": "2023-02-15",
        "benchmark": "SH000300",
        "holdout": "2023-01-15",
        "strategy_args": {"hold_bars": "4"},
      },
      tim_dir,
    )
  assert m_tim["summary"]["has_summary_curve"] is True
  assert m_tim["summary"]["holdout"] is not None
  tbl_tim = query_table(tim_dir, "metrics", page=1, page_size=1, sort_by="obj", sort_dir="asc")
  assert tbl_tim["total_rows"] == 2
  assert tbl_tim["total_pages"] == 2
  assert len(tbl_tim["rows"]) == 1

  c_sum = query_curve(tim_dir, obj="__SUMMARY__", max_points=10)
  assert c_sum["total_points"] == 30
  assert c_sum["downsampled"] is True
  assert len(c_sum["dates"]) <= 14
  c_stock = query_curve(tim_dir, obj="SH600000", start_idx=5, end_idx=20, max_points=100)
  assert c_stock["indices"][0] == 5
  assert c_stock["indices"][-1] == 19

  # 3. Portfolio mode + exports
  port_dir = tmp_path / "run_portfolio"
  dp_port = _make_mock_dp(["SH000300", "SH600000", "SZ000001"], bars=30)
  with (
    patch("mbt.server.worker.MsdSelectorDataProvider", return_value=object()),
    patch("mbt.run.cli.load_data", return_value=(dp_port, dp_port.objs)),
  ):
    m_port = execute_and_store(
      {
        "strategy_path": str(strat_file),
        "mode": RUN_MODE_PORTFOLIO,
        "objs": "SH600000 SZ000001",
        "begin": "2023-01-01",
        "end": "2023-02-15",
        "benchmark": "SH000300",
        "cash": 500000.0,
        "strategy_args": {"hold_bars": "4"},
      },
      port_dir,
    )
  assert m_port["summary"]["rebalance_records_count"] > 0
  assert query_table(port_dir, "rebalances")["total_rows"] > 0
  assert query_table(port_dir, "trades")["total_rows"] > 0
  assert query_table(port_dir, "attribution")["total_rows"] > 0
  c_port = query_curve(port_dir)
  assert len(c_port["series"]["ror"]) == 30

  csv_bytes, _, _ = export_run_data(port_dir, "csv", table="trades")
  assert b"entry_date" in csv_bytes
  xlsx_bytes, _, _ = export_run_data(port_dir, "xlsx")
  assert xlsx_bytes[:2] == b"PK"
  json_bytes, _, _ = export_run_data(port_dir, "json")
  assert "manifest" in json.loads(json_bytes.decode("utf-8"))
  zip_bytes, _, _ = export_run_data(port_dir, "parquet")
  with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
    assert "rebalances.parquet" in zf.namelist()
    assert "manifest.json" in zf.namelist()

  # 4. Factor mode (multi-factor)
  fac_dir = tmp_path / "run_factor"
  fac_objs = ["SH000300"] + [f"SH6000{i:02d}" for i in range(15)]
  dp_fac = _make_mock_dp(fac_objs, bars=50)
  with patch("mbt.run.msd.load_data", return_value=(dp_fac, dp_fac.objs)):
    m_fac = execute_and_store(
      {
        "strategy_path": str(factor_file),
        "mode": RUN_MODE_FACTOR,
        "objs": " ".join(fac_objs[1:]),
        "begin": "2023-01-01",
        "end": "2023-04-01",
        "benchmark": "SH000300",
        "factor_periods": "1,5",
        "factor_primary_period": "5",
        "factor_quantiles": "3",
        "factor_min_obs": "5",
        "holdout": "2023-01-25",
        "strategy_args": {"lookback": "3"},
      },
      fac_dir,
    )
  assert m_fac["mode"] == RUN_MODE_FACTOR
  assert m_fac["summary"]["is_multi_factor"] is True
  assert set(m_fac["summary"]["factor_names"]) == {"mom_a", "mom_b"}
  assert read_manifest(fac_dir)["tables"]["leaderboard"]["rows"] == 2
  q_tbl = query_table(fac_dir, "quantiles", factor="mom_a", period=5)
  assert q_tbl["total_rows"] >= 5
  c_fac = query_curve(fac_dir, factor="mom_a", period=5, max_points=20)
  assert "Q1_nav" in c_fac["series"]
  assert "cum_rank_ic" in c_fac["series"]


def test_async_subprocess_latest_only_and_cancel(tmp_path):
  cfg_dir = tmp_path / "cfg"
  data_dir = tmp_path / "data"
  app = create_app(config_dir=cfg_dir, data_dir=data_dir)

  fast_sel = tmp_path / "fast_sel.py"
  fast_sel.write_text(
    """
import polars as pl
import mbt

class FastSel(mbt.Selector):
  def __init__(self, tag: str = "v1"):
    super().__init__()
    self.tag = tag
    self.factors = pl.DataFrame()

  def step00(self, stocks):
    picks = stocks or ["SH600000", "SZ000001"]
    self.factors = pl.DataFrame({"obj": picks, "tag": [self.tag] * len(picks)})
    return picks
""",
    encoding="utf-8",
  )

  # Slow strategy for cancellation test (sleeps in execute before finishing)
  slow_sel = tmp_path / "slow_sel.py"
  slow_sel.write_text(
    """
import time
import mbt

class SlowSel(mbt.Selector):
  def step00(self, stocks):
    print("SLOW_STEP_STARTED", flush=True)
    time.sleep(20)
    return stocks or ["SH600000"]
""",
    encoding="utf-8",
  )

  # Note: MsdSelectorDataProvider only initializes URL string in __init__ (no HTTP call until query)
  with TestClient(app) as client:
    s_fast = client.post("/api/strategies", json={"path": str(fast_sel)}).json()
    sc = client.post(
      f"/api/strategies/{s_fast['id']}/shortcuts",
      json={
        "name": "快速选股",
        "mode": "select",
        "params": {"objs": "SH600000 SZ000001", "strategy_args": {"tag": "r1"}},
      },
    ).json()

    # Run #1
    r1 = client.post("/api/runs", json={"shortcut_id": sc["id"]})
    assert r1.status_code == 202
    run1_id = r1.json()["id"]
    run1_dir = Path(app.state.store.get_run(run1_id)["result_dir"])

    deadline = time.time() + 15.0
    while time.time() < deadline:
      cur = client.get(f"/api/runs/{run1_id}").json()
      if cur["status"] != "running":
        break
      time.sleep(0.1)
    assert cur["status"] == "completed"
    assert run1_dir.is_dir()

    # Verify logs & query endpoints over HTTP
    logs_resp = client.get(f"/api/runs/{run1_id}/logs")
    assert logs_resp.status_code == 200
    assert "FastSel" in logs_resp.text

    tbl_resp = client.get(f"/api/runs/{run1_id}/tables/select_factors")
    assert tbl_resp.status_code == 200
    assert tbl_resp.json()["total_rows"] == 2

    exp_resp = client.get(f"/api/runs/{run1_id}/export?format=csv")
    assert exp_resp.status_code == 200
    assert "SH600000" in exp_resp.text

    # Run #2 supersedes Run #1 and removes run1_dir from disk
    r2 = client.post(
      "/api/runs",
      json={"shortcut_id": sc["id"], "overrides": {"strategy_args": {"tag": "r2"}}},
    )
    assert r2.status_code == 202
    run2_id = r2.json()["id"]
    deadline = time.time() + 15.0
    while time.time() < deadline:
      cur2 = client.get(f"/api/runs/{run2_id}").json()
      if cur2["status"] != "running":
        break
      time.sleep(0.1)
    assert cur2["status"] == "completed"
    assert not run1_dir.exists()
    assert client.get(f"/api/runs/{run1_id}").status_code == 404

    # Slow run: test conflict 409 and cancel
    s_slow = client.post("/api/strategies", json={"path": str(slow_sel)}).json()
    r_slow = client.post(
      "/api/runs",
      json={"strategy_id": s_slow["id"], "overrides": {"objs": "SH600000"}},
    )
    assert r_slow.status_code == 202
    slow_id = r_slow.json()["id"]

    # Duplicate active run returns 409
    r_dup = client.post(
      "/api/runs",
      json={"strategy_id": s_slow["id"], "overrides": {"objs": "SH600000"}},
    )
    assert r_dup.status_code == 409

    # Cancel active run
    r_cancel = client.post(f"/api/runs/{slow_id}/cancel")
    assert r_cancel.status_code == 200
    deadline = time.time() + 10.0
    while time.time() < deadline:
      st_slow = client.get(f"/api/runs/{slow_id}").json()
      if st_slow["status"] != "running":
        break
      time.sleep(0.1)
    assert st_slow["status"] == "cancelled"


def test_cli_register_and_status(tmp_path, capsys):
  from mbt.server.__main__ import main as server_main

  strat_file = _write_sample_strategy(tmp_path / "cli_reg_strat.py")
  received: list[dict] = []

  class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      if self.path == "/api/health":
        self.send_response(200)
        self.end_headers()
        self.wfile.write(f"OK, msd-backtest {get_version()}".encode("utf-8"))
        return
      self.send_response(404)
      self.end_headers()

    def do_POST(self):
      if self.path == "/api/strategies":
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        received.append(payload)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
          json.dumps({"id": "abc123", "created": True, "path": payload["path"]}).encode("utf-8")
        )
        return
      self.send_response(404)
      self.end_headers()

    def log_message(self, *_args):
      pass

  srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
  t = threading.Thread(target=srv.serve_forever, daemon=True)
  t.start()
  try:
    url = f"http://127.0.0.1:{srv.server_port}"
    assert server_main(["status", "--url", url]) == 0
    out_status = capsys.readouterr().out
    assert "OK, msd-backtest" in out_status

    assert server_main(["register", str(strat_file), "--name", "cli_demo", "--url", url]) == 0
    out_reg = capsys.readouterr().out.strip()
    assert out_reg == f"{url}/strategies/abc123"
    assert len(received) == 1
    assert received[0]["path"] == os.path.realpath(str(strat_file))
    assert received[0]["name"] == "cli_demo"
  finally:
    srv.shutdown()
    srv.server_close()


def test_strategy_default_name_uses_docstring(tmp_path):
  cfg_dir = tmp_path / "cfg"
  data_dir = tmp_path / "data"
  app = create_app(config_dir=cfg_dir, data_dir=data_dir)

  doc_strat = tmp_path / "unnamed_strategy.py"
  doc_strat.write_text(
    '"""多因子价值成长轮动策略"""\n\nimport mbt\n\nclass DocStrat(mbt.PortfolioStrategy):\n  def execute(self, ctx): pass\n',
    encoding="utf-8",
  )
  no_doc_strat = _write_sample_strategy(tmp_path / "fallback_stem.py")

  with TestClient(app) as client:
    r1 = client.post("/api/strategies", json={"path": str(doc_strat)})
    assert r1.status_code == 200
    assert r1.json()["name"] == "多因子价值成长轮动策略"

    r2 = client.post("/api/strategies", json={"path": str(no_doc_strat)})
    assert r2.status_code == 200
    assert r2.json()["name"] == "fallback_stem"

    # Explicit name still takes precedence
    doc_strat2 = tmp_path / "explicit_name.py"
    doc_strat2.write_text(doc_strat.read_text(encoding="utf-8"), encoding="utf-8")
    r3 = client.post("/api/strategies", json={"path": str(doc_strat2), "name": "自定义标题"})
    assert r3.status_code == 200
    assert r3.json()["name"] == "自定义标题"


def test_delta_fixed_int_parquet_encoding_and_compression(tmp_path):
  from mbt.server.storage import _decode_curve_lf, _encode_delta_int_1d, _encode_fixed_int_1d, write_timing_result
  from mbt.run.report import Report

  # 1. Roundtrip accuracy with NaNs & sliced start_idx > 0
  raw_delta = np.array([0.01234, np.nan, 0.05678, -0.02349, 0.11112], dtype=np.float64)
  raw_fixed = np.array([np.nan, 0.03216, -0.01454, 0.0, 0.08884], dtype=np.float64)
  lf = pl.DataFrame({
    "bar_idx": np.arange(5, dtype=np.int32),
    "d": _encode_delta_int_1d(raw_delta),
    "f": _encode_fixed_int_1d(raw_fixed),
  }).lazy()
  enc = {"kind": "delta_fixed_int", "scale": 10000, "delta_columns": ["d"], "fixed_columns": ["f"]}

  full_df = _decode_curve_lf(lf.sort("bar_idx"), enc).collect()
  assert full_df["d"][1] is None
  assert full_df["f"][0] is None
  assert np.isclose(full_df["d"][0], 0.0123, atol=1e-6)
  assert np.isclose(full_df["d"][2], 0.0568, atol=1e-6)
  assert np.isclose(full_df["d"][3], -0.0235, atol=1e-6)
  assert np.isclose(full_df["d"][4], 0.1111, atol=1e-6)
  assert np.isclose(full_df["f"][1], 0.0322, atol=1e-6)
  assert np.isclose(full_df["f"][2], -0.0145, atol=1e-6)

  # Slicing from bar 2 must still reconstruct from bar 0 via cum_sum()
  sliced_df = _decode_curve_lf(lf.sort("bar_idx"), enc).slice(2, 3).collect()
  assert np.allclose(sliced_df["d"].to_numpy(), [0.0568, -0.0235, 0.1111], atol=1e-6)

  # 2. Verify schema uses integer types and achieves major size savings vs Float64 zstd
  rng = np.random.default_rng(7)
  n_objs, bars = 120, 250
  reports: list[Report] = []
  dates = np.array([20230101 + i for i in range(bars)], dtype=np.int64)
  raw_frames = []
  for g in range(n_objs):
    obj = f"SH{600000 + g:06d}"
    daily = rng.normal(0.0002, 0.015, bars)
    ror = np.round(np.cumprod(1.0 + daily) - 1.0, 4)
    orig = np.round(np.cumprod(1.0 + rng.normal(0.0001, 0.015, bars)) - 1.0, 4)
    bench = np.round(np.linspace(0.0, 0.1, bars), 4)
    px = np.round(10.0 * (1.0 + ror), 3)

    rep = Report.__new__(Report)
    rep.obj = obj
    rep.date = dates
    rep.ror = ror
    rep.ror_hold = ror
    rep.ror_original = orig
    rep.benchmark_ror = bench
    rep.benchmark_obj = "SH000300"
    rep.actions = np.zeros(bars, dtype=np.int32)
    rep.price = px
    rep.close = px
    rep.fw_price = px
    rep.bw_price = px
    rep.alpha = rep.beta = rep.sharpe = rep.information_ratio = rep.ir = 0.0
    rep.sortino = rep.treynor = rep.max_drawdown = rep.calmar = 0.0
    rep.win_rate = rep.avg_win = rep.avg_loss = rep.profit_factor = 0.0
    rep.profit = float(ror[-1])
    rep.profit_original = float(orig[-1])
    rep.benchmark_profit = float(bench[-1])
    rep.fee_rate = 0.0
    rep.trade_count = 0
    rep.rebalance_count = 0
    reports.append(rep)

    raw_frames.append(
      pl.DataFrame({
        "obj": np.repeat(obj, bars),
        "bar_idx": np.arange(bars, dtype=np.int32),
        "date": [str(d) for d in dates],
        "ror": ror,
        "ror_original": orig,
        "benchmark_ror": bench,
        "drawdown": ror,
        "actions": np.zeros(bars, dtype=np.int32),
        "price": px,
      })
    )

  out_dir = tmp_path / "encoded_timing"
  write_timing_result(
    out_dir,
    strategy_name="SizeBench",
    reports=reports,
    benchmark_obj="SH000300",
    holdout=None,
    diagnostics=None,
    elapsed=0.1,
  )
  encoded_schema = pl.scan_parquet(out_dir / "curves.parquet").collect_schema()
  assert encoded_schema["ror"] in (pl.Int32, pl.Int64)
  assert encoded_schema["actions"] == pl.Int8

  raw_pq = tmp_path / "raw_float64.parquet"
  pl.concat(raw_frames).write_parquet(raw_pq, compression="zstd")
  encoded_size = (out_dir / "curves.parquet").stat().st_size
  raw_size = raw_pq.stat().st_size
  assert encoded_size < raw_size * 0.55

  # Verify query_curve reads back accurate values across slice boundary
  q_res = query_curve(out_dir, obj="SH600005", start_idx=50, end_idx=120, max_points=500)
  assert np.allclose(q_res["series"]["ror"], reports[5].ror[50:120], atol=1e-4)
  assert np.allclose(q_res["series"]["price"], reports[5].price[50:120], atol=1e-3)

  with pytest.raises(ValueError, match="Unsupported curve encoding"):
    _decode_curve_lf(lf, {"kind": "unknown_v99"})

  # Verify summary_curve.parquet decodes accurately (full and sliced)
  mean_ror = np.mean([r.ror for r in reports], axis=0)
  mean_orig = np.mean([r.ror_original for r in reports], axis=0)
  bench_arr = reports[0].benchmark_ror
  mean_eq = 1.0 + mean_ror
  mean_peaks = np.maximum.accumulate(mean_eq)
  mean_dd = (mean_eq - mean_peaks) / mean_peaks

  sum_full = query_curve(out_dir, obj="__SUMMARY__", max_points=500)
  assert sum_full["total_points"] == bars
  assert np.allclose(sum_full["series"]["ror"], mean_ror, atol=2e-4)
  assert np.allclose(sum_full["series"]["ror_original"], mean_orig, atol=2e-4)
  assert np.allclose(sum_full["series"]["benchmark_ror"], bench_arr, atol=2e-4)
  assert np.allclose(sum_full["series"]["drawdown"], mean_dd, atol=2e-4)

  sum_slice = query_curve(out_dir, obj="__SUMMARY__", start_idx=50, end_idx=120, max_points=500)
  assert np.allclose(sum_slice["series"]["ror"], mean_ror[50:120], atol=2e-4)
  assert np.allclose(sum_slice["series"]["drawdown"], mean_dd[50:120], atol=2e-4)


def test_cli_link_subcommand(tmp_path, capsys):
  from mbt.server.__main__ import main as server_main

  strat_file = _write_sample_strategy(tmp_path / "linked_strat.py")
  symlink_path = tmp_path / "linked_strat_sym.py"
  os.symlink(strat_file, symlink_path)
  unreg_file = _write_sample_strategy(tmp_path / "unregistered.py")
  canon_path = os.path.realpath(str(strat_file))

  class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
      if self.path == "/api/strategies":
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(
          json.dumps([{"id": "strat42", "path": canon_path}]).encode("utf-8")
        )
        return
      self.send_response(404)
      self.end_headers()

    def log_message(self, *_args):
      pass

  srv = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
  t = threading.Thread(target=srv.serve_forever, daemon=True)
  t.start()
  try:
    url = f"http://127.0.0.1:{srv.server_port}"
    assert server_main(["link", str(symlink_path), "--url", url]) == 0
    out_link = capsys.readouterr().out.strip()
    assert out_link == f"{url}/strategies/strat42"

    assert server_main(["link", str(unreg_file), "--url", url]) == 1
    err_unreg = capsys.readouterr().err
    assert "not registered" in err_unreg
  finally:
    srv.shutdown()
    srv.server_close()

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
  assert server_main(["link", str(strat_file), "--url", f"http://127.0.0.1:{free_port}"]) == 1
  err_conn = capsys.readouterr().err
  assert "serve --daemon" in err_conn


def test_cli_serve_daemon_register_link_and_spa(tmp_path, capsys):
  import re
  import signal
  import urllib.request
  from mbt.server.__main__ import main as server_main

  cfg_dir = tmp_path / "cfg"
  data_dir = tmp_path / "data"
  strat_file = _write_sample_strategy(tmp_path / "daemon_strat.py")

  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]

  url = f"http://127.0.0.1:{free_port}"
  pid: int | None = None
  try:
    rc = server_main([
      "serve",
      "--daemon",
      "--host",
      "127.0.0.1",
      "--port",
      str(free_port),
      "--config-dir",
      str(cfg_dir),
      "--data-dir",
      str(data_dir),
    ])
    assert rc == 0
    out_start = capsys.readouterr().out
    m = re.search(r"pid:\s*(\d+)", out_start)
    assert m is not None
    pid = int(m.group(1))
    assert (data_dir / "server.log").is_file()

    # Second daemon invocation is idempotent
    assert server_main([
      "serve",
      "--daemon",
      "--host",
      "127.0.0.1",
      "--port",
      str(free_port),
      "--config-dir",
      str(cfg_dir),
      "--data-dir",
      str(data_dir),
    ]) == 0
    out_dup = capsys.readouterr().out
    assert "already running" in out_dup

    assert server_main(["register", str(strat_file), "--url", url]) == 0
    reg_url = capsys.readouterr().out.strip()

    assert server_main(["link", str(strat_file), "--url", url]) == 0
    link_url = capsys.readouterr().out.strip()
    assert reg_url == link_url
    assert link_url.startswith(f"{url}/strategies/")

    # Registering a missing file fails fast and prints error reason
    assert server_main(["register", str(tmp_path / "missing.py"), "--url", url]) == 1
    err_reg = capsys.readouterr().err
    assert "Registration failed (404):" in err_reg
    assert "missing.py" in err_reg

    with urllib.request.urlopen(link_url, timeout=5.0) as resp:
      assert resp.status == 200
      assert "text/html" in resp.headers.get("Content-Type", "")
      html = resp.read().decode("utf-8", errors="replace")
      assert "MSD Backtest" in html
  finally:
    if pid is not None:
      try:
        os.kill(pid, signal.SIGTERM)
      except OSError:
        pass
      deadline = time.monotonic() + 5.0
      while time.monotonic() < deadline:
        try:
          os.kill(pid, 0)
          time.sleep(0.05)
        except OSError:
          break


def test_resolve_spec_default_objs_and_internal_selector(tmp_path):
  from mbt.server.runner import RunManager

  cfg_dir = tmp_path / "cfg"
  data_dir = tmp_path / "data"
  app = create_app(config_dir=cfg_dir, data_dir=data_dir)
  rm: RunManager = app.state.run_manager
  defaults = SystemDefaults()

  sel_only = tmp_path / "sel_only.py"
  sel_only.write_text(
    "import mbt\nclass OnlySel(mbt.Selector):\n  def step00(self, stocks): return stocks\n",
    encoding="utf-8",
  )
  strat_only = tmp_path / "strat_only.py"
  strat_only.write_text(
    "import mbt\nclass OnlyStrat(mbt.Strategy):\n  def execute(self, ctx): pass\n",
    encoding="utf-8",
  )
  port_only = tmp_path / "port_only.py"
  port_only.write_text(
    "import mbt\nclass OnlyPort(mbt.PortfolioStrategy):\n  def execute(self, ctx): pass\n",
    encoding="utf-8",
  )
  combo_timing = _write_sample_strategy(tmp_path / "combo_timing.py")
  combo_port = tmp_path / "combo_port.py"
  combo_port.write_text(
    "import mbt\nclass CSelector(mbt.Selector):\n  def step00(self, stocks): return stocks\n"
    "class CPort(mbt.PortfolioStrategy):\n  def execute(self, ctx): pass\n",
    encoding="utf-8",
  )
  fac_file = _write_sample_factor(tmp_path / "fac.py")

  with TestClient(app) as client:
    s_sel = client.post("/api/strategies", json={"path": str(sel_only)}).json()
    s_tim = client.post("/api/strategies", json={"path": str(strat_only)}).json()
    s_prt = client.post("/api/strategies", json={"path": str(port_only)}).json()
    s_ctim = client.post("/api/strategies", json={"path": str(combo_timing)}).json()
    s_cprt = client.post("/api/strategies", json={"path": str(combo_port)}).json()
    s_fac = client.post("/api/strategies", json={"path": str(fac_file)}).json()

    # Select, factor, timing without selector, portfolio without selector -> A_STOCKS_EXCLUDE_ST
    for strat in (s_sel, s_tim, s_prt, s_fac):
      spec = rm.resolve_spec(defaults, strat)
      assert spec["objs"] == "A_STOCKS_EXCLUDE_ST"
      assert spec["use_internal_selector"] is False

    # Timing and portfolio with embedded selector -> use_internal_selector=True
    for strat in (s_ctim, s_cprt):
      spec = rm.resolve_spec(defaults, strat)
      assert spec["use_internal_selector"] is True
      assert spec["objs"] == ""

    # Switching combo strategy to select mode -> defaults to A_STOCKS_EXCLUDE_ST
    spec_sel_mode = rm.resolve_spec(defaults, s_ctim, mode_override=RUN_MODE_SELECT)
    assert spec_sel_mode["objs"] == "A_STOCKS_EXCLUDE_ST"
    assert spec_sel_mode["use_internal_selector"] is False

    # Explicitly disabling internal selector without objs falls back to A_STOCKS_EXCLUDE_ST
    spec_fallback = rm.resolve_spec(
      defaults,
      s_cprt,
      overrides={"objs": "", "use_internal_selector": False},
    )
    assert spec_fallback["objs"] == "A_STOCKS_EXCLUDE_ST"
    assert spec_fallback["use_internal_selector"] is False

    # Explicit objs is always respected
    spec_explicit = rm.resolve_spec(defaults, s_cprt, overrides={"objs": "SH600000"})
    assert spec_explicit["objs"] == "SH600000"
    assert spec_explicit["use_internal_selector"] is False




