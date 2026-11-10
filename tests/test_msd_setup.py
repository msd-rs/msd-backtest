import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

# Ensure script directory or script itself is importable
SCRIPT_PATH = Path(__file__).resolve().parent.parent / ".agents" / "skills" / "msd-strategy-guide" / "scripts" / "msd_setup.py"

import importlib.util

spec = importlib.util.spec_from_file_location("msd_setup", SCRIPT_PATH)
msd_setup = importlib.util.module_from_spec(spec)
sys.modules["msd_setup"] = msd_setup
spec.loader.exec_module(msd_setup)


def test_resolve_workdir_default(monkeypatch):
  monkeypatch.delenv("MSD_WORKDIR", raising=False)
  workdir = msd_setup.resolve_workdir(None)
  assert workdir == Path(os.path.expanduser("~/data/msd")).resolve()


def test_resolve_workdir_custom(monkeypatch, tmp_path):
  custom_dir = tmp_path / "custom_msd"
  workdir = msd_setup.resolve_workdir(str(custom_dir))
  assert workdir == custom_dir.resolve()

  monkeypatch.setenv("MSD_WORKDIR", str(custom_dir))
  workdir_env = msd_setup.resolve_workdir(None)
  assert workdir_env == custom_dir.resolve()


def test_resolve_msd_host(monkeypatch, tmp_path):
  monkeypatch.delenv("MSD_HOST", raising=False)
  workdir = tmp_path / "msd_test"
  workdir.mkdir()

  # 1. Custom arg
  assert msd_setup.resolve_msd_host("http://1.2.3.4:9999/", workdir) == "http://1.2.3.4:9999"

  # 2. Env variable
  monkeypatch.setenv("MSD_HOST", "http://env-host:50510")
  assert msd_setup.resolve_msd_host(None, workdir) == "http://env-host:50510"
  monkeypatch.delenv("MSD_HOST")

  # 3. Workdir .env
  (workdir / ".env").write_text("MSD_SERVER_URL=http://workdir-host:50510\n", encoding="utf-8")
  assert msd_setup.resolve_msd_host(None, workdir) == "http://workdir-host:50510"


def test_parse_listen_addr():
  assert msd_setup.parse_listen_addr("http://127.0.0.1:50510") == "127.0.0.1:50510"
  assert msd_setup.parse_listen_addr("http://localhost:50511") == "127.0.0.1:50511"
  assert msd_setup.parse_listen_addr("http://0.0.0.0:8080") == "127.0.0.1:8080"


def test_get_executable_name(monkeypatch):
  monkeypatch.setattr("platform.system", lambda: "Windows")
  assert msd_setup.get_executable_name() == "msd.exe"

  monkeypatch.setattr("platform.system", lambda: "Linux")
  assert msd_setup.get_executable_name() == "msd"


def test_get_platform_asset_name(monkeypatch):
  monkeypatch.setattr("platform.system", lambda: "Linux")
  monkeypatch.setattr("platform.machine", lambda: "x86_64")
  assert msd_setup.get_platform_asset_name() == "msd-x86_64-unknown-linux-gnu.tar.gz"

  monkeypatch.setattr("platform.system", lambda: "Darwin")
  monkeypatch.setattr("platform.machine", lambda: "arm64")
  assert msd_setup.get_platform_asset_name() == "msd-aarch64-apple-darwin.tar.gz"

  monkeypatch.setattr("platform.system", lambda: "Windows")
  monkeypatch.setattr("platform.machine", lambda: "AMD64")
  assert msd_setup.get_platform_asset_name() == "msd-x86_64-pc-windows-msvc.zip"


def test_build_parser_subcommands():
  parser = msd_setup.build_parser()

  # install
  args = parser.parse_args(["install", "--force"])
  assert args.command == "install"
  assert args.force is True
  assert args.func == msd_setup.cmd_install

  # update and updte alias
  args_update = parser.parse_args(["update", "-d", "2026-10"])
  assert args_update.command == "update"
  assert args_update.date == "2026-10"
  assert args_update.func == msd_setup.cmd_update

  args_updte = parser.parse_args(["updte"])
  assert args_updte.command in ("update", "updte")
  assert args_updte.func == msd_setup.cmd_update

  # up
  args_up = parser.parse_args(["up", "-w", "/tmp/msd"])
  assert args_up.command == "up"
  assert args_up.workdir == "/tmp/msd"
  assert args_up.func == msd_setup.cmd_up

  # info
  args_info = parser.parse_args(["info", "-H", "http://localhost:50510"])
  assert args_info.command == "info"
  assert args_info.host == "http://localhost:50510"
  assert args_info.func == msd_setup.cmd_info


def test_get_pid_from_file(tmp_path):
  assert msd_setup.get_pid_from_file(tmp_path) is None

  pid_file = tmp_path / "msd.pid"
  pid_file.write_text(f"{os.getpid()}\n", encoding="utf-8")
  assert msd_setup.get_pid_from_file(tmp_path) == os.getpid()

  pid_file.write_text("999999999\n", encoding="utf-8")
  assert msd_setup.get_pid_from_file(tmp_path) is None


def test_cmd_info_stopped(capsys, tmp_path):
  args = argparse.Namespace(workdir=str(tmp_path), host="http://127.0.0.1:54321")
  with patch.object(msd_setup, "is_server_healthy", return_value=False):
    code = msd_setup.cmd_info(args)
    assert code == 0
    captured = capsys.readouterr().out
    assert "MSD 目录" in captured
    assert "服务状态: 未启动" in captured
    assert "MSD_HOST" not in captured


def test_generate_month_range():
  assert msd_setup.generate_month_range("2026-09", "2026-11") == ["2026-09", "2026-10", "2026-11"]
  assert msd_setup.generate_month_range("2026-09", "2026-09") == ["2026-09"]
  assert msd_setup.generate_month_range("2026-11", "2027-02") == ["2026-11", "2026-12", "2027-01", "2027-02"]
  assert msd_setup.generate_month_range("2026-11", "2026-09") == ["2026-11"]


def test_cmd_install_months(capsys, tmp_path):
  args = argparse.Namespace(workdir=str(tmp_path), host="http://127.0.0.1:50510", force=False, skip_import=False)
  bin_file = tmp_path / msd_setup.get_executable_name()
  bin_file.write_text("#!/bin/sh\n", encoding="utf-8")

  imported_months = []
  def fake_import_dataset(workdir, host, ym):
    imported_months.append(ym)
    return {"stock_kline_1d": True}

  with patch.object(msd_setup, "find_msd_bin", return_value=bin_file), \
       patch.object(msd_setup, "start_service", return_value=True), \
       patch.object(msd_setup, "init_tables", return_value=None), \
       patch.object(msd_setup, "import_dataset", side_effect=fake_import_dataset), \
       patch.object(msd_setup, "get_latest_data_date", return_value="2026-11-10"), \
       patch("msd_setup.datetime") as mock_dt:
    mock_dt.now.return_value = datetime(2026, 11, 10)
    mock_dt.strptime = datetime.strptime

    code = msd_setup.cmd_install(args)
    assert code == 0
    # Must import sequentially from 2026-09 to 2026-11
    assert imported_months == ["2026-09", "2026-10", "2026-11"]

  args = argparse.Namespace(workdir=str(tmp_path), host="http://127.0.0.1:50510")
  # Pretend binary is installed
  bin_file = tmp_path / msd_setup.get_executable_name()
  bin_file.write_text("#!/bin/sh\n", encoding="utf-8")

  with patch.object(msd_setup, "is_server_healthy", return_value=True), \
       patch.object(msd_setup, "find_running_pid", return_value=54321):
    code = msd_setup.cmd_up(args)
    assert code == 0
    captured = capsys.readouterr().out
    assert "MSD 服务已在运行" in captured
    assert "54321" in captured
    assert "http://127.0.0.1:50510" in captured


def test_cmd_up_start_service(capsys, tmp_path):
  args = argparse.Namespace(workdir=str(tmp_path), host="http://127.0.0.1:50510")
  bin_file = tmp_path / msd_setup.get_executable_name()
  bin_file.write_text("#!/bin/sh\n", encoding="utf-8")

  healthy_calls = [False, True]

  with patch.object(msd_setup, "is_server_healthy", side_effect=lambda *a, **k: healthy_calls.pop(0)), \
       patch.object(msd_setup, "start_service", return_value=True), \
       patch.object(msd_setup, "find_running_pid", return_value=67890):
    code = msd_setup.cmd_up(args)
    assert code == 0
    captured = capsys.readouterr().out
    assert "启动 MSD 服务" in captured
    assert "67890" in captured


def test_cmd_update(capsys, tmp_path):
  args = argparse.Namespace(workdir=str(tmp_path), host="http://127.0.0.1:50510", date="2026-11")
  bin_file = tmp_path / msd_setup.get_executable_name()
  bin_file.write_text("#!/bin/sh\n", encoding="utf-8")

  with patch.object(msd_setup, "is_server_healthy", return_value=True), \
       patch.object(msd_setup, "find_msd_bin", return_value=bin_file), \
       patch.object(msd_setup, "import_dataset", return_value={"stock_kline_1d": True}), \
       patch.object(msd_setup, "get_latest_data_date", return_value="2026-11-06"):
    code = msd_setup.cmd_update(args)
    assert code == 0
    captured = capsys.readouterr().out
    assert "2026-11" in captured
    assert "成功导入 1/1" in captured
    assert "2026-11-06" in captured

