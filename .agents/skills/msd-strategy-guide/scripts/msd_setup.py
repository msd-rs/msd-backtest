#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MSD database setup, launch, and data management tool.

Supported subcommands:
  install - Full installation of MSD to default location, start service, and import initial data.
  update  - Update data for the latest month (alias: updte).
  up      - Start MSD service; if not installed, perform installation first.
  info    - Print MSD directory, whether service is running, and MSD_HOST if running.
"""

import argparse
from datetime import datetime
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Optional
import urllib.error
import urllib.parse
import urllib.request
import zipfile

DEFAULT_WORKDIR = Path(os.environ.get("MSD_WORKDIR", os.path.expanduser("~/data/msd")))
DEFAULT_MSD_HOST = "http://127.0.0.1:50510"
HISTORY_PACKAGE_DATE = "2026-09"
CREATE_SQL_URL = "https://cnb.cool/elsejj/msd/data/-/git/raw/main/create.sql"

TABLES = [
  "stock_capital_flow",
  "stock_shares",
  "stock_kline_1d",
  "stock_infos",
  "stock_financial",
  "stock_dividend",
  "stock_block",
]


def parse_env_file(path: Path) -> dict[str, str]:
  """Parse key-value pairs from .env file."""
  env = {}
  if path.is_file():
    try:
      for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
          key, val = line.split("=", 1)
          env[key.strip()] = val.strip().strip("'\"")
    except Exception:
      pass
  return env


def resolve_workdir(args_workdir: Optional[str]) -> Path:
  """Resolve MSD working directory."""
  if args_workdir:
    return Path(args_workdir).expanduser().resolve()
  env_workdir = os.environ.get("MSD_WORKDIR")
  if env_workdir:
    return Path(env_workdir).expanduser().resolve()
  return DEFAULT_WORKDIR.expanduser().resolve()


def resolve_msd_host(args_host: Optional[str], workdir: Path) -> str:
  """Resolve MSD_HOST in priority order: arg -> os.environ -> workdir/.env -> cwd/.env -> default."""
  if args_host:
    return args_host.rstrip("/")

  if os.environ.get("MSD_HOST"):
    return os.environ["MSD_HOST"].rstrip("/")

  workdir_env = parse_env_file(workdir / ".env")
  if "MSD_SERVER_URL" in workdir_env:
    return workdir_env["MSD_SERVER_URL"].rstrip("/")
  if "MSD_HOST" in workdir_env:
    return workdir_env["MSD_HOST"].rstrip("/")

  cwd_env = parse_env_file(Path.cwd() / ".env")
  if "MSD_HOST" in cwd_env:
    return cwd_env["MSD_HOST"].rstrip("/")
  if "MSD_SERVER_URL" in cwd_env:
    return cwd_env["MSD_SERVER_URL"].rstrip("/")

  return DEFAULT_MSD_HOST


def parse_listen_addr(msd_host: str) -> str:
  """Extract host:port listen address from URL for 'msd server -l'."""
  parsed = urllib.parse.urlparse(msd_host)
  host = parsed.hostname or "127.0.0.1"
  port = parsed.port or 50510
  if host in ("localhost", "0.0.0.0"):
    host = "127.0.0.1"
  return f"{host}:{port}"


def get_executable_name() -> str:
  """Return executable file name depending on OS."""
  return "msd.exe" if platform.system() == "Windows" else "msd"


def find_msd_bin(workdir: Path) -> Optional[Path]:
  """Find msd binary in workdir, or fallback to system PATH."""
  local_bin = workdir / get_executable_name()
  if local_bin.is_file() and os.access(local_bin, os.X_OK):
    return local_bin
  which_bin = shutil.which("msd")
  if which_bin:
    return Path(which_bin).resolve()
  if local_bin.is_file():
    return local_bin
  return None


def get_platform_asset_name() -> str:
  """Determine binary release archive name based on OS and architecture."""
  sys_name = platform.system().lower()
  machine = platform.machine().lower()

  if sys_name == "linux":
    if machine in ("x86_64", "amd64"):
      return "msd-x86_64-unknown-linux-gnu.tar.gz"
    if machine in ("aarch64", "arm64"):
      return "msd-aarch64-unknown-linux-gnu.tar.gz"
  elif sys_name == "darwin":
    if machine in ("arm64", "aarch64"):
      return "msd-aarch64-apple-darwin.tar.gz"
    if machine in ("x86_64", "amd64"):
      return "msd-x86_64-apple-darwin.tar.gz"
  elif sys_name in ("windows", "win32"):
    return "msd-x86_64-pc-windows-msvc.zip"

  raise RuntimeError(f"Unsupported platform: {platform.system()} {platform.machine()}")


def get_latest_version() -> str:
  """Query latest release tag from CNB or GitHub via HTTP redirect."""
  endpoints = [
    "https://cnb.cool/elsejj/msd-rs/-/releases/latest",
    "https://github.com/msd-rs/msd/releases/latest",
  ]
  for url in endpoints:
    try:
      req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})

      class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, headers, newurl):
          return None

      opener = urllib.request.build_opener(NoRedirect)
      loc = None
      try:
        resp = opener.open(req, timeout=10)
        loc = resp.headers.get("Location")
      except urllib.error.HTTPError as e:
        loc = e.headers.get("Location")

      if loc and "/releases/tag/" in loc:
        version = loc.split("/releases/tag/")[-1].strip().strip("/")
        if version:
          return version
    except Exception:
      continue

  # Fallback to known stable version if network redirects fail
  return "v0.1.27"


def download_file(url: str, dest_path: Path, timeout: int = 120) -> None:
  """Download a file with user-agent header."""
  req = urllib.request.Request(url, headers={"User-Agent": "curl/7.68.0"})
  with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest_path, "wb") as f:
    shutil.copyfileobj(resp, f)


def download_and_extract_msd(workdir: Path, version: Optional[str] = None) -> Path:
  """Download MSD archive from CNB or GitHub and extract to workdir."""
  workdir.mkdir(parents=True, exist_ok=True)
  if not version:
    print("Checking latest MSD release version...")
    version = get_latest_version()
  print(f"Target MSD version: {version}")

  asset_name = get_platform_asset_name()
  archive_path = workdir / asset_name
  bin_name = get_executable_name()
  bin_path = workdir / bin_name

  download_urls = [
    f"https://cnb.cool/elsejj/msd-rs/-/releases/download/{version}/{asset_name}",
    f"https://github.com/msd-rs/msd/releases/download/{version}/{asset_name}",
  ]

  download_success = False
  for url in download_urls:
    try:
      print(f"Downloading {asset_name} from {url} ...")
      download_file(url, archive_path)
      download_success = True
      break
    except Exception as e:
      print(f"Download failed from {url}: {e}")

  if not download_success or not archive_path.is_file():
    raise RuntimeError(f"Failed to download {asset_name} from mirrors.")

  print(f"Extracting {asset_name} into {workdir} ...")
  if asset_name.endswith(".tar.gz") or asset_name.endswith(".tgz"):
    with tarfile.open(archive_path, "r:gz") as tar:
      for member in tar.getmembers():
        if member.name.endswith(bin_name) or Path(member.name).name == bin_name:
          extracted = tar.extractfile(member)
          if extracted:
            with open(bin_path, "wb") as out:
              shutil.copyfileobj(extracted, out)
            break
      else:
        tar.extractall(path=workdir)
  elif asset_name.endswith(".zip"):
    with zipfile.ZipFile(archive_path, "r") as zf:
      for name in zf.namelist():
        if name.endswith(bin_name) or Path(name).name == bin_name:
          with zf.open(name) as src, open(bin_path, "wb") as out:
            shutil.copyfileobj(src, out)
          break
      else:
        zf.extractall(path=workdir)

  # Cleanup archive file
  try:
    archive_path.unlink()
  except Exception:
    pass

  if not bin_path.is_file():
    raise RuntimeError(f"Extracted binary not found at {bin_path}")

  # Ensure executable permission
  if platform.system() != "Windows":
    bin_path.chmod(0o755)

  print(f"MSD binary installed to: {bin_path}")
  return bin_path


def is_server_healthy(msd_host: str, timeout: float = 1.0) -> bool:
  """Check whether MSD server /health endpoint returns 200 OK."""
  health_url = f"{msd_host.rstrip('/')}/health"
  try:
    req = urllib.request.Request(health_url, headers={"User-Agent": "curl/7.68.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
      return resp.status == 200 and resp.read().strip() == b"OK"
  except Exception:
    return False


def get_pid_from_file(workdir: Path) -> Optional[int]:
  """Read PID from msd.pid and check whether the process is alive."""
  pid_file = workdir / "msd.pid"
  if not pid_file.is_file():
    return None
  try:
    pid_str = pid_file.read_text().strip()
    if not pid_str:
      return None
    pid = int(pid_str)
    if platform.system() != "Windows":
      os.kill(pid, 0)
    return pid
  except OSError:
    return None
  except Exception:
    return None


def find_running_pid(workdir: Path, msd_host: str) -> Optional[int]:
  """Locate PID from workdir msd.pid, or by inspecting running processes."""
  file_pid = get_pid_from_file(workdir)
  if file_pid:
    return file_pid

  if not is_server_healthy(msd_host):
    return None

  # Parse port to match process
  parsed = urllib.parse.urlparse(msd_host)
  port = parsed.port or 50510

  if Path("/proc").is_dir():
    try:
      for proc_dir in Path("/proc").glob("[0-9]*"):
        try:
          cmdline_file = proc_dir / "cmdline"
          if cmdline_file.is_file():
            cmdline = cmdline_file.read_bytes().decode("utf-8", errors="ignore")
            if "msd" in cmdline and "server" in cmdline:
              if str(port) in cmdline or (port == 50510 and "-l" not in cmdline):
                return int(proc_dir.name)
        except Exception:
          continue
    except Exception:
      pass

  return None


def start_service(workdir: Path, msd_host: str, max_wait: int = 30) -> bool:
  """Start MSD server in background and poll /health until ready."""
  if is_server_healthy(msd_host):
    print(f"MSD service is already running on {msd_host}.")
    return True

  bin_path = find_msd_bin(workdir)
  if not bin_path or not bin_path.is_file():
    raise FileNotFoundError(f"MSD binary not found in {workdir}. Please install it first.")

  # Ensure workdir and logs directory exist
  workdir.mkdir(parents=True, exist_ok=True)
  logs_dir = workdir / "logs"
  logs_dir.mkdir(parents=True, exist_ok=True)

  # Ensure .env exists in workdir
  workdir_env_file = workdir / ".env"
  if not workdir_env_file.exists():
    workdir_env_file.write_text("RUST_LOG=msd=info\n", encoding="utf-8")

  listen_addr = parse_listen_addr(msd_host)
  pid_file = "msd.pid"
  cmd = [str(bin_path), "--log-dir", "./logs", "server", "-p", pid_file, "-l", listen_addr]

  print(f"Starting MSD server in {workdir} listening on {listen_addr} ...")

  popen_kwargs = {
    "cwd": str(workdir),
    "stdout": subprocess.DEVNULL,
    "stderr": subprocess.DEVNULL,
    "stdin": subprocess.DEVNULL,
  }

  if platform.system() == "Windows":
    popen_kwargs["creationflags"] = (
      getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
    )
  else:
    popen_kwargs["start_new_session"] = True

  subprocess.Popen(cmd, **popen_kwargs)

  # Bounded polling for /health check (msd.md recommendation)
  print(f"Waiting for MSD server to become ready at {msd_host}/health (up to {max_wait}s)...")
  for _ in range(max_wait):
    if is_server_healthy(msd_host):
      pid = find_running_pid(workdir, msd_host)
      pid_msg = f" (PID: {pid})" if pid else ""
      print(f"MSD server is ready{pid_msg}!")
      return True
    time.sleep(1)

  # Check logs for diagnostic info if startup failed
  latest_logs = ""
  try:
    log_files = sorted(logs_dir.glob("msd.log.*"))
    if log_files:
      latest_logs = log_files[-1].read_text(encoding="utf-8", errors="ignore")[-1000:]
  except Exception:
    pass

  err_msg = f"MSD server failed to start within {max_wait} seconds."
  if latest_logs:
    err_msg += f"\nRecent log snippet:\n{latest_logs}"
  raise RuntimeError(err_msg)


def init_tables(workdir: Path, msd_host: str) -> None:
  """Download create.sql if needed and execute it via 'msd shell'."""
  bin_path = find_msd_bin(workdir)
  if not bin_path:
    raise FileNotFoundError("MSD executable not found.")

  workdir.mkdir(parents=True, exist_ok=True)
  sql_path = workdir / "create.sql"
  if not sql_path.is_file():
    print(f"Downloading table creation SQL from {CREATE_SQL_URL} ...")
    try:
      download_file(CREATE_SQL_URL, sql_path)
    except Exception as e:
      raise RuntimeError(f"Failed to download create.sql: {e}")

  print("Initializing tables using create.sql ...")
  cwd = str(workdir) if workdir.is_dir() else None
  cmd = [str(bin_path), "shell", "-s", msd_host, str(sql_path)]
  res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
  if res.returncode != 0:
    print(f"Table initialization notice: {res.stderr or res.stdout}")
  else:
    print("Tables initialized successfully.")


def import_table(bin_path: Path, workdir: Path, msd_host: str, date_str: str, table_name: str) -> bool:
  """Import a single table data using 'msd shell .import'."""
  url = f"https://cnb.cool/elsejj/msd/data/-/releases/download/{date_str}/{table_name}.tbl.zstd"
  import_cmd = f".import {url} {table_name}"
  cwd = str(workdir) if workdir.is_dir() else None
  cmd = [str(bin_path), "shell", "-s", msd_host, import_cmd]
  res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
  output = (res.stdout + "\n" + res.stderr).strip()

  if "Import successful" in output or res.returncode == 0:
    match = re.search(r"Import successful: (\{.*?\})", output)
    info = f" -> {match.group(1)}" if match else ""
    print(f"  [OK] {table_name} ({date_str}){info}")
    return True
  else:
    print(f"  [WARN] Failed to import {table_name} from {url}:\n    {output}")
    return False


def import_dataset(workdir: Path, msd_host: str, date_str: str) -> dict[str, bool]:
  """Import all standard tables for a given month package."""
  bin_path = find_msd_bin(workdir)
  if not bin_path:
    raise FileNotFoundError("MSD executable not found.")

  print(f"\nImporting dataset for {date_str} ({len(TABLES)} tables)...")
  results = {}
  for table in TABLES:
    results[table] = import_table(bin_path, workdir, msd_host, date_str, table)
  return results


def generate_month_range(start_ym: str, end_ym: str) -> list[str]:
  """Generate list of 'YYYY-MM' strings from start_ym to end_ym inclusive."""
  try:
    start_dt = datetime.strptime(start_ym, "%Y-%m")
    end_dt = datetime.strptime(end_ym, "%Y-%m")
  except ValueError:
    return [start_ym]

  if start_dt > end_dt:
    return [start_ym]

  months = []
  cur_year = start_dt.year
  cur_month = start_dt.month
  end_year = end_dt.year
  end_month = end_dt.month

  while (cur_year, cur_month) <= (end_year, end_month):
    months.append(f"{cur_year:04d}-{cur_month:02d}")
    if cur_month == 12:
      cur_year += 1
      cur_month = 1
    else:
      cur_month += 1
  return months


def get_latest_data_date(workdir: Path, msd_host: str) -> Optional[str]:
  """Query the latest timestamp of SH000001 from stock_kline_1d."""
  bin_path = find_msd_bin(workdir)
  if not bin_path or not is_server_healthy(msd_host):
    return None
  try:
    sql = "select ts from stock_kline_1d where obj='SH000001' limit -1"
    cwd = str(workdir) if workdir.is_dir() else None
    cmd = [str(bin_path), "shell", "-s", msd_host, sql]
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=5)
    if res.returncode == 0 and "SH000001" in res.stdout:
      for line in res.stdout.splitlines():
        if "SH000001" in line:
          parts = [p.strip() for p in line.split("|") if p.strip()]
          if len(parts) >= 3:
            return parts[2]
  except Exception:
    pass
  return None


# ==============================================================================
# Subcommand implementations
# ==============================================================================

def cmd_install(args) -> int:
  """install: 完整的安装msd到默认位置, 启动后, 并进行一次数据导入."""
  workdir = resolve_workdir(args.workdir)
  msd_host = resolve_msd_host(args.host, workdir)

  print("==================================================")
  print("开始完整安装 MSD (Full Installation)")
  print(f"安装目录 (Workdir) : {workdir}")
  print(f"服务地址 (MSD_HOST): {msd_host}")
  print("==================================================")

  # 1. Download & Extract binary
  bin_path = find_msd_bin(workdir)
  if bin_path and bin_path.parent == workdir and not getattr(args, "force", False):
    print(f"MSD 可执行文件已存在: {bin_path}")
  else:
    download_and_extract_msd(workdir)

  # 2. Start MSD server
  start_service(workdir, msd_host)

  # 3. Create tables
  init_tables(workdir, msd_host)

  # 4. Import initial data
  if not getattr(args, "skip_import", False):
    current_month = datetime.now().strftime("%Y-%m")
    months_to_import = generate_month_range(HISTORY_PACKAGE_DATE, current_month)
    total_months = len(months_to_import)
    print(f"\n--- 导入自历史总包至本月数据 ({HISTORY_PACKAGE_DATE} ~ {current_month}, 共 {total_months} 个月份) ---")
    for idx, ym in enumerate(months_to_import, 1):
      desc = "历史总包" if ym == HISTORY_PACKAGE_DATE else "月度增量"
      print(f"\n[{idx}/{total_months}] 正在导入 {ym} ({desc})...")
      import_dataset(workdir, msd_host, ym)

  latest_ts = get_latest_data_date(workdir, msd_host)
  print("\n==================================================")
  print("MSD 安装与初始化完成！")
  print(f"MSD 目录  : {workdir}")
  print(f"服务状态  : 已启动")
  print(f"MSD_HOST  : {msd_host}")
  if latest_ts:
    print(f"最新数据  : {latest_ts} (SH000001)")
  print("==================================================")
  return 0


def cmd_update(args) -> int:
  """update: 更新最新月份的数据."""
  workdir = resolve_workdir(args.workdir)
  msd_host = resolve_msd_host(args.host, workdir)

  if not is_server_healthy(msd_host):
    print("MSD 服务未运行，正在启动服务...")
    start_service(workdir, msd_host)

  bin_path = find_msd_bin(workdir)
  if not bin_path:
    raise FileNotFoundError(f"未找到 MSD 可执行文件。请先运行 install 子命令。")

  current_month = datetime.now().strftime("%Y-%m")
  target_date = getattr(args, "date", None)

  if target_date:
    months_to_update = [target_date]
  else:
    # 检查数据库最后时间，若落后于当前月份，则补充中间缺失月份
    latest_ts = get_latest_data_date(workdir, msd_host)
    if latest_ts and len(latest_ts) >= 7 and latest_ts[:4].isdigit():
      db_month = latest_ts[:7]
      months_to_update = generate_month_range(db_month, current_month)
    else:
      months_to_update = [current_month]

  print(f"准备更新月份数据 ({', '.join(months_to_update)})...")
  total_success = 0
  total_tables = 0
  for ym in months_to_update:
    print(f"\n--- 正在更新 {ym} 月份数据 ---")
    results = import_dataset(workdir, msd_host, ym)
    total_success += sum(1 for v in results.values() if v)
    total_tables += len(results)

  print(f"\n数据更新完成: 成功导入 {total_success}/{total_tables} 表次。")

  latest_ts = get_latest_data_date(workdir, msd_host)
  if latest_ts:
    print(f"数据库最新交易日 (SH000001): {latest_ts}")
  return 0 if total_success > 0 else 1


def cmd_up(args) -> int:
  """up: 启动msd服务, 如果没有安装, 则执行安装."""
  workdir = resolve_workdir(args.workdir)
  msd_host = resolve_msd_host(args.host, workdir)

  bin_path = find_msd_bin(workdir)
  # Check if installed at target workdir
  if not (workdir / get_executable_name()).is_file():
    print(f"目录 {workdir} 尚未安装 MSD，将执行安装...")
    return cmd_install(args)

  if is_server_healthy(msd_host):
    pid = find_running_pid(workdir, msd_host)
    print("MSD 服务已在运行。")
    print(f"MSD 目录  : {workdir}")
    print(f"服务状态  : 已启动")
    if pid:
      print(f"进程 PID  : {pid}")
    print(f"MSD_HOST  : {msd_host}")
    return 0

  print(f"正在启动 MSD 服务 (工作目录: {workdir}) ...")
  start_service(workdir, msd_host)
  pid = find_running_pid(workdir, msd_host)
  print("MSD 服务启动成功。")
  print(f"MSD 目录  : {workdir}")
  print(f"服务状态  : 已启动")
  if pid:
    print(f"进程 PID  : {pid}")
  print(f"MSD_HOST  : {msd_host}")
  return 0


def cmd_info(args) -> int:
  """info: 打印msd所在的目录, 服务是否启动, 若启动了, 同时打印 MSD_HOST."""
  workdir = resolve_workdir(args.workdir)
  msd_host = resolve_msd_host(args.host, workdir)
  bin_path = find_msd_bin(workdir)
  is_running = is_server_healthy(msd_host)
  pid = find_running_pid(workdir, msd_host)

  print(f"MSD 目录: {workdir}")
  if bin_path:
    print(f"MSD 程序: {bin_path}")
  if is_running:
    print("服务状态: 已启动")
    if pid:
      print(f"进程 PID: {pid}")
    print(f"MSD_HOST: {msd_host}")
    latest_ts = get_latest_data_date(workdir, msd_host)
    if latest_ts:
      print(f"最新数据: {latest_ts} (SH000001)")
  else:
    print("服务状态: 未启动")

  return 0


def build_parser() -> argparse.ArgumentParser:
  """Construct CLI argument parser with required subcommands."""
  common_parser = argparse.ArgumentParser(add_help=False)
  common_parser.add_argument(
    "-w", "--workdir",
    type=str,
    default=None,
    help="MSD 数据库工作目录 (默认: $MSD_WORKDIR 或 ~/data/msd)",
  )
  common_parser.add_argument(
    "-H", "--host",
    type=str,
    default=None,
    help="MSD 服务地址 (默认: $MSD_HOST 或 http://127.0.0.1:50510)",
  )

  parser = argparse.ArgumentParser(
    description="MSD 数据库安装、启动与数据更新管理工具",
    parents=[common_parser],
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )

  subparsers = parser.add_subparsers(dest="command", required=True)

  # install
  p_install = subparsers.add_parser(
    "install",
    parents=[common_parser],
    help="完整的安装msd到默认位置, 启动后, 并进行一次数据导入",
  )
  p_install.add_argument("--force", action="store_true", help="强制重新下载安装")
  p_install.add_argument("--skip-import", action="store_true", help="跳过数据导入步骤")
  p_install.set_defaults(func=cmd_install)

  # update (with alias 'updte')
  p_update = subparsers.add_parser(
    "update",
    aliases=["updte"],
    parents=[common_parser],
    help="更新最新月份的数据",
  )
  p_update.add_argument(
    "-d", "--date",
    type=str,
    default=None,
    help="要更新的月份 (格式: YYYY-MM，默认当月)",
  )
  p_update.set_defaults(func=cmd_update)

  # up
  p_up = subparsers.add_parser(
    "up",
    parents=[common_parser],
    help="启动msd服务, 如果没有安装, 则执行安装",
  )
  p_up.set_defaults(func=cmd_up)

  # info
  p_info = subparsers.add_parser(
    "info",
    parents=[common_parser],
    help="打印msd所在的目录, 服务是否启动, 若启动了, 同时打印 MSD_HOST",
  )
  p_info.set_defaults(func=cmd_info)

  return parser


def main() -> int:
  parser = build_parser()
  args = parser.parse_args()
  return args.func(args)


if __name__ == "__main__":
  sys.exit(main())
