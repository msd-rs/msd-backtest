# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Command-line entry point for ``mbt.server`` and ``mbt-server``."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from mbt.server.app import create_app
from mbt.server.config import get_config_file, get_data_dir, load_server_config


def _probe_host_for(listen_host: str) -> str:
  h = listen_host.strip()
  if h in ("0.0.0.0", "::", "", "*"):
    return "127.0.0.1"
  return h


def _is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
  try:
    with socket.create_connection((host, port), timeout=timeout):
      return True
  except OSError:
    return False


def check_existing_server(
  host: str,
  port: int,
  *,
  timeout: float = 1.0,
) -> int | None:
  """Check whether a server already listens at ``host:port``.

  Returns:
    ``0`` if a healthy ``msd-backtest`` server is already running (caller should exit 0).
    ``1`` if the port is occupied by another unrelated service (caller should exit 1).
    ``None`` if the port is free to bind.
  """
  probe_host = _probe_host_for(host)
  url = f"http://{probe_host}:{port}/api/health"
  try:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
      body = resp.read().decode("utf-8", errors="replace").strip()
      if resp.status == 200 and body.startswith("OK, msd-backtest"):
        print(f"mbt.server is already running at http://{probe_host}:{port} ({body}).")
        return 0
      print(
        f"Error: Port {port} on {probe_host} responded with unexpected payload from non-mbt service.",
        file=sys.stderr,
      )
      return 1
  except urllib.error.HTTPError as exc:
    print(
      f"Error: Port {port} on {probe_host} is occupied by another HTTP service (status {exc.code}).",
      file=sys.stderr,
    )
    return 1
  except urllib.error.URLError:
    if _is_port_open(probe_host, port, timeout=0.3):
      print(
        f"Error: Port {port} on {probe_host} is occupied by a non-HTTP service.",
        file=sys.stderr,
      )
      return 1
    return None


def _resolve_server_url(
  explicit_url: str | None,
  config_dir: str | None,
) -> str:
  if explicit_url:
    return explicit_url.rstrip("/")
  cfg = load_server_config(config_dir)
  probe_host = _probe_host_for(cfg.server.host)
  return f"http://{probe_host}:{cfg.server.port}"


def _start_daemon(
  host: str,
  port: int,
  cfg_dir: Path | None,
  data_dir: Path | None,
  *,
  timeout: float = 15.0,
) -> int:
  """Spawn ``mbt.server serve`` in the background and wait until ``/api/health`` responds."""
  probe_host = _probe_host_for(host)
  data_root = get_data_dir(data_dir)
  data_root.mkdir(parents=True, exist_ok=True)
  log_path = data_root / "server.log"

  cmd = [
    sys.executable,
    "-m",
    "mbt.server",
    "serve",
    "--host",
    host,
    "--port",
    str(port),
  ]
  if cfg_dir is not None:
    cmd.extend(["--config-dir", str(cfg_dir)])
  if data_dir is not None:
    cmd.extend(["--data-dir", str(data_dir)])

  popen_kwargs: dict[str, Any] = {"close_fds": True}
  if sys.platform == "win32":
    popen_kwargs["creationflags"] = (
      subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    )
  else:
    popen_kwargs["start_new_session"] = True

  with open(log_path, "a", encoding="utf-8") as log_fh:
    proc = subprocess.Popen(
      cmd,
      stdin=subprocess.DEVNULL,
      stdout=log_fh,
      stderr=subprocess.STDOUT,
      **popen_kwargs,
    )

  health_url = f"http://{probe_host}:{port}/api/health"
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    rc = proc.poll()
    if rc is not None:
      tail = ""
      try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        if lines:
          tail = "\n" + "\n".join(lines[-20:])
      except OSError:
        pass
      print(
        f"Error: mbt.server exited early with status {rc}. Log: {log_path}{tail}",
        file=sys.stderr,
      )
      return 1
    try:
      with urllib.request.urlopen(health_url, timeout=0.5) as resp:
        body = resp.read().decode("utf-8", errors="replace").strip()
        if resp.status == 200 and body.startswith("OK, msd-backtest"):
          print(
            f"Started mbt.server in background at http://{probe_host}:{port} "
            f"(pid: {proc.pid}, log: {log_path})"
          )
          return 0
    except (urllib.error.URLError, OSError):
      pass
    time.sleep(0.1)

  print(
    f"Error: Timed out waiting for mbt.server at {health_url} after {timeout:.1f}s. "
    f"Check log: {log_path}",
    file=sys.stderr,
  )
  return 1


def _cmd_serve(args: argparse.Namespace) -> int:
  import uvicorn

  cfg_dir = Path(args.config_dir) if args.config_dir else None
  data_dir = Path(args.data_dir) if args.data_dir else None
  cfg = load_server_config(cfg_dir)

  host = args.host or cfg.server.host
  port = int(args.port or cfg.server.port)

  existing_rc = check_existing_server(host, port)
  if existing_rc is not None:
    return existing_rc

  if getattr(args, "daemon", False):
    return _start_daemon(host, port, cfg_dir, data_dir)

  app = create_app(config_dir=cfg_dir, data_dir=data_dir)
  print(
    f"Starting mbt.server on http://{_probe_host_for(host)}:{port} "
    f"(config: {get_config_file(cfg_dir)})"
  )
  uvicorn.run(app, host=host, port=port, log_level="info")
  return 0


def _format_http_error(exc: urllib.error.HTTPError) -> str:
  raw = exc.read().decode("utf-8", errors="replace").strip()
  if raw:
    try:
      parsed = json.loads(raw)
      if isinstance(parsed, dict) and "detail" in parsed:
        return str(parsed["detail"])
    except json.JSONDecodeError:
      pass
    return raw
  return exc.reason or f"HTTP {exc.code}"


def _cmd_register(args: argparse.Namespace) -> int:
  base_url = _resolve_server_url(args.url, args.config_dir)
  abs_path = os.path.realpath(os.path.abspath(args.path))
  payload = {
    "path": abs_path,
    "name": args.name,
    "notes": args.notes,
  }
  req = urllib.request.Request(
    f"{base_url}/api/strategies",
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
  )
  try:
    with urllib.request.urlopen(req, timeout=10.0) as resp:
      body = resp.read().decode("utf-8")
      rec = json.loads(body)
  except urllib.error.HTTPError as exc:
    print(f"Registration failed ({exc.code}): {_format_http_error(exc)}", file=sys.stderr)
    return 1
  except urllib.error.URLError as exc:
    print(
      f"Failed to connect to mbt.server at {base_url}: {exc.reason}. "
      "Start the server first via `uv run mbt-server serve --daemon`.",
      file=sys.stderr,
    )
    return 1
  except json.JSONDecodeError as exc:
    print(f"Invalid JSON response from mbt.server at {base_url}: {exc}", file=sys.stderr)
    return 1

  strat_id = rec.get("id") if isinstance(rec, dict) else None
  if not strat_id:
    print(f"Server response missing strategy id: {rec}", file=sys.stderr)
    return 1

  print(f"{base_url}/strategies/{strat_id}")
  return 0


def _cmd_link(args: argparse.Namespace) -> int:
  base_url = _resolve_server_url(args.url, args.config_dir)
  abs_path = os.path.realpath(os.path.abspath(args.path))
  try:
    with urllib.request.urlopen(f"{base_url}/api/strategies", timeout=10.0) as resp:
      strategies = json.loads(resp.read().decode("utf-8"))
  except urllib.error.HTTPError as exc:
    print(f"Failed to query strategies ({exc.code}): {_format_http_error(exc)}", file=sys.stderr)
    return 1
  except urllib.error.URLError as exc:
    print(
      f"Failed to connect to mbt.server at {base_url}: {exc.reason}. "
      "Start the server first via `uv run mbt-server serve --daemon`.",
      file=sys.stderr,
    )
    return 1

  matched = next(
    (
      item
      for item in strategies
      if item.get("path")
      and os.path.realpath(os.path.abspath(str(item["path"]))) == abs_path
    ),
    None,
  )
  if matched is None:
    print(
      f"Strategy path '{abs_path}' is not registered on mbt.server at {base_url}. "
      f"Register it first via `uv run mbt-server register {args.path}`.",
      file=sys.stderr,
    )
    return 1

  print(f"{base_url}/strategies/{matched['id']}")
  return 0


def _cmd_status(args: argparse.Namespace) -> int:
  base_url = _resolve_server_url(args.url, args.config_dir)
  try:
    with urllib.request.urlopen(f"{base_url}/api/health", timeout=3.0) as resp:
      body = resp.read().decode("utf-8", errors="replace").strip()
      print(f"{base_url} -> {body}")
      return 0
  except Exception as exc:
    print(f"mbt.server not reachable at {base_url}: {exc}", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
  """Construct the CLI parser for ``mbt-server`` / ``python -m mbt.server``."""
  parser = argparse.ArgumentParser(
    prog="mbt-server",
    description="MSD Backtest strategy management server & Web UI",
  )
  parser.add_argument("--host", default=None, help="Override listen host from server.json")
  parser.add_argument("--port", type=int, default=None, help="Override listen port from server.json")
  parser.add_argument("--config-dir", default=None, help="Override OS config directory")
  parser.add_argument("--data-dir", default=None, help="Override OS user data directory")
  parser.add_argument(
    "-d",
    "--daemon",
    action="store_true",
    default=False,
    help="Start the server in the background and exit once healthy",
  )

  sub = parser.add_subparsers(dest="command")

  p_serve = sub.add_parser("serve", help="Start the strategy management HTTP server")
  p_serve.add_argument("--host", default=argparse.SUPPRESS, help="Listen host")
  p_serve.add_argument("--port", type=int, default=argparse.SUPPRESS, help="Listen port")
  p_serve.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")
  p_serve.add_argument("--data-dir", default=argparse.SUPPRESS, help="Override OS user data directory")
  p_serve.add_argument(
    "-d",
    "--daemon",
    action="store_true",
    default=argparse.SUPPRESS,
    help="Start the server in the background and exit once healthy",
  )

  p_reg = sub.add_parser("register", help="Submit a strategy or factor file path to the running server")
  p_reg.add_argument("path", help="Path to .py strategy/selector/factor or .txt/.json expression file")
  p_reg.add_argument("--name", default=None, help="Optional display name")
  p_reg.add_argument("--notes", default=None, help="Optional notes")
  p_reg.add_argument("--url", default=None, help="Override server URL (default from server.json)")
  p_reg.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")

  p_link = sub.add_parser("link", help="Print the Web UI URL for a registered strategy file path")
  p_link.add_argument("path", help="Path to .py strategy/selector/factor or .txt/.json expression file")
  p_link.add_argument("--url", default=None, help="Override server URL (default from server.json)")
  p_link.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")

  p_stat = sub.add_parser("status", help="Check whether mbt.server is running")
  p_stat.add_argument("--url", default=None, help="Override server URL")
  p_stat.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")

  return parser


def main(argv: Sequence[str] | None = None) -> int:
  """Run the ``mbt-server`` CLI."""
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.command == "register":
    return _cmd_register(args)
  if args.command == "link":
    return _cmd_link(args)
  if args.command == "status":
    return _cmd_status(args)
  return _cmd_serve(args)


if __name__ == "__main__":
  sys.exit(main())
