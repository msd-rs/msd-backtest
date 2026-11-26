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
from mbt.server.schemas import ShortcutParamsModel
from mbt.shared.inspect_strategy import VALID_RUN_MODES


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
  *,
  host: str | None = None,
  port: int | None = None,
) -> str:
  if explicit_url:
    return explicit_url.rstrip("/")
  cfg = load_server_config(config_dir)
  probe_host = _probe_host_for(host or cfg.server.host)
  resolved_port = int(port or cfg.server.port)
  return f"http://{probe_host}:{resolved_port}"


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


def _request_json(
  url: str,
  *,
  method: str = "GET",
  payload: dict[str, Any] | None = None,
  timeout: float = 10.0,
) -> Any:
  """Send an HTTP JSON request and return the decoded response body."""
  data = (
    json.dumps(payload, ensure_ascii=False).encode("utf-8")
    if payload is not None
    else None
  )
  headers = {"Content-Type": "application/json"} if payload is not None else {}
  req = urllib.request.Request(url, data=data, headers=headers, method=method)
  with urllib.request.urlopen(req, timeout=timeout) as resp:
    return json.loads(resp.read().decode("utf-8"))


def _extract_register_params(args: argparse.Namespace) -> tuple[str | None, dict[str, Any]]:
  """Parse run mode and default/shortcut parameters from ``register`` CLI arguments."""
  params: dict[str, Any] = {}
  valid_param_keys = set(ShortcutParamsModel.model_fields)

  raw_json = getattr(args, "params", None)
  mode: str | None = getattr(args, "mode", None)
  if raw_json is not None:
    try:
      loaded = json.loads(raw_json)
    except json.JSONDecodeError as exc:
      raise ValueError(f"Invalid JSON for --params: {exc}") from exc
    if not isinstance(loaded, dict):
      raise ValueError("--params must be a JSON object.")
    if "mode" in loaded:
      raw_mode = loaded.pop("mode")
      if mode is None and raw_mode is not None:
        mode = str(raw_mode)
    unknown = sorted(set(loaded) - valid_param_keys)
    if unknown:
      raise ValueError(f"Unknown parameter key(s) in --params: {unknown}.")
    params.update(loaded)

  if mode is not None and mode not in VALID_RUN_MODES:
    raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_RUN_MODES}.")

  if getattr(args, "objs", None):
    objs_str = " ".join(tok.strip() for tok in args.objs if tok.strip())
    if objs_str:
      params["objs"] = objs_str

  str_fields = (
    "begin",
    "end",
    "select_date",
    "benchmark",
    "cash",
    "fee",
    "exec_mode",
    "holdout",
    "msd_host",
    "fin_fields",
    "extra_args",
    "factor_periods",
    "factor_entry",
    "factor_winsorize",
    "factor_standardize",
    "factor_neutralize",
    "factor_industry_category",
  )
  for key in str_fields:
    val = getattr(args, key, None)
    if val is not None:
      params[key] = str(val)

  num_as_str_fields = (
    "warmup_bars",
    "fin_lookback_days",
    "factor_primary_period",
    "factor_quantiles",
    "factor_min_listed_days",
    "factor_cost_bps",
    "factor_min_obs",
  )
  for key in num_as_str_fields:
    val = getattr(args, key, None)
    if val is not None:
      params[key] = str(val)

  bool_fields = (
    "use_internal_selector",
    "fin_only_year",
    "factor_exclude_limit",
    "factor_by_industry",
    "factor_causality_check",
  )
  for key in bool_fields:
    val = getattr(args, key, None)
    if val is not None:
      params[key] = bool(val)

  factor_direction = getattr(args, "factor_direction", None)
  if factor_direction is not None:
    params["factor_direction"] = int(factor_direction)

  raw_kv = getattr(args, "args", None)
  if raw_kv:
    kv_dict = dict(params.get("strategy_args") or {})
    for tok in raw_kv:
      if "=" not in tok:
        raise ValueError(f"Invalid -a/--args token '{tok}': expected key=value format.")
      k, v = tok.split("=", 1)
      k = k.strip()
      if not k:
        raise ValueError(f"Invalid -a/--args token '{tok}': key cannot be empty.")
      kv_dict[k] = v.strip()
    params["strategy_args"] = kv_dict

  return mode, params


def _cmd_register(args: argparse.Namespace) -> int:
  try:
    mode, params = _extract_register_params(args)
  except ValueError as exc:
    print(f"Error: {exc}", file=sys.stderr)
    return 1

  base_url = _resolve_server_url(args.url, args.config_dir)
  abs_path = os.path.realpath(os.path.abspath(args.path))
  has_params = bool(params) or (mode is not None)

  try:
    if not has_params:
      payload: dict[str, Any] = {
        "path": abs_path,
        "name": args.name,
        "notes": args.notes,
      }
      rec = _request_json(f"{base_url}/api/strategies", method="POST", payload=payload)
      strat_id = rec.get("id") if isinstance(rec, dict) else None
      if not strat_id:
        print(f"Server response missing strategy id: {rec}", file=sys.stderr)
        return 1
      print(f"{base_url}/strategies/{strat_id}")
      return 0

    strategies = _request_json(f"{base_url}/api/strategies", method="GET")
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
      payload = {
        "path": abs_path,
        "name": args.name,
        "notes": args.notes,
        "default_params": params,
      }
      if mode is not None:
        payload["default_mode"] = mode
      rec = _request_json(f"{base_url}/api/strategies", method="POST", payload=payload)
      strat_id = rec.get("id") if isinstance(rec, dict) else None
      if not strat_id:
        print(f"Server response missing strategy id: {rec}", file=sys.stderr)
        return 1
      print(f"{base_url}/strategies/{strat_id}")
      return 0

    sc_name = (args.name or "").strip()
    sc_notes = (args.notes or "").strip()
    if not sc_name or not sc_notes:
      print(
        f"Error: Strategy '{abs_path}' is already registered. "
        "Creating a shortcut with parameters requires both --name and --notes.",
        file=sys.stderr,
      )
      return 1

    existing_sc = next(
      (
        sc
        for sc in (matched.get("shortcuts") or [])
        if str(sc.get("name") or "").strip() == sc_name
      ),
      None,
    )
    if existing_sc is not None:
      sc_payload: dict[str, Any] = {
        "name": sc_name,
        "notes": sc_notes,
        "mode": mode or existing_sc.get("mode") or matched["default_mode"],
        "params": params,
      }
      sc_rec = _request_json(
        f"{base_url}/api/shortcuts/{existing_sc['id']}",
        method="PUT",
        payload=sc_payload,
      )
    else:
      sc_payload = {
        "name": sc_name,
        "notes": sc_notes,
        "mode": mode or matched["default_mode"],
        "params": params,
      }
      sc_rec = _request_json(
        f"{base_url}/api/strategies/{matched['id']}/shortcuts",
        method="POST",
        payload=sc_payload,
      )
    sc_id = sc_rec.get("id") if isinstance(sc_rec, dict) else None
    if not sc_id:
      print(f"Server response missing shortcut id: {sc_rec}", file=sys.stderr)
      return 1
    print(f"{base_url}/strategies/{matched['id']}/shortcuts/{sc_id}")
    return 0
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


def _cmd_down(args: argparse.Namespace) -> int:
  base_url = _resolve_server_url(
    args.url,
    args.config_dir,
    host=getattr(args, "host", None),
    port=getattr(args, "port", None),
  )
  health_url = f"{base_url}/api/health"
  try:
    with urllib.request.urlopen(health_url, timeout=2.0) as resp:
      body = resp.read().decode("utf-8", errors="replace").strip()
      if resp.status != 200 or not body.startswith("OK, msd-backtest"):
        print(
          f"Error: Service at {base_url} is not mbt.server.",
          file=sys.stderr,
        )
        return 1
  except urllib.error.HTTPError as exc:
    print(
      f"Error: Service at {base_url} is not mbt.server (HTTP {exc.code}).",
      file=sys.stderr,
    )
    return 1
  except (urllib.error.URLError, OSError):
    print(f"mbt.server is not running at {base_url}.")
    return 0

  req = urllib.request.Request(
    f"{base_url}/api/shutdown",
    data=b"",
    method="POST",
  )
  try:
    with urllib.request.urlopen(req, timeout=5.0) as resp:
      if resp.status != 200:
        print(f"Error: Shutdown request failed with HTTP {resp.status}.", file=sys.stderr)
        return 1
  except urllib.error.HTTPError as exc:
    print(f"Shutdown failed ({exc.code}): {_format_http_error(exc)}", file=sys.stderr)
    return 1
  except (urllib.error.URLError, OSError) as exc:
    print(f"Failed to send shutdown request to {base_url}: {exc}", file=sys.stderr)
    return 1

  deadline = time.monotonic() + 10.0
  while time.monotonic() < deadline:
    try:
      with urllib.request.urlopen(health_url, timeout=0.5):
        pass
    except (urllib.error.URLError, OSError):
      print(f"Stopped mbt.server at {base_url}.")
      return 0
    time.sleep(0.1)

  print(
    f"Error: Timed out waiting for mbt.server at {base_url} to stop.",
    file=sys.stderr,
  )
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
  p_reg.add_argument("--name", default=None, help="Display name (required with --notes when creating a shortcut)")
  p_reg.add_argument("--notes", default=None, help="Notes/description (required with --name when creating a shortcut)")
  p_reg.add_argument("--url", default=None, help="Override server URL (default from server.json)")
  p_reg.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")
  p_reg.add_argument(
    "--mode",
    choices=list(VALID_RUN_MODES),
    default=None,
    help="Execution mode (select, timing, portfolio, factor)",
  )
  p_reg.add_argument("-s", "--objs", nargs="+", action="extend", default=None, help="Stock codes, pool name, or selector path")
  p_reg.add_argument(
    "--use-internal-selector",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Use embedded Selector class in timing/portfolio mode",
  )
  p_reg.add_argument("-b", "--begin", default=None, help="Start date (YYYY-MM-DD)")
  p_reg.add_argument("-e", "--end", default=None, help="End date (YYYY-MM-DD)")
  p_reg.add_argument("-d", "--select-date", default=None, help="Selection date (YYYY-MM-DD) for select mode")
  p_reg.add_argument("-B", "--benchmark", default=None, help="Benchmark index code")
  p_reg.add_argument("-c", "--cash", default=None, help="Initial cash")
  p_reg.add_argument("-f", "--fee", default=None, help="Fee config path")
  p_reg.add_argument("--exec-mode", choices=["close", "next_open"], default=None, help="Execution mode ('close' or 'next_open')")
  p_reg.add_argument("--warmup-bars", type=int, default=None, help="Warmup bars before start date")
  p_reg.add_argument("--holdout", default=None, help="Holdout split date or duration")
  p_reg.add_argument("-m", "--msd-host", default=None, help="MSD server URL")
  p_reg.add_argument("--fin-fields", default=None, help="Comma-separated financial fields")
  p_reg.add_argument(
    "--fin-only-year",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Only query annual financial reports",
  )
  p_reg.add_argument("--fin-lookback-days", type=int, default=None, help="Financial lookback window in days")
  p_reg.add_argument("-a", "--args", nargs="*", action="extend", default=None, help="Strategy/Selector/Factor arguments in key=value format")
  p_reg.add_argument("--extra-args", default=None, help="Extra key=value arguments string")
  p_reg.add_argument("--factor-periods", default=None, help="Comma-separated forward return periods")
  p_reg.add_argument("--factor-primary-period", type=int, default=None, help="Primary holding period")
  p_reg.add_argument("--factor-quantiles", type=int, default=None, help="Number of quantile buckets")
  p_reg.add_argument("--factor-entry", choices=["next_open", "open_close", "close"], default=None, help="Forward return entry timing")
  p_reg.add_argument(
    "--factor-exclude-limit",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Exclude limit-up bars from entry mask",
  )
  p_reg.add_argument("--factor-min-listed-days", type=int, default=None, help="Minimum listed trading days")
  p_reg.add_argument("--factor-winsorize", choices=["none", "mad", "sigma", "percentile"], default=None, help="Winsorization method")
  p_reg.add_argument("--factor-standardize", choices=["none", "zscore", "rank"], default=None, help="Standardization method")
  p_reg.add_argument("--factor-neutralize", default=None, help="Neutralization methods")
  p_reg.add_argument("--factor-cost-bps", type=float, default=None, help="Transaction cost in bps")
  p_reg.add_argument("--factor-min-obs", type=int, default=None, help="Minimum cross-sectional observations")
  p_reg.add_argument(
    "--factor-by-industry",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Evaluate factor by industry",
  )
  p_reg.add_argument("--factor-industry-category", default=None, help="Industry classification category")
  p_reg.add_argument(
    "--factor-causality-check",
    action=argparse.BooleanOptionalAction,
    default=None,
    help="Enable causality check",
  )
  p_reg.add_argument("--factor-direction", type=int, choices=[-1, 1], default=None, help="Factor direction (-1 or 1)")
  p_reg.add_argument("--params", default=None, help="JSON object of default/shortcut parameters (merged with CLI flags)")

  p_link = sub.add_parser("link", help="Print the Web UI URL for a registered strategy file path")
  p_link.add_argument("path", help="Path to .py strategy/selector/factor or .txt/.json expression file")
  p_link.add_argument("--url", default=None, help="Override server URL (default from server.json)")
  p_link.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")

  p_stat = sub.add_parser("status", help="Check whether mbt.server is running")
  p_stat.add_argument("--url", default=None, help="Override server URL")
  p_stat.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")

  p_down = sub.add_parser("down", help="Stop the running mbt.server instance")
  p_down.add_argument("--url", default=None, help="Override server URL")
  p_down.add_argument("--host", default=argparse.SUPPRESS, help="Override server host from server.json")
  p_down.add_argument("--port", type=int, default=argparse.SUPPRESS, help="Override server port from server.json")
  p_down.add_argument("--config-dir", default=argparse.SUPPRESS, help="Override OS config directory")

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
  if args.command == "down":
    return _cmd_down(args)
  return _cmd_serve(args)


if __name__ == "__main__":
  sys.exit(main())
