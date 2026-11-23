# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Async subprocess execution manager with real-time log streaming and cleanup."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
from typing import Any, Callable
import uuid

from mbt.server.config import SystemDefaults
from mbt.server.db import StateStore
from mbt.server.storage import read_manifest
from mbt.shared.inspect_strategy import VALID_RUN_MODES


class RunConflictError(RuntimeError):
  """Raised when attempting to launch a duplicate run for an already-running target."""


class RunManager:
  """Manages isolated worker subprocesses, live log subscribers, and disk cleanup."""

  def __init__(
    self,
    store: StateStore,
    defaults_getter: Callable[[], SystemDefaults],
    runs_dir: Path,
  ):
    self.store = store
    self._get_defaults = defaults_getter
    self.runs_dir = Path(runs_dir)
    self.runs_dir.mkdir(parents=True, exist_ok=True)
    self._lock = threading.RLock()
    self._procs: dict[str, subprocess.Popen[str]] = {}
    self._cancelled: set[str] = set()
    self._subscribers: dict[str, list[tuple[asyncio.AbstractEventLoop, asyncio.Queue[str | None]]]] = {}
    self._sync_subscribers: dict[str, list[queue.Queue[str | None]]] = {}

  def resolve_spec(
    self,
    defaults: SystemDefaults,
    strategy: dict[str, Any],
    shortcut: dict[str, Any] | None = None,
    mode_override: str | None = None,
    overrides: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Merge system defaults <- shortcut params <- temporary draft overrides."""
    base = defaults.to_dict()
    sc_params = dict(shortcut.get("params") or {}) if shortcut else {}
    ovr = dict(overrides or {})

    mode = (
      mode_override
      or ovr.get("mode")
      or (shortcut.get("mode") if shortcut else None)
      or strategy.get("default_mode")
    )
    if mode not in VALID_RUN_MODES:
      raise ValueError(f"Unsupported run mode '{mode}'. Expected one of {VALID_RUN_MODES}.")

    spec: dict[str, Any] = {
      "strategy_path": strategy["path"],
      "mode": mode,
      "objs": "",
      "use_internal_selector": False,
      "begin": base["begin"],
      "end": base["end"],
      "select_date": base["select_date"],
      "benchmark": base["benchmark"],
      "cash": base["cash"],
      "fee": base["fee"],
      "exec_mode": base["exec_mode"],
      "warmup_bars": base["warmup_bars"],
      "holdout": base["holdout"],
      "msd_host": base["msd_host"],
      "fin_fields": base["fin_fields"],
      "fin_only_year": base["fin_only_year"],
      "fin_lookback_days": base["fin_lookback_days"],
      "strategy_args": {},
      "extra_args": "",
      "factor_periods": base["factor_periods"],
      "factor_primary_period": base["factor_primary_period"],
      "factor_quantiles": base["factor_quantiles"],
      "factor_entry": base["factor_entry"],
      "factor_exclude_limit": base["factor_exclude_limit"],
      "factor_min_listed_days": base["factor_min_listed_days"],
      "factor_winsorize": base["factor_winsorize"],
      "factor_standardize": base["factor_standardize"],
      "factor_neutralize": base["factor_neutralize"],
      "factor_cost_bps": base["factor_cost_bps"],
      "factor_min_obs": base["factor_min_obs"],
      "factor_by_industry": base["factor_by_industry"],
      "factor_industry_category": base["factor_industry_category"],
      "factor_causality_check": base["factor_causality_check"],
      "factor_direction": None,
    }

    def _apply_layer(layer: dict[str, Any], *, replace_strategy_args: bool = False) -> None:
      for k, v in layer.items():
        if k == "mode":
          continue
        if k == "strategy_args" and isinstance(v, dict):
          if replace_strategy_args:
            spec["strategy_args"] = dict(v)
          else:
            spec["strategy_args"] = {**spec["strategy_args"], **v}
          continue
        if k not in spec:
          continue
        if v is None:
          continue
        if isinstance(v, str) and v.strip() == "":
          continue
        spec[k] = v

    _apply_layer(sc_params, replace_strategy_args=False)
    _apply_layer(ovr, replace_strategy_args=("strategy_args" in ovr))

    # For direct strategy runs (no shortcut) on portfolio with embedded selector and empty objs
    meta = strategy.get("metadata") or {}
    if (
      shortcut is None
      and mode == "portfolio"
      and not str(spec.get("objs") or "").strip()
      and not spec.get("use_internal_selector")
      and meta.get("has_selector")
      and "use_internal_selector" not in ovr
    ):
      spec["use_internal_selector"] = True

    return spec

  def _broadcast_line(self, run_id: str, line: str | None) -> None:
    with self._lock:
      async_subs = list(self._subscribers.get(run_id, []))
      sync_subs = list(self._sync_subscribers.get(run_id, []))
    for loop, q in async_subs:
      try:
        loop.call_soon_threadsafe(q.put_nowait, line)
      except Exception:
        pass
    for sq in sync_subs:
      try:
        sq.put_nowait(line)
      except Exception:
        pass

  def start_run(
    self,
    *,
    strategy_id: str | None = None,
    shortcut_id: str | None = None,
    mode: str | None = None,
    overrides: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Resolve parameters and launch an isolated worker subprocess."""
    shortcut: dict[str, Any] | None = None
    if shortcut_id:
      shortcut = self.store.get_shortcut(shortcut_id)
      if shortcut is None:
        raise KeyError(f"Shortcut '{shortcut_id}' not found.")
      strategy = self.store.get_strategy(shortcut["strategy_id"])
      if strategy is None:
        raise KeyError(f"Parent strategy '{shortcut['strategy_id']}' not found.")
      target_type = "shortcut"
      target_id = shortcut["id"]
    elif strategy_id:
      strategy = self.store.get_strategy(strategy_id)
      if strategy is None:
        raise KeyError(f"Strategy '{strategy_id}' not found.")
      target_type = "strategy"
      target_id = strategy["id"]
    else:
      raise ValueError("Either 'shortcut_id' or 'strategy_id' must be specified.")

    if not os.path.isfile(strategy["path"]):
      raise FileNotFoundError(f"Strategy file no longer exists on disk: '{strategy['path']}'.")

    active = self.store.get_active_run(target_type, target_id)
    if active is not None:
      raise RunConflictError(
        f"A run ({active['id']}) is already active for {target_type} '{target_id}'."
      )

    defaults = self._get_defaults()
    spec = self.resolve_spec(
      defaults,
      strategy,
      shortcut=shortcut,
      mode_override=mode,
      overrides=overrides,
    )

    run_id = uuid.uuid4().hex[:16]
    run_dir = self.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    spec_path = run_dir / "spec.json"
    log_path = run_dir / "run.log"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")

    run_rec = self.store.create_run(
      run_id=run_id,
      target_type=target_type,
      target_id=target_id,
      strategy_id=strategy["id"],
      shortcut_id=shortcut["id"] if shortcut else None,
      mode=spec["mode"],
      resolved_params=spec,
      result_dir=str(run_dir),
    )

    cmd = [
      sys.executable,
      "-u",
      "-m",
      "mbt.server.worker",
      "--spec",
      str(spec_path),
      "--out-dir",
      str(run_dir),
    ]
    t_start = time.perf_counter()
    proc = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      encoding="utf-8",
      errors="replace",
      bufsize=1,
    )
    with self._lock:
      self._procs[run_id] = proc
    self.store.update_run_pid(run_id, proc.pid)

    def _watch() -> None:
      tail_lines: list[str] = []
      try:
        with open(log_path, "w", encoding="utf-8") as lf:
          lf.write(f"$ {' '.join(cmd)}\n")
          lf.flush()
          self._broadcast_line(run_id, f"$ {' '.join(cmd)}\n")
          if proc.stdout is not None:
            for line in proc.stdout:
              lf.write(line)
              lf.flush()
              tail_lines.append(line.rstrip("\r\n"))
              if len(tail_lines) > 40:
                tail_lines.pop(0)
              self._broadcast_line(run_id, line)
        rc = proc.wait()
      except Exception as exc:
        rc = -1
        tail_lines.append(f"[server] watcher error: {exc}")
      finally:
        with self._lock:
          self._procs.pop(run_id, None)
          was_cancelled = run_id in self._cancelled
          self._cancelled.discard(run_id)

      elapsed = round(time.perf_counter() - t_start, 4)
      manifest_file = run_dir / "manifest.json"
      summary_payload: dict[str, Any] | None = None
      error_msg = ""

      if was_cancelled:
        status = "cancelled"
        error_msg = "Run cancelled by user."
      elif rc == 0 and manifest_file.is_file():
        status = "completed"
        try:
          manifest = read_manifest(run_dir)
          summary_payload = manifest.get("summary")
          elapsed = float(manifest.get("elapsed", elapsed))
        except Exception as exc:
          status = "failed"
          error_msg = f"Failed to parse run manifest: {exc}"
      else:
        status = "failed"
        tail_snippet = "\n".join(tail_lines[-10:]).strip()
        error_msg = (
          f"Worker exited with code {rc}."
          + (f"\n{tail_snippet}" if tail_snippet else "")
        )

      obsolete_dirs = self.store.finish_run(
        run_id,
        status=status,
        summary=summary_payload,
        error=error_msg,
        elapsed=elapsed,
      )
      for old_dir in obsolete_dirs:
        if old_dir and os.path.abspath(old_dir) != os.path.abspath(str(run_dir)):
          shutil.rmtree(old_dir, ignore_errors=True)

      self._broadcast_line(run_id, None)
      with self._lock:
        self._subscribers.pop(run_id, None)
        self._sync_subscribers.pop(run_id, None)

    threading.Thread(target=_watch, name=f"mbt-run-{run_id[:8]}", daemon=True).start()
    updated = self.store.get_run(run_id)
    return updated or run_rec

  def cancel_run(self, run_id: str) -> dict[str, Any]:
    """Terminate a running worker subprocess."""
    run_rec = self.store.get_run(run_id)
    if run_rec is None:
      raise KeyError(f"Run '{run_id}' not found.")

    with self._lock:
      proc = self._procs.get(run_id)
      if proc is not None and proc.poll() is None:
        self._cancelled.add(run_id)
        try:
          proc.terminate()
          try:
            proc.wait(timeout=2.0)
          except subprocess.TimeoutExpired:
            proc.kill()
        except Exception:
          pass
    return self.store.get_run(run_id) or run_rec

  def read_logs(self, run_id: str) -> str:
    """Return accumulated log text for ``run_id``."""
    run_rec = self.store.get_run(run_id)
    if run_rec is None:
      raise KeyError(f"Run '{run_id}' not found.")
    log_file = Path(run_rec["result_dir"]) / "run.log"
    if not log_file.is_file():
      return ""
    return log_file.read_text(encoding="utf-8", errors="replace")

  def subscribe_async(
    self, run_id: str, loop: asyncio.AbstractEventLoop
  ) -> asyncio.Queue[str | None]:
    """Subscribe an async queue to live log lines for ``run_id``."""
    q: asyncio.Queue[str | None] = asyncio.Queue()
    with self._lock:
      self._subscribers.setdefault(run_id, []).append((loop, q))
    return q

  def unsubscribe_async(
    self, run_id: str, q: asyncio.Queue[str | None]
  ) -> None:
    """Remove an async log subscriber queue."""
    with self._lock:
      items = self._subscribers.get(run_id)
      if items:
        self._subscribers[run_id] = [pair for pair in items if pair[1] is not q]

  def cleanup_dirs(self, dirs: list[str]) -> None:
    """Delete run artifact directories safely from disk."""
    for d in dirs:
      if d and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)

  def shutdown(self) -> None:
    """Terminate any active worker subprocesses on server shutdown."""
    with self._lock:
      active_ids = list(self._procs.keys())
    for rid in active_ids:
      try:
        self.cancel_run(rid)
      except Exception:
        pass
