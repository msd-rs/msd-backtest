# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""SQLite state store for strategies, shortcuts, and latest run metadata."""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import sqlite3
import threading
from typing import Any
import uuid

from mbt.shared.inspect_strategy import VALID_RUN_MODES, inspect_strategy_file


def _utc_now() -> str:
  return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


class StateStore:
  """Thread-safe SQLite repository managing strategies, shortcuts, and latest runs."""

  def __init__(self, db_path: Path | str):
    self.db_path = Path(db_path)
    self.db_path.parent.mkdir(parents=True, exist_ok=True)
    self._lock = threading.RLock()
    self._conn = sqlite3.connect(
      str(self.db_path),
      timeout=30.0,
      check_same_thread=False,
    )
    self._conn.row_factory = sqlite3.Row
    self._init_schema()

  def _init_schema(self) -> None:
    with self._lock:
      self._conn.executescript(
        """
        PRAGMA journal_mode=WAL;
        PRAGMA foreign_keys=ON;

        CREATE TABLE IF NOT EXISTS strategies (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          path TEXT NOT NULL UNIQUE,
          default_mode TEXT NOT NULL,
          metadata_json TEXT NOT NULL,
          notes TEXT NOT NULL DEFAULT '',
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS shortcuts (
          id TEXT PRIMARY KEY,
          strategy_id TEXT NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
          name TEXT NOT NULL,
          mode TEXT NOT NULL,
          params_json TEXT NOT NULL,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS runs (
          id TEXT PRIMARY KEY,
          target_type TEXT NOT NULL,
          target_id TEXT NOT NULL,
          strategy_id TEXT NOT NULL REFERENCES strategies(id) ON DELETE CASCADE,
          shortcut_id TEXT REFERENCES shortcuts(id) ON DELETE CASCADE,
          mode TEXT NOT NULL,
          status TEXT NOT NULL,
          pid INTEGER,
          resolved_params_json TEXT NOT NULL,
          result_dir TEXT NOT NULL,
          summary_json TEXT,
          error TEXT NOT NULL DEFAULT '',
          started_at TEXT NOT NULL,
          finished_at TEXT,
          elapsed REAL,
          is_latest INTEGER NOT NULL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_shortcuts_strategy ON shortcuts(strategy_id);
        CREATE INDEX IF NOT EXISTS idx_runs_target_latest ON runs(target_type, target_id, is_latest);
        CREATE INDEX IF NOT EXISTS idx_runs_strategy ON runs(strategy_id);
        """
      )
      self._conn.commit()

  def close(self) -> None:
    """Close the underlying SQLite connection."""
    with self._lock:
      self._conn.close()

  def _row_to_run(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
      return None
    summary_raw = row["summary_json"]
    return {
      "id": row["id"],
      "target_type": row["target_type"],
      "target_id": row["target_id"],
      "strategy_id": row["strategy_id"],
      "shortcut_id": row["shortcut_id"],
      "mode": row["mode"],
      "status": row["status"],
      "pid": row["pid"],
      "resolved_params": json.loads(row["resolved_params_json"]),
      "result_dir": row["result_dir"],
      "summary": json.loads(summary_raw) if summary_raw else None,
      "error": row["error"] or "",
      "started_at": row["started_at"],
      "finished_at": row["finished_at"],
      "elapsed": row["elapsed"],
      "is_latest": bool(row["is_latest"]),
    }

  def _row_to_shortcut(
    self,
    row: sqlite3.Row,
    latest_run: dict[str, Any] | None = None,
    active_run: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    return {
      "id": row["id"],
      "strategy_id": row["strategy_id"],
      "name": row["name"],
      "mode": row["mode"],
      "params": json.loads(row["params_json"]),
      "created_at": row["created_at"],
      "updated_at": row["updated_at"],
      "latest_run": latest_run,
      "active_run": active_run,
    }

  def _row_to_strategy(
    self,
    row: sqlite3.Row,
    shortcuts: list[dict[str, Any]] | None = None,
    latest_run: dict[str, Any] | None = None,
    active_run: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    path_str = row["path"]
    return {
      "id": row["id"],
      "name": row["name"],
      "path": path_str,
      "default_mode": row["default_mode"],
      "metadata": json.loads(row["metadata_json"]),
      "notes": row["notes"] or "",
      "file_exists": os.path.isfile(path_str),
      "created_at": row["created_at"],
      "updated_at": row["updated_at"],
      "shortcuts": shortcuts if shortcuts is not None else [],
      "latest_run": latest_run,
      "active_run": active_run,
    }

  def _latest_and_active_maps(
    self,
  ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    latest_map: dict[tuple[str, str], dict[str, Any]] = {}
    active_map: dict[tuple[str, str], dict[str, Any]] = {}
    rows = self._conn.execute(
      "SELECT * FROM runs WHERE is_latest = 1 OR status = 'running' ORDER BY started_at DESC"
    ).fetchall()
    for row in rows:
      key = (str(row["target_type"]), str(row["target_id"]))
      run_dict = self._row_to_run(row)
      assert run_dict is not None
      if row["is_latest"] and key not in latest_map:
        latest_map[key] = run_dict
      if row["status"] == "running" and key not in active_map:
        active_map[key] = run_dict
    return latest_map, active_map

  def upsert_strategy(
    self,
    path: str | Path,
    name: str | None = None,
    notes: str | None = None,
  ) -> tuple[dict[str, Any], bool]:
    """Register or idempotently refresh a strategy by its canonical real path."""
    canonical = os.path.realpath(os.path.abspath(str(path)))
    meta = inspect_strategy_file(canonical)
    meta_json = json.dumps(meta.to_dict(), ensure_ascii=False)
    now = _utc_now()

    with self._lock:
      existing = self._conn.execute(
        "SELECT * FROM strategies WHERE path = ?",
        (canonical,),
      ).fetchone()
      if existing is not None:
        new_name = (
          name.strip()
          if (name is not None and name.strip())
          else existing["name"]
        )
        new_notes = notes if notes is not None else existing["notes"]
        self._conn.execute(
          """
          UPDATE strategies
          SET name = ?, default_mode = ?, metadata_json = ?, notes = ?, updated_at = ?
          WHERE id = ?
          """,
          (new_name, meta.default_mode, meta_json, new_notes, now, existing["id"]),
        )
        self._conn.commit()
        updated = self.get_strategy(existing["id"])
        assert updated is not None
        return updated, False

      sid = uuid.uuid4().hex[:12]
      default_name = (
        meta.docstring.strip()
        if meta.docstring and meta.docstring.strip()
        else Path(canonical).stem
      )
      strat_name = (name.strip() if name and name.strip() else default_name)
      self._conn.execute(
        """
        INSERT INTO strategies (id, name, path, default_mode, metadata_json, notes, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (sid, strat_name, canonical, meta.default_mode, meta_json, notes or "", now, now),
      )
      self._conn.commit()
      created_rec = self.get_strategy(sid)
      assert created_rec is not None
      return created_rec, True

  def list_strategies(self) -> list[dict[str, Any]]:
    """Return all registered strategies along with their shortcuts and latest run summaries."""
    with self._lock:
      strat_rows = self._conn.execute(
        "SELECT * FROM strategies ORDER BY updated_at DESC, created_at DESC"
      ).fetchall()
      sc_rows = self._conn.execute(
        "SELECT * FROM shortcuts ORDER BY created_at ASC"
      ).fetchall()
      latest_map, active_map = self._latest_and_active_maps()

      shortcuts_by_strat: dict[str, list[dict[str, Any]]] = {}
      for sc in sc_rows:
        sc_id = str(sc["id"])
        sc_dict = self._row_to_shortcut(
          sc,
          latest_run=latest_map.get(("shortcut", sc_id)),
          active_run=active_map.get(("shortcut", sc_id)),
        )
        shortcuts_by_strat.setdefault(str(sc["strategy_id"]), []).append(sc_dict)

      out: list[dict[str, Any]] = []
      for srow in strat_rows:
        sid = str(srow["id"])
        out.append(
          self._row_to_strategy(
            srow,
            shortcuts=shortcuts_by_strat.get(sid, []),
            latest_run=latest_map.get(("strategy", sid)),
            active_run=active_map.get(("strategy", sid)),
          )
        )
      return out

  def get_strategy(self, strategy_id: str) -> dict[str, Any] | None:
    """Return a single strategy by ID, or ``None`` if not found."""
    with self._lock:
      srow = self._conn.execute(
        "SELECT * FROM strategies WHERE id = ?",
        (strategy_id,),
      ).fetchone()
      if srow is None:
        return None
      latest_map, active_map = self._latest_and_active_maps()
      sc_rows = self._conn.execute(
        "SELECT * FROM shortcuts WHERE strategy_id = ? ORDER BY created_at ASC",
        (strategy_id,),
      ).fetchall()
      shortcuts = [
        self._row_to_shortcut(
          sc,
          latest_run=latest_map.get(("shortcut", str(sc["id"]))),
          active_run=active_map.get(("shortcut", str(sc["id"]))),
        )
        for sc in sc_rows
      ]
      return self._row_to_strategy(
        srow,
        shortcuts=shortcuts,
        latest_run=latest_map.get(("strategy", strategy_id)),
        active_run=active_map.get(("strategy", strategy_id)),
      )

  def update_strategy(
    self,
    strategy_id: str,
    *,
    name: str | None = None,
    notes: str | None = None,
  ) -> dict[str, Any]:
    """Update a strategy's display name or notes."""
    with self._lock:
      existing = self._conn.execute(
        "SELECT * FROM strategies WHERE id = ?",
        (strategy_id,),
      ).fetchone()
      if existing is None:
        raise KeyError(f"Strategy '{strategy_id}' not found.")
      new_name = existing["name"]
      if name is not None:
        stripped = name.strip()
        if not stripped:
          raise ValueError("Strategy name cannot be empty.")
        new_name = stripped
      new_notes = notes if notes is not None else existing["notes"]
      now = _utc_now()
      self._conn.execute(
        "UPDATE strategies SET name = ?, notes = ?, updated_at = ? WHERE id = ?",
        (new_name, new_notes, now, strategy_id),
      )
      self._conn.commit()
      updated = self.get_strategy(strategy_id)
      assert updated is not None
      return updated

  def refresh_strategy(self, strategy_id: str) -> dict[str, Any]:
    """Re-inspect a strategy file on disk and refresh its cached metadata."""
    with self._lock:
      existing = self._conn.execute(
        "SELECT * FROM strategies WHERE id = ?",
        (strategy_id,),
      ).fetchone()
      if existing is None:
        raise KeyError(f"Strategy '{strategy_id}' not found.")
      meta = inspect_strategy_file(existing["path"])
      meta_json = json.dumps(meta.to_dict(), ensure_ascii=False)
      now = _utc_now()
      self._conn.execute(
        """
        UPDATE strategies
        SET default_mode = ?, metadata_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (meta.default_mode, meta_json, now, strategy_id),
      )
      self._conn.commit()
      updated = self.get_strategy(strategy_id)
      assert updated is not None
      return updated

  def delete_strategy(self, strategy_id: str) -> list[str]:
    """Delete a strategy (cascading shortcuts and runs) and return obsolete result dirs."""
    with self._lock:
      existing = self._conn.execute(
        "SELECT id FROM strategies WHERE id = ?",
        (strategy_id,),
      ).fetchone()
      if existing is None:
        raise KeyError(f"Strategy '{strategy_id}' not found.")
      run_dirs = [
        str(r["result_dir"])
        for r in self._conn.execute(
          "SELECT result_dir FROM runs WHERE strategy_id = ?",
          (strategy_id,),
        ).fetchall()
        if r["result_dir"]
      ]
      self._conn.execute("DELETE FROM strategies WHERE id = ?", (strategy_id,))
      self._conn.commit()
      return run_dirs

  def list_shortcuts(self, strategy_id: str) -> list[dict[str, Any]]:
    """Return all shortcuts belonging to ``strategy_id``."""
    with self._lock:
      strat = self._conn.execute(
        "SELECT id FROM strategies WHERE id = ?",
        (strategy_id,),
      ).fetchone()
      if strat is None:
        raise KeyError(f"Strategy '{strategy_id}' not found.")
      latest_map, active_map = self._latest_and_active_maps()
      rows = self._conn.execute(
        "SELECT * FROM shortcuts WHERE strategy_id = ? ORDER BY created_at ASC",
        (strategy_id,),
      ).fetchall()
      return [
        self._row_to_shortcut(
          r,
          latest_run=latest_map.get(("shortcut", str(r["id"]))),
          active_run=active_map.get(("shortcut", str(r["id"]))),
        )
        for r in rows
      ]

  def get_shortcut(self, shortcut_id: str) -> dict[str, Any] | None:
    """Return a single shortcut by ID, or ``None`` if not found."""
    with self._lock:
      row = self._conn.execute(
        "SELECT * FROM shortcuts WHERE id = ?",
        (shortcut_id,),
      ).fetchone()
      if row is None:
        return None
      latest_map, active_map = self._latest_and_active_maps()
      return self._row_to_shortcut(
        row,
        latest_run=latest_map.get(("shortcut", shortcut_id)),
        active_run=active_map.get(("shortcut", shortcut_id)),
      )

  def create_shortcut(
    self,
    strategy_id: str,
    name: str,
    mode: str,
    params: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Create a new execution shortcut for an existing strategy."""
    if mode not in VALID_RUN_MODES:
      raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_RUN_MODES}.")
    clean_name = name.strip()
    if not clean_name:
      raise ValueError("Shortcut name cannot be empty.")
    params_dict = dict(params or {})
    now = _utc_now()
    sc_id = uuid.uuid4().hex[:12]

    with self._lock:
      strat = self._conn.execute(
        "SELECT id FROM strategies WHERE id = ?",
        (strategy_id,),
      ).fetchone()
      if strat is None:
        raise KeyError(f"Strategy '{strategy_id}' not found.")
      self._conn.execute(
        """
        INSERT INTO shortcuts (id, strategy_id, name, mode, params_json, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (sc_id, strategy_id, clean_name, mode, json.dumps(params_dict, ensure_ascii=False), now, now),
      )
      self._conn.commit()
      created = self.get_shortcut(sc_id)
      assert created is not None
      return created

  def update_shortcut(
    self,
    shortcut_id: str,
    *,
    name: str | None = None,
    mode: str | None = None,
    params: dict[str, Any] | None = None,
  ) -> dict[str, Any]:
    """Update a shortcut's name, mode, and/or parameter dictionary."""
    if mode is not None and mode not in VALID_RUN_MODES:
      raise ValueError(f"Unsupported mode '{mode}'. Expected one of {VALID_RUN_MODES}.")

    with self._lock:
      existing = self._conn.execute(
        "SELECT * FROM shortcuts WHERE id = ?",
        (shortcut_id,),
      ).fetchone()
      if existing is None:
        raise KeyError(f"Shortcut '{shortcut_id}' not found.")
      new_name = existing["name"]
      if name is not None:
        stripped = name.strip()
        if not stripped:
          raise ValueError("Shortcut name cannot be empty.")
        new_name = stripped
      new_mode = mode if mode is not None else existing["mode"]
      new_params_json = (
        json.dumps(dict(params), ensure_ascii=False)
        if params is not None
        else existing["params_json"]
      )
      now = _utc_now()
      self._conn.execute(
        """
        UPDATE shortcuts
        SET name = ?, mode = ?, params_json = ?, updated_at = ?
        WHERE id = ?
        """,
        (new_name, new_mode, new_params_json, now, shortcut_id),
      )
      self._conn.commit()
      updated = self.get_shortcut(shortcut_id)
      assert updated is not None
      return updated

  def duplicate_shortcut(
    self,
    shortcut_id: str,
    name: str | None = None,
  ) -> dict[str, Any]:
    """Clone an existing shortcut under the same strategy."""
    with self._lock:
      orig = self.get_shortcut(shortcut_id)
      if orig is None:
        raise KeyError(f"Shortcut '{shortcut_id}' not found.")
      copy_name = name.strip() if name and name.strip() else f"{orig['name']} (副本)"
      return self.create_shortcut(
        orig["strategy_id"],
        name=copy_name,
        mode=orig["mode"],
        params=orig["params"],
      )

  def delete_shortcut(self, shortcut_id: str) -> list[str]:
    """Delete a shortcut (cascading its runs) and return obsolete result dirs."""
    with self._lock:
      existing = self._conn.execute(
        "SELECT id FROM shortcuts WHERE id = ?",
        (shortcut_id,),
      ).fetchone()
      if existing is None:
        raise KeyError(f"Shortcut '{shortcut_id}' not found.")
      run_dirs = [
        str(r["result_dir"])
        for r in self._conn.execute(
          "SELECT result_dir FROM runs WHERE shortcut_id = ?",
          (shortcut_id,),
        ).fetchall()
        if r["result_dir"]
      ]
      self._conn.execute("DELETE FROM shortcuts WHERE id = ?", (shortcut_id,))
      self._conn.commit()
      return run_dirs

  def create_run(
    self,
    *,
    run_id: str,
    target_type: str,
    target_id: str,
    strategy_id: str,
    shortcut_id: str | None,
    mode: str,
    resolved_params: dict[str, Any],
    result_dir: str,
    pid: int | None = None,
  ) -> dict[str, Any]:
    """Insert a new active run record."""
    now = _utc_now()
    with self._lock:
      self._conn.execute(
        """
        INSERT INTO runs (
          id, target_type, target_id, strategy_id, shortcut_id, mode,
          status, pid, resolved_params_json, result_dir, started_at, is_latest
        )
        VALUES (?, ?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, 0)
        """,
        (
          run_id,
          target_type,
          target_id,
          strategy_id,
          shortcut_id,
          mode,
          pid,
          json.dumps(resolved_params, ensure_ascii=False),
          result_dir,
          now,
        ),
      )
      self._conn.commit()
      rec = self.get_run(run_id)
      assert rec is not None
      return rec

  def update_run_pid(self, run_id: str, pid: int | None) -> None:
    """Record the spawned OS process PID for an active run."""
    with self._lock:
      self._conn.execute("UPDATE runs SET pid = ? WHERE id = ?", (pid, run_id))
      self._conn.commit()

  def get_run(self, run_id: str) -> dict[str, Any] | None:
    """Return a single run record by ID, or ``None`` if not found."""
    with self._lock:
      row = self._conn.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
      return self._row_to_run(row)

  def get_active_run(self, target_type: str, target_id: str) -> dict[str, Any] | None:
    """Return currently running record for ``(target_type, target_id)``, if any."""
    with self._lock:
      row = self._conn.execute(
        """
        SELECT * FROM runs
        WHERE target_type = ? AND target_id = ? AND status = 'running'
        ORDER BY started_at DESC LIMIT 1
        """,
        (target_type, target_id),
      ).fetchone()
      return self._row_to_run(row)

  def get_latest_run(self, target_type: str, target_id: str) -> dict[str, Any] | None:
    """Return the latest finished run record for ``(target_type, target_id)``."""
    with self._lock:
      row = self._conn.execute(
        """
        SELECT * FROM runs
        WHERE target_type = ? AND target_id = ? AND is_latest = 1
        ORDER BY started_at DESC LIMIT 1
        """,
        (target_type, target_id),
      ).fetchone()
      return self._row_to_run(row)

  def finish_run(
    self,
    run_id: str,
    *,
    status: str,
    summary: dict[str, Any] | None = None,
    error: str = "",
    elapsed: float | None = None,
  ) -> list[str]:
    """Finalize a run as latest for its target and return superseded result dirs to delete."""
    now = _utc_now()
    summary_str = json.dumps(summary, ensure_ascii=False) if summary is not None else None
    with self._lock:
      cur_row = self._conn.execute(
        "SELECT target_type, target_id FROM runs WHERE id = ?",
        (run_id,),
      ).fetchone()
      if cur_row is None:
        return []
      t_type = str(cur_row["target_type"])
      t_id = str(cur_row["target_id"])

      old_rows = self._conn.execute(
        """
        SELECT id, result_dir FROM runs
        WHERE target_type = ? AND target_id = ? AND id != ? AND status != 'running'
        """,
        (t_type, t_id, run_id),
      ).fetchall()
      obsolete_dirs = [str(r["result_dir"]) for r in old_rows if r["result_dir"]]
      obsolete_ids = [str(r["id"]) for r in old_rows]

      self._conn.execute(
        """
        UPDATE runs
        SET status = ?, summary_json = ?, error = ?, finished_at = ?, elapsed = ?, is_latest = 1
        WHERE id = ?
        """,
        (status, summary_str, error, now, elapsed, run_id),
      )
      if obsolete_ids:
        placeholders = ",".join("?" for _ in obsolete_ids)
        self._conn.execute(f"DELETE FROM runs WHERE id IN ({placeholders})", obsolete_ids)
      self._conn.commit()
      return obsolete_dirs

  def mark_stale_running_runs(self) -> int:
    """Mark leftover ``running`` records from a crashed server as ``failed``."""
    now = _utc_now()
    with self._lock:
      cur = self._conn.execute(
        """
        UPDATE runs
        SET status = 'failed',
            error = CASE WHEN error = '' THEN 'Server restarted before process completed.' ELSE error END,
            finished_at = COALESCE(finished_at, ?)
        WHERE status = 'running'
        """,
        (now,),
      )
      self._conn.commit()
      return cur.rowcount
