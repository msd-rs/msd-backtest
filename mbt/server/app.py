# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""FastAPI application factory for ``mbt.server``."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from mbt.server import get_version
from mbt.server.config import (
  ServerConfig,
  ServerListenConfig,
  SystemDefaults,
  get_config_file,
  get_data_dir,
  get_db_path,
  get_runs_dir,
  load_server_config,
  save_server_config,
)
from mbt.server.db import StateStore
from mbt.server.runner import RunConflictError, RunManager
from mbt.server.schemas import (
  RunCreateRequest,
  RunSummaryResponse,
  ShortcutCreateRequest,
  ShortcutDuplicateRequest,
  ShortcutResponse,
  ShortcutUpdateRequest,
  StrategyRegisterRequest,
  StrategyResponse,
  StrategyUpdateRequest,
  SystemConfigResponse,
  SystemConfigUpdateRequest,
)
from mbt.server.storage import (
  export_run_data,
  query_curve,
  query_table,
  read_manifest,
)

UI_DIST_DIR = Path(__file__).resolve().parent / "ui" / "dist"


def _fallback_html(version: str) -> str:
  return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>MSD Backtest Server v{version}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 64px auto; padding: 0 24px; color: #0f172a; line-height: 1.6; }}
    code {{ background: #f1f5f9; padding: 2px 6px; border-radius: 4px; }}
    a {{ color: #2563eb; }}
  </style>
</head>
<body>
  <h1>MSD Backtest Server (<code>{version}</code>)</h1>
  <p>Web UI 静态资源尚未构建（<code>mbt/server/ui/dist/index.html</code> 不存在）。</p>
  <p>请运行 <code>cd mbt/server/ui && pnpm install && pnpm build</code> 构建前端界面，或访问 API 文档：</p>
  <ul>
    <li><a href="/api/docs">Swagger / OpenAPI Docs (/api/docs)</a></li>
    <li><a href="/api/openapi.json">OpenAPI JSON (/api/openapi.json)</a></li>
    <li><a href="/api/health">Health Check (/api/health)</a></li>
  </ul>
</body>
</html>"""


def create_app(
  config_dir: Path | str | None = None,
  data_dir: Path | str | None = None,
) -> FastAPI:
  """Create and configure the ``mbt.server`` FastAPI instance."""
  cfg_dir_path = Path(config_dir) if config_dir is not None else None
  data_dir_path = Path(data_dir) if data_dir is not None else None

  resolved_cfg_file = get_config_file(cfg_dir_path)
  resolved_data_dir = get_data_dir(data_dir_path)
  db_path = get_db_path(resolved_data_dir)
  runs_dir = get_runs_dir(resolved_data_dir)

  cfg = load_server_config(cfg_dir_path)
  store = StateStore(db_path)
  store.mark_stale_running_runs()

  cfg_holder: dict[str, ServerConfig] = {"cfg": cfg}
  run_manager = RunManager(
    store=store,
    defaults_getter=lambda: cfg_holder["cfg"].defaults,
    runs_dir=runs_dir,
  )

  @asynccontextmanager
  async def lifespan(_app: FastAPI):
    try:
      yield
    finally:
      run_manager.shutdown()
      store.close()

  version = get_version()
  app = FastAPI(
    title="MSD Backtest Server",
    version=version,
    openapi_url="/api/openapi.json",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
  )

  app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
  )

  app.state.store = store
  app.state.run_manager = run_manager
  app.state.config_dir = cfg_dir_path
  app.state.data_dir = resolved_data_dir

  # ------------------------------------------------------------------
  # Health & System Configuration
  # ------------------------------------------------------------------

  @app.get("/api/health", response_class=PlainTextResponse, tags=["system"])
  async def health_check() -> str:
    """Return service health status and installed msd-backtest version."""
    return f"OK, msd-backtest {version}"

  @app.get("/api/config", response_model=SystemConfigResponse, tags=["system"])
  async def get_system_config() -> dict[str, Any]:
    """Return current server listen settings and system-level run defaults."""
    cur = cfg_holder["cfg"]
    return {
      "config_file": str(resolved_cfg_file),
      "data_dir": str(resolved_data_dir),
      "server": cur.server.to_dict(),
      "defaults": cur.defaults.to_dict(),
    }

  @app.put("/api/config", response_model=SystemConfigResponse, tags=["system"])
  async def update_system_config(req: SystemConfigUpdateRequest) -> dict[str, Any]:
    """Update and persist server configuration and system-level run defaults."""
    cur = cfg_holder["cfg"]
    try:
      new_listen = (
        ServerListenConfig.from_dict(req.server.model_dump())
        if req.server is not None
        else cur.server
      )
      new_defaults = SystemDefaults.from_dict(req.defaults.model_dump())
      updated_cfg = ServerConfig(
        version=cur.version,
        server=new_listen,
        defaults=new_defaults,
      )
      save_server_config(updated_cfg, config_dir=cfg_dir_path)
      cfg_holder["cfg"] = updated_cfg
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {
      "config_file": str(resolved_cfg_file),
      "data_dir": str(resolved_data_dir),
      "server": updated_cfg.server.to_dict(),
      "defaults": updated_cfg.defaults.to_dict(),
    }

  # ------------------------------------------------------------------
  # Strategy Registry
  # ------------------------------------------------------------------

  @app.get("/api/strategies", response_model=list[StrategyResponse], tags=["strategies"])
  async def list_strategies() -> list[dict[str, Any]]:
    """List all registered strategies along with their shortcuts and latest runs."""
    return store.list_strategies()

  @app.post("/api/strategies", response_model=StrategyResponse, tags=["strategies"])
  async def register_strategy(req: StrategyRegisterRequest) -> dict[str, Any]:
    """Register a strategy/factor file by path, deduplicating idempotently on canonical realpath."""
    try:
      rec, created = store.upsert_strategy(req.path, name=req.name, notes=req.notes)
    except FileNotFoundError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc
    rec["created"] = created
    return rec

  @app.get("/api/strategies/{strategy_id}", response_model=StrategyResponse, tags=["strategies"])
  async def get_strategy(strategy_id: str) -> dict[str, Any]:
    """Fetch a single strategy record by ID."""
    rec = store.get_strategy(strategy_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Strategy '{strategy_id}' not found.")
    return rec

  @app.patch("/api/strategies/{strategy_id}", response_model=StrategyResponse, tags=["strategies"])
  async def update_strategy(strategy_id: str, req: StrategyUpdateRequest) -> dict[str, Any]:
    """Rename or update notes on an existing strategy."""
    try:
      return store.update_strategy(strategy_id, name=req.name, notes=req.notes)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.post("/api/strategies/{strategy_id}/refresh", response_model=StrategyResponse, tags=["strategies"])
  async def refresh_strategy(strategy_id: str) -> dict[str, Any]:
    """Re-parse a strategy file on disk to update parameter metadata."""
    try:
      return store.refresh_strategy(strategy_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.delete("/api/strategies/{strategy_id}", tags=["strategies"])
  async def delete_strategy(strategy_id: str) -> dict[str, Any]:
    """Delete a strategy, its shortcuts, and any stored run artifacts."""
    try:
      obsolete_dirs = store.delete_strategy(strategy_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    run_manager.cleanup_dirs(obsolete_dirs)
    return {"deleted": True, "id": strategy_id}

  # ------------------------------------------------------------------
  # Shortcut CRUD
  # ------------------------------------------------------------------

  @app.get(
    "/api/strategies/{strategy_id}/shortcuts",
    response_model=list[ShortcutResponse],
    tags=["shortcuts"],
  )
  async def list_shortcuts(strategy_id: str) -> list[dict[str, Any]]:
    """List all execution shortcuts for a strategy."""
    try:
      return store.list_shortcuts(strategy_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc

  @app.post(
    "/api/strategies/{strategy_id}/shortcuts",
    response_model=ShortcutResponse,
    status_code=201,
    tags=["shortcuts"],
  )
  async def create_shortcut(strategy_id: str, req: ShortcutCreateRequest) -> dict[str, Any]:
    """Create a parameter preset shortcut under ``strategy_id``."""
    try:
      return store.create_shortcut(
        strategy_id,
        name=req.name,
        mode=req.mode,
        params=req.params,
      )
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.get("/api/shortcuts/{shortcut_id}", response_model=ShortcutResponse, tags=["shortcuts"])
  async def get_shortcut(shortcut_id: str) -> dict[str, Any]:
    """Fetch a single shortcut by ID."""
    sc = store.get_shortcut(shortcut_id)
    if sc is None:
      raise HTTPException(status_code=404, detail=f"Shortcut '{shortcut_id}' not found.")
    return sc

  @app.put("/api/shortcuts/{shortcut_id}", response_model=ShortcutResponse, tags=["shortcuts"])
  @app.patch("/api/shortcuts/{shortcut_id}", response_model=ShortcutResponse, tags=["shortcuts"])
  async def update_shortcut(shortcut_id: str, req: ShortcutUpdateRequest) -> dict[str, Any]:
    """Update a shortcut's name, mode, or saved parameters."""
    try:
      return store.update_shortcut(
        shortcut_id,
        name=req.name,
        mode=req.mode,
        params=req.params,
      )
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.post(
    "/api/shortcuts/{shortcut_id}/duplicate",
    response_model=ShortcutResponse,
    status_code=201,
    tags=["shortcuts"],
  )
  async def duplicate_shortcut(
    shortcut_id: str,
    req: ShortcutDuplicateRequest | None = None,
  ) -> dict[str, Any]:
    """Clone an existing shortcut under the same strategy."""
    try:
      return store.duplicate_shortcut(shortcut_id, name=req.name if req else None)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc

  @app.delete("/api/shortcuts/{shortcut_id}", tags=["shortcuts"])
  async def delete_shortcut(shortcut_id: str) -> dict[str, Any]:
    """Delete a shortcut and clean up its run artifacts."""
    try:
      obsolete_dirs = store.delete_shortcut(shortcut_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    run_manager.cleanup_dirs(obsolete_dirs)
    return {"deleted": True, "id": shortcut_id}

  # ------------------------------------------------------------------
  # Execution & On-Demand Result Query APIs
  # ------------------------------------------------------------------

  @app.post("/api/runs", response_model=RunSummaryResponse, status_code=202, tags=["runs"])
  async def start_run(req: RunCreateRequest) -> dict[str, Any]:
    """Start an asynchronous subprocess run for a shortcut or strategy default params."""
    try:
      return run_manager.start_run(
        strategy_id=req.strategy_id,
        shortcut_id=req.shortcut_id,
        mode=req.mode,
        overrides=req.overrides,
      )
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FileNotFoundError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RunConflictError as exc:
      raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.get("/api/runs/{run_id}", response_model=RunSummaryResponse, tags=["runs"])
  async def get_run(run_id: str) -> dict[str, Any]:
    """Query status and lightweight summary for a run."""
    rec = store.get_run(run_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    return rec

  @app.post("/api/runs/{run_id}/cancel", response_model=RunSummaryResponse, tags=["runs"])
  async def cancel_run(run_id: str) -> dict[str, Any]:
    """Terminate an active worker subprocess."""
    try:
      return run_manager.cancel_run(run_id)
    except KeyError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc

  @app.get("/api/runs/{run_id}/logs", tags=["runs"])
  async def get_run_logs(
    run_id: str,
    stream: bool = Query(default=False),
  ) -> Response:
    """Return full run log text or stream real-time SSE log events."""
    rec = store.get_run(run_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    if not stream:
      return PlainTextResponse(run_manager.read_logs(run_id))

    loop = asyncio.get_running_loop()
    q = run_manager.subscribe_async(run_id, loop)

    async def _sse_generator():
      try:
        existing_text = run_manager.read_logs(run_id)
        if existing_text:
          for line in existing_text.splitlines(keepends=True):
            yield f"data: {line.rstrip(chr(10))}\n\n"
        latest_rec = store.get_run(run_id)
        if latest_rec is not None and latest_rec["status"] != "running":
          yield f"event: done\ndata: {latest_rec['status']}\n\n"
          return
        while True:
          item = await q.get()
          if item is None:
            done_rec = store.get_run(run_id)
            st = done_rec["status"] if done_rec else "finished"
            yield f"event: done\ndata: {st}\n\n"
            break
          yield f"data: {item.rstrip(chr(10))}\n\n"
      finally:
        run_manager.unsubscribe_async(run_id, q)

    return StreamingResponse(_sse_generator(), media_type="text/event-stream")

  @app.get("/api/runs/{run_id}/summary", tags=["runs"])
  async def get_run_summary(run_id: str) -> dict[str, Any]:
    """Return the compact ``manifest.json`` summary for a completed run."""
    rec = store.get_run(run_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
      return read_manifest(rec["result_dir"])
    except FileNotFoundError as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc

  @app.get("/api/runs/{run_id}/tables/{table_name}", tags=["runs"])
  async def get_run_table(
    run_id: str,
    table_name: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=100, ge=1, le=2000),
    sort_by: str | None = Query(default=None),
    sort_dir: str = Query(default="desc"),
    q: str | None = Query(default=None),
    factor: str | None = Query(default=None),
    period: int | None = Query(default=None),
  ) -> dict[str, Any]:
    """Query paginated, sorted, filtered table rows lazily from Parquet."""
    rec = store.get_run(run_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
      return query_table(
        rec["result_dir"],
        table_name,
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_dir=sort_dir,
        q=q,
        factor=factor,
        period=period,
      )
    except (KeyError, FileNotFoundError) as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.get("/api/runs/{run_id}/curve", tags=["runs"])
  async def get_run_curve(
    run_id: str,
    obj: str | None = Query(default=None),
    factor: str | None = Query(default=None),
    period: int | None = Query(default=None),
    start_idx: int = Query(default=0, ge=0),
    end_idx: int | None = Query(default=None, ge=0),
    max_points: int = Query(default=2000, ge=4, le=20000),
  ) -> dict[str, Any]:
    """Query viewport-downsampled time-series curves from Parquet."""
    rec = store.get_run(run_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
      return query_curve(
        rec["result_dir"],
        obj=obj,
        factor=factor,
        period=period,
        start_idx=start_idx,
        end_idx=end_idx,
        max_points=max_points,
      )
    except (KeyError, FileNotFoundError) as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

  @app.get("/api/runs/{run_id}/export", tags=["runs"])
  async def export_run(
    run_id: str,
    format: str = Query(default="csv"),
    table: str | None = Query(default=None),
  ) -> Response:
    """Export run tables/reports on demand as CSV, XLSX, JSON, or Parquet."""
    rec = store.get_run(run_id)
    if rec is None:
      raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    try:
      payload, media_type, filename = export_run_data(
        rec["result_dir"],
        fmt=format,
        table=table,
      )
    except (KeyError, FileNotFoundError) as exc:
      raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
      raise HTTPException(status_code=422, detail=str(exc)) from exc

    return Response(
      content=payload,
      media_type=media_type,
      headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )

  # ------------------------------------------------------------------
  # Web UI Static Mount & SPA Fallback
  # ------------------------------------------------------------------

  assets_dir = UI_DIST_DIR / "assets"
  if assets_dir.is_dir():
    app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="ui-assets")

  @app.get("/{full_path:path}", include_in_schema=False)
  async def serve_ui(full_path: str) -> Response:
    if full_path == "api" or full_path.startswith("api/"):
      return JSONResponse(status_code=404, content={"detail": f"Unknown API endpoint '/{full_path}'."})

    if UI_DIST_DIR.is_dir():
      candidate = (UI_DIST_DIR / full_path).resolve() if full_path else None
      dist_root = UI_DIST_DIR.resolve()
      if (
        candidate is not None
        and str(candidate).startswith(str(dist_root) + os.sep)
        and candidate.is_file()
      ):
        return FileResponse(candidate)
      index_file = UI_DIST_DIR / "index.html"
      if index_file.is_file():
        return FileResponse(index_file)

    return HTMLResponse(_fallback_html(version))

  return app
