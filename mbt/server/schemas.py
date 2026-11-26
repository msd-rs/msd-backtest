# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Pydantic request/response schemas for the ``mbt.server`` REST API."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field

RunModeLiteral = Literal["select", "timing", "portfolio", "factor"]


class ServerListenSchema(BaseModel):
  host: str = "127.0.0.1"
  port: int = Field(default=8765, ge=1, le=65535)


class SystemDefaultsSchema(BaseModel):
  msd_host: str = "http://localhost:50510"
  begin: str = "2023-01-01"
  end: str = "2025-12-31"
  select_date: str = ""
  benchmark: str = "SH000300"
  cash: float = Field(default=10000000.0, gt=0)
  fee: str = ""
  exec_mode: Literal["close", "next_open"] = "close"
  warmup_bars: int = Field(default=0, ge=0)
  holdout: str = ""
  fin_fields: str = ""
  fin_only_year: bool = False
  fin_lookback_days: int | None = None
  factor_periods: str = "1,5,10,20"
  factor_primary_period: int | None = None
  factor_quantiles: int = Field(default=5, ge=2)
  factor_entry: Literal["next_open", "open_close", "close"] = "next_open"
  factor_exclude_limit: bool = True
  factor_min_listed_days: int = Field(default=0, ge=0)
  factor_winsorize: Literal["none", "mad", "sigma", "percentile"] = "none"
  factor_standardize: Literal["none", "zscore", "rank"] = "none"
  factor_neutralize: str = "none"
  factor_cost_bps: float = 0.0
  factor_min_obs: int = Field(default=10, ge=2)
  factor_by_industry: bool = False
  factor_industry_category: str = "申万行业"
  factor_causality_check: bool = True


class SystemConfigResponse(BaseModel):
  config_file: str
  data_dir: str
  server: ServerListenSchema
  defaults: SystemDefaultsSchema


class SystemConfigUpdateRequest(BaseModel):
  server: ServerListenSchema | None = None
  defaults: SystemDefaultsSchema


class StrategyRegisterRequest(BaseModel):
  path: str = Field(..., min_length=1, description="Absolute or relative path to .py/.txt/.json strategy or factor file")
  name: str | None = None
  notes: str | None = None
  default_mode: RunModeLiteral | None = None
  default_params: dict[str, Any] | None = None


class StrategyUpdateRequest(BaseModel):
  name: str | None = None
  notes: str | None = None
  default_mode: RunModeLiteral | None = None
  default_params: dict[str, Any] | None = None


class ShortcutParamsModel(BaseModel):
  objs: str = ""
  use_internal_selector: bool = False
  begin: str = ""
  end: str = ""
  select_date: str = ""
  benchmark: str = ""
  cash: str = ""
  fee: str = ""
  exec_mode: str = ""
  warmup_bars: str = ""
  holdout: str = ""
  msd_host: str = ""
  fin_fields: str = ""
  fin_only_year: bool | None = None
  fin_lookback_days: str = ""
  strategy_args: dict[str, Any] = Field(default_factory=dict)
  extra_args: str = ""
  factor_periods: str = ""
  factor_primary_period: str = ""
  factor_quantiles: str = ""
  factor_entry: str = ""
  factor_exclude_limit: bool | None = None
  factor_min_listed_days: str = ""
  factor_winsorize: str = ""
  factor_standardize: str = ""
  factor_neutralize: str = ""
  factor_cost_bps: str = ""
  factor_min_obs: str = ""
  factor_by_industry: bool | None = None
  factor_industry_category: str = ""
  factor_causality_check: bool | None = None
  factor_direction: int | None = None


class ShortcutCreateRequest(BaseModel):
  name: str = Field(..., min_length=1)
  mode: RunModeLiteral
  params: dict[str, Any] = Field(default_factory=dict)
  notes: str = ""


class ShortcutUpdateRequest(BaseModel):
  name: str | None = None
  mode: RunModeLiteral | None = None
  params: dict[str, Any] | None = None
  notes: str | None = None


class ShortcutDuplicateRequest(BaseModel):
  name: str | None = None


class RunCreateRequest(BaseModel):
  strategy_id: str | None = None
  shortcut_id: str | None = None
  mode: RunModeLiteral | None = None
  overrides: dict[str, Any] = Field(default_factory=dict)


class RunSummaryResponse(BaseModel):
  id: str
  target_type: str
  target_id: str
  strategy_id: str
  shortcut_id: str | None = None
  mode: str
  status: str
  pid: int | None = None
  resolved_params: dict[str, Any] = Field(default_factory=dict)
  summary: dict[str, Any] | None = None
  error: str = ""
  started_at: str
  finished_at: str | None = None
  elapsed: float | None = None
  is_latest: bool = False


class ShortcutResponse(BaseModel):
  id: str
  strategy_id: str
  name: str
  mode: str
  params: dict[str, Any]
  notes: str = ""
  created_at: str
  updated_at: str
  latest_run: RunSummaryResponse | None = None
  active_run: RunSummaryResponse | None = None


class StrategyResponse(BaseModel):
  id: str
  name: str
  path: str
  default_mode: str
  metadata: dict[str, Any]
  default_params: dict[str, Any] = Field(default_factory=dict)
  notes: str
  file_exists: bool
  created_at: str
  updated_at: str
  created: bool | None = None
  shortcuts: list[ShortcutResponse] = Field(default_factory=list)
  latest_run: RunSummaryResponse | None = None
  active_run: RunSummaryResponse | None = None
