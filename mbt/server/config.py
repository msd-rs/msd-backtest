# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""OS-aware configuration and data directory management for ``mbt.server``."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

VALID_EXEC_MODES = ("close", "next_open")
VALID_FACTOR_ENTRIES = ("next_open", "open_close", "close")
VALID_WINSORIZE = ("none", "mad", "sigma", "percentile")
VALID_STANDARDIZE = ("none", "zscore", "rank")


def get_config_dir(override: Path | str | None = None) -> Path:
  """Return the OS-standard configuration directory for ``mbt``.

  Respects ``override`` or ``MBT_SERVER_CONFIG_DIR`` when set.
  """
  if override is not None:
    return Path(override).expanduser()
  env_dir = os.environ.get("MBT_SERVER_CONFIG_DIR", "").strip()
  if env_dir:
    return Path(env_dir).expanduser()

  if sys.platform == "win32":
    base = os.environ.get("APPDATA", "").strip()
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
  elif sys.platform == "darwin":
    root = Path.home() / "Library" / "Application Support"
  else:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
  return root / "mbt"


def get_config_file(config_dir: Path | str | None = None) -> Path:
  """Return the ``server.json`` configuration file path."""
  return get_config_dir(config_dir) / "server.json"


def get_data_dir(override: Path | str | None = None) -> Path:
  """Return the OS-standard user data directory for ``mbt.server``.

  Respects ``override`` or ``MBT_SERVER_DATA_DIR`` when set.
  """
  if override is not None:
    return Path(override).expanduser()
  env_dir = os.environ.get("MBT_SERVER_DATA_DIR", "").strip()
  if env_dir:
    return Path(env_dir).expanduser()

  if sys.platform == "win32":
    base = os.environ.get("LOCALAPPDATA", "").strip()
    root = Path(base) if base else Path.home() / "AppData" / "Local"
  elif sys.platform == "darwin":
    root = Path.home() / "Library" / "Application Support"
  else:
    xdg = os.environ.get("XDG_DATA_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".local" / "share"
  return root / "mbt" / "server"


def get_db_path(data_dir: Path | str | None = None) -> Path:
  """Return the SQLite state database path under the data directory."""
  return get_data_dir(data_dir) / "state.db"


def get_runs_dir(data_dir: Path | str | None = None) -> Path:
  """Return the directory storing run Parquet artifacts and logs."""
  return get_data_dir(data_dir) / "runs"


@dataclasses.dataclass
class ServerListenConfig:
  """HTTP server network binding configuration."""

  host: str = "127.0.0.1"
  port: int = 8765

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "ServerListenConfig":
    """Deserialize and validate listener configuration."""
    host = str(data.get("host") or "127.0.0.1").strip()
    if not host:
      raise ValueError("Server host cannot be empty.")
    port = int(data.get("port", 8765))
    if not (1 <= port <= 65535):
      raise ValueError(f"Invalid server port {port}: expected 1..65535.")
    return cls(host=host, port=port)

  def to_dict(self) -> dict[str, Any]:
    """Serialize listener configuration to a JSON-friendly dictionary."""
    return {"host": self.host, "port": self.port}


@dataclasses.dataclass
class SystemDefaults:
  """System-wide default execution parameters inherited by strategies and shortcuts."""

  msd_host: str = dataclasses.field(
    default_factory=lambda: os.environ.get("MSD_HOST", "http://localhost:50510")
  )
  begin: str = dataclasses.field(
    default_factory=lambda: os.environ.get("MSD_BT_BEGIN", "2023-01-01")
  )
  end: str = dataclasses.field(
    default_factory=lambda: os.environ.get("MSD_BT_END", "2025-12-31")
  )
  select_date: str = ""
  benchmark: str = dataclasses.field(
    default_factory=lambda: os.environ.get("MSD_BT_BENCHMARK", "SH000300")
  )
  cash: float = 10000000.0
  fee: str = dataclasses.field(
    default_factory=lambda: os.environ.get("MSD_BT_FEE", "")
  )
  exec_mode: str = "close"
  warmup_bars: int = 0
  holdout: str = ""
  fin_fields: str = ""
  fin_only_year: bool = False
  fin_lookback_days: int | None = None

  # Factor evaluation defaults (aligned with mbt.factor.cli)
  factor_periods: str = "1,5,10,20"
  factor_primary_period: int | None = None
  factor_quantiles: int = 5
  factor_entry: str = "next_open"
  factor_exclude_limit: bool = True
  factor_min_listed_days: int = 0
  factor_winsorize: str = "none"
  factor_standardize: str = "none"
  factor_neutralize: str = "none"
  factor_cost_bps: float = 0.0
  factor_min_obs: int = 10
  factor_by_industry: bool = False
  factor_industry_category: str = "申万行业"
  factor_causality_check: bool = True

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "SystemDefaults":
    """Deserialize and validate system defaults dictionary."""
    base = cls()
    exec_mode = str(data.get("exec_mode", base.exec_mode) or "close").strip()
    if exec_mode not in VALID_EXEC_MODES:
      raise ValueError(
        f"Invalid exec_mode '{exec_mode}'. Expected one of {VALID_EXEC_MODES}."
      )

    factor_entry = str(data.get("factor_entry", base.factor_entry) or "next_open").strip()
    if factor_entry not in VALID_FACTOR_ENTRIES:
      raise ValueError(
        f"Invalid factor_entry '{factor_entry}'. Expected one of {VALID_FACTOR_ENTRIES}."
      )

    factor_winsorize = str(data.get("factor_winsorize", base.factor_winsorize) or "none").strip()
    if factor_winsorize not in VALID_WINSORIZE:
      raise ValueError(
        f"Invalid factor_winsorize '{factor_winsorize}'. Expected one of {VALID_WINSORIZE}."
      )

    factor_standardize = str(
      data.get("factor_standardize", base.factor_standardize) or "none"
    ).strip()
    if factor_standardize not in VALID_STANDARDIZE:
      raise ValueError(
        f"Invalid factor_standardize '{factor_standardize}'. Expected one of {VALID_STANDARDIZE}."
      )

    cash_raw = data.get("cash", base.cash)
    cash = float(cash_raw) if cash_raw not in ("", None) else base.cash
    if cash <= 0:
      raise ValueError(f"Default cash must be > 0, got {cash}.")

    warmup_raw = data.get("warmup_bars", base.warmup_bars)
    warmup_bars = int(warmup_raw) if warmup_raw not in ("", None) else 0
    if warmup_bars < 0:
      raise ValueError(f"Default warmup_bars must be >= 0, got {warmup_bars}.")

    fin_lb_raw = data.get("fin_lookback_days", base.fin_lookback_days)
    fin_lookback_days = int(fin_lb_raw) if fin_lb_raw not in ("", None) else None

    prim_raw = data.get("factor_primary_period", base.factor_primary_period)
    factor_primary_period = int(prim_raw) if prim_raw not in ("", None) else None

    quantiles_raw = data.get("factor_quantiles", base.factor_quantiles)
    factor_quantiles = int(quantiles_raw) if quantiles_raw not in ("", None) else 5
    if factor_quantiles < 2:
      raise ValueError(f"factor_quantiles must be >= 2, got {factor_quantiles}.")

    min_listed_raw = data.get("factor_min_listed_days", base.factor_min_listed_days)
    factor_min_listed_days = int(min_listed_raw) if min_listed_raw not in ("", None) else 0
    if factor_min_listed_days < 0:
      raise ValueError(
        f"factor_min_listed_days must be >= 0, got {factor_min_listed_days}."
      )

    cost_raw = data.get("factor_cost_bps", base.factor_cost_bps)
    factor_cost_bps = float(cost_raw) if cost_raw not in ("", None) else 0.0

    min_obs_raw = data.get("factor_min_obs", base.factor_min_obs)
    factor_min_obs = int(min_obs_raw) if min_obs_raw not in ("", None) else 10
    if factor_min_obs < 2:
      raise ValueError(f"factor_min_obs must be >= 2, got {factor_min_obs}.")

    return cls(
      msd_host=str(data.get("msd_host", base.msd_host) or base.msd_host).strip(),
      begin=str(data.get("begin", base.begin) or base.begin).strip(),
      end=str(data.get("end", base.end) or base.end).strip(),
      select_date=str(data.get("select_date", base.select_date) or "").strip(),
      benchmark=str(data.get("benchmark", base.benchmark) or base.benchmark).strip(),
      cash=cash,
      fee=str(data.get("fee", base.fee) or "").strip(),
      exec_mode=exec_mode,
      warmup_bars=warmup_bars,
      holdout=str(data.get("holdout", base.holdout) or "").strip(),
      fin_fields=str(data.get("fin_fields", base.fin_fields) or "").strip(),
      fin_only_year=bool(data.get("fin_only_year", base.fin_only_year)),
      fin_lookback_days=fin_lookback_days,
      factor_periods=str(data.get("factor_periods", base.factor_periods) or "1,5,10,20").strip(),
      factor_primary_period=factor_primary_period,
      factor_quantiles=factor_quantiles,
      factor_entry=factor_entry,
      factor_exclude_limit=bool(data.get("factor_exclude_limit", base.factor_exclude_limit)),
      factor_min_listed_days=factor_min_listed_days,
      factor_winsorize=factor_winsorize,
      factor_standardize=factor_standardize,
      factor_neutralize=str(
        data.get("factor_neutralize", base.factor_neutralize) or "none"
      ).strip(),
      factor_cost_bps=factor_cost_bps,
      factor_min_obs=factor_min_obs,
      factor_by_industry=bool(data.get("factor_by_industry", base.factor_by_industry)),
      factor_industry_category=str(
        data.get("factor_industry_category", base.factor_industry_category) or "申万行业"
      ).strip(),
      factor_causality_check=bool(
        data.get("factor_causality_check", base.factor_causality_check)
      ),
    )

  def to_dict(self) -> dict[str, Any]:
    """Serialize system defaults to a JSON-friendly dictionary."""
    return dataclasses.asdict(self)


@dataclasses.dataclass
class ServerConfig:
  """Top-level persisted configuration stored in ``server.json``."""

  version: int = 1
  server: ServerListenConfig = dataclasses.field(default_factory=ServerListenConfig)
  defaults: SystemDefaults = dataclasses.field(default_factory=SystemDefaults)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "ServerConfig":
    """Deserialize full server configuration dictionary."""
    srv_raw = data.get("server") or {}
    def_raw = data.get("defaults") or {}
    if not isinstance(srv_raw, dict) or not isinstance(def_raw, dict):
      raise ValueError("Invalid server.json structure: 'server' and 'defaults' must be objects.")
    return cls(
      version=int(data.get("version", 1)),
      server=ServerListenConfig.from_dict(srv_raw),
      defaults=SystemDefaults.from_dict(def_raw),
    )

  def to_dict(self) -> dict[str, Any]:
    """Serialize full server configuration to a JSON-friendly dictionary."""
    return {
      "version": self.version,
      "server": self.server.to_dict(),
      "defaults": self.defaults.to_dict(),
    }


def load_server_config(config_dir: Path | str | None = None) -> ServerConfig:
  """Load ``server.json`` from the config directory, creating defaults if missing."""
  cfg_path = get_config_file(config_dir)
  if not cfg_path.exists():
    cfg = ServerConfig()
    save_server_config(cfg, config_dir=config_dir)
    return cfg

  text = cfg_path.read_text(encoding="utf-8")
  if not text.strip():
    raise ValueError(f"Server config file '{cfg_path}' is empty; expected a JSON object.")

  try:
    data = json.loads(text)
  except json.JSONDecodeError as exc:
    raise ValueError(f"Malformed JSON in server config '{cfg_path}': {exc}") from exc

  if not isinstance(data, dict):
    raise ValueError(f"Invalid server config root in '{cfg_path}': expected a JSON object.")
  return ServerConfig.from_dict(data)


def save_server_config(cfg: ServerConfig, config_dir: Path | str | None = None) -> Path:
  """Atomically persist ``ServerConfig`` to ``server.json``."""
  cfg_path = get_config_file(config_dir)
  cfg_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = cfg_path.with_suffix(".json.tmp")
  payload = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)
  tmp_path.write_text(payload + "\n", encoding="utf-8")
  os.replace(tmp_path, cfg_path)
  return cfg_path
