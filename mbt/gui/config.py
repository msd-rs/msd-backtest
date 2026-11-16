# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Persistent configuration management for the mbt GUI."""

from __future__ import annotations

import dataclasses
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

RUN_MODE_SELECT = "select"
RUN_MODE_TIMING = "timing"
RUN_MODE_PORTFOLIO = "portfolio"
VALID_RUN_MODES = (RUN_MODE_SELECT, RUN_MODE_TIMING, RUN_MODE_PORTFOLIO)

RUN_MODE_LABELS = {
  RUN_MODE_SELECT: "选股型",
  RUN_MODE_TIMING: "择时型",
  RUN_MODE_PORTFOLIO: "资金账户型",
}


def get_config_dir() -> Path:
  """Return the well-known user configuration directory for mbt.

  Respects ``MBT_CONFIG_DIR`` if set; otherwise resolves the platform's
  standard user config root and appends ``mbt``.
  """
  override = os.environ.get("MBT_CONFIG_DIR", "").strip()
  if override:
    return Path(override).expanduser()

  if sys.platform == "win32":
    base = os.environ.get("APPDATA", "").strip()
    root = Path(base) if base else Path.home() / "AppData" / "Roaming"
  elif sys.platform == "darwin":
    root = Path.home() / "Library" / "Application Support"
  else:
    xdg = os.environ.get("XDG_CONFIG_HOME", "").strip()
    root = Path(xdg).expanduser() if xdg else Path.home() / ".config"
  return root / "mbt"


def get_config_file() -> Path:
  """Return the default GUI config JSON file path."""
  return get_config_dir() / "gui.json"


@dataclasses.dataclass
class ShortcutConfig:
  """Saved parameter preset (shortcut) belonging to a strategy."""

  id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
  name: str = "默认运行"
  mode: str = RUN_MODE_TIMING
  objs: str = ""
  use_internal_selector: bool = False
  begin: str = "2023-01-01"
  end: str = "2024-12-31"
  select_date: str = ""
  benchmark: str = "SH000300"
  cash: str = ""
  exec_mode: str = ""
  warmup_bars: str = ""
  msd_host: str = ""
  strategy_args: dict[str, str] = dataclasses.field(default_factory=dict)
  extra_args: str = ""

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "ShortcutConfig":
    """Deserialize a shortcut dictionary with validation."""
    mode = str(data.get("mode", RUN_MODE_TIMING))
    if mode not in VALID_RUN_MODES:
      raise ValueError(f"Unsupported shortcut mode '{mode}'. Expected one of {VALID_RUN_MODES}.")
    raw_args = data.get("strategy_args") or {}
    if not isinstance(raw_args, dict):
      raise ValueError(f"Invalid strategy_args in shortcut '{data.get('name', '')}'.")
    return cls(
      id=str(data.get("id") or uuid.uuid4().hex[:12]),
      name=str(data.get("name") or "未命名快捷路径"),
      mode=mode,
      objs=str(data.get("objs") or ""),
      use_internal_selector=bool(data.get("use_internal_selector", False)),
      begin=str(data.get("begin") or "2023-01-01"),
      end=str(data.get("end") or "2024-12-31"),
      select_date=str(data.get("select_date") or ""),
      benchmark=str(data.get("benchmark") or "SH000300"),
      cash=str(data.get("cash") or ""),
      exec_mode=str(data.get("exec_mode") or ""),
      warmup_bars=str(data.get("warmup_bars") or ""),
      msd_host=str(data.get("msd_host") or ""),
      strategy_args={str(k): str(v) for k, v in raw_args.items()},
      extra_args=str(data.get("extra_args") or ""),
    )

  def to_dict(self) -> dict[str, Any]:
    """Serialize shortcut configuration to JSON-friendly dict."""
    return dataclasses.asdict(self)


@dataclasses.dataclass
class StrategyConfig:
  """Imported strategy entry and its associated shortcuts."""

  id: str = dataclasses.field(default_factory=lambda: uuid.uuid4().hex[:12])
  name: str = ""
  path: str = ""
  default_mode: str = RUN_MODE_TIMING
  shortcuts: list[ShortcutConfig] = dataclasses.field(default_factory=list)

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "StrategyConfig":
    """Deserialize a strategy entry dictionary with validation."""
    mode = str(data.get("default_mode", RUN_MODE_TIMING))
    if mode not in VALID_RUN_MODES:
      raise ValueError(f"Unsupported strategy mode '{mode}'. Expected one of {VALID_RUN_MODES}.")
    shortcuts = [ShortcutConfig.from_dict(s) for s in (data.get("shortcuts") or [])]
    path = str(data.get("path") or "")
    name = str(data.get("name") or Path(path).stem or "未命名策略")
    return cls(
      id=str(data.get("id") or uuid.uuid4().hex[:12]),
      name=name,
      path=path,
      default_mode=mode,
      shortcuts=shortcuts,
    )

  def to_dict(self) -> dict[str, Any]:
    """Serialize strategy configuration to JSON-friendly dict."""
    return {
      "id": self.id,
      "name": self.name,
      "path": self.path,
      "default_mode": self.default_mode,
      "shortcuts": [s.to_dict() for s in self.shortcuts],
    }


@dataclasses.dataclass
class GuiConfig:
  """Top-level persisted GUI state."""

  version: int = 1
  strategies: list[StrategyConfig] = dataclasses.field(default_factory=list)
  last_dir: str = ""

  @classmethod
  def from_dict(cls, data: dict[str, Any]) -> "GuiConfig":
    """Deserialize full GUI config dictionary."""
    strategies = [StrategyConfig.from_dict(s) for s in (data.get("strategies") or [])]
    return cls(
      version=int(data.get("version", 1)),
      strategies=strategies,
      last_dir=str(data.get("last_dir") or ""),
    )

  def to_dict(self) -> dict[str, Any]:
    """Serialize full GUI config to JSON-friendly dict."""
    return {
      "version": self.version,
      "last_dir": self.last_dir,
      "strategies": [s.to_dict() for s in self.strategies],
    }


def load_config(path: Path | None = None) -> GuiConfig:
  """Load GUI configuration from disk, returning empty defaults if not yet created."""
  cfg_path = path or get_config_file()
  if not cfg_path.exists():
    return GuiConfig()
  text = cfg_path.read_text(encoding="utf-8")
  if not text.strip():
    return GuiConfig()
  data = json.loads(text)
  if not isinstance(data, dict):
    raise ValueError(f"Invalid GUI config root in {cfg_path}: expected JSON object.")
  return GuiConfig.from_dict(data)


def save_config(cfg: GuiConfig, path: Path | None = None) -> Path:
  """Atomically save GUI configuration to disk, creating parent directories if needed."""
  cfg_path = path or get_config_file()
  cfg_path.parent.mkdir(parents=True, exist_ok=True)
  tmp_path = cfg_path.with_suffix(".json.tmp")
  payload = json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2)
  tmp_path.write_text(payload + "\n", encoding="utf-8")
  os.replace(tmp_path, cfg_path)
  return cfg_path
