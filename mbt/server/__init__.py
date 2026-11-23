# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Local strategy management and backtest/factor execution server (`mbt.server`)."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
import re
from typing import TYPE_CHECKING


def get_version() -> str:
  """Return the installed ``msd-backtest`` version string."""
  try:
    return _pkg_version("msd-backtest")
  except PackageNotFoundError:
    pass

  pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
  if pyproject.is_file():
    text = pyproject.read_text(encoding="utf-8")
    m = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if m:
      return m.group(1)
  return "0.3.0"


from .config import (
  ServerConfig,
  ServerListenConfig,
  SystemDefaults,
  get_config_dir,
  get_config_file,
  get_data_dir,
  get_db_path,
  get_runs_dir,
  load_server_config,
  save_server_config,
)

if TYPE_CHECKING:
  from .app import create_app


def __getattr__(name: str):
  if name == "create_app":
    from .app import create_app
    return create_app
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "ServerConfig",
  "ServerListenConfig",
  "SystemDefaults",
  "create_app",
  "get_config_dir",
  "get_config_file",
  "get_data_dir",
  "get_db_path",
  "get_runs_dir",
  "get_version",
  "load_server_config",
  "save_server_config",
]
