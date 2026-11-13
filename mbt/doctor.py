# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""mbt doctor: Environment, database and data coverage diagnostics tool."""

from __future__ import annotations

import os
import sys
import time
import importlib.metadata
from typing import Any


def check_env() -> dict[str, Any]:
  """Check Python runtime and core quantitative dependencies."""
  packages = ["polars", "numpy", "pymsd", "alpha", "orjson"]
  installed = {}
  for pkg in packages:
    try:
      ver = importlib.metadata.version(pkg)
      installed[pkg] = ver
    except importlib.metadata.PackageNotFoundError:
      try:
        mod = __import__(pkg)
        installed[pkg] = getattr(mod, "__version__", "unknown")
      except Exception:
        installed[pkg] = None

  return {
    "python": sys.version.split()[0],
    "packages": installed,
  }


def check_msd(host: str) -> dict[str, Any]:
  """Check MSD database connection, latency, and table coverage."""
  import pymsd

  t0 = time.time()
  try:
    client = pymsd.MsdConnection(host)
    latency_ms = (time.time() - t0) * 1000
    connected = True
  except Exception as e:
    return {
      "connected": False,
      "host": host,
      "error": str(e),
      "latency_ms": None,
      "tables": {},
    }

  tables_info = {}
  # Check calendar baseline
  try:
    res = client.query_concat(
      'SELECT ts FROM stock_kline_1d WHERE obj="SH000001" LIMIT 1; '
      'SELECT ts FROM stock_kline_1d WHERE obj="SH000001" LIMIT -1;'
    )
    if "stock_kline_1d" in res:
      t, _ = res["stock_kline_1d"]
      d = t.to_dict()["ts"]
      tables_info["stock_kline_1d (calendar: SH000001)"] = {
        "min_ts": str(d[0])[:10],
        "max_ts": str(d[-1])[:10],
      }
  except Exception as e:
    tables_info["stock_kline_1d (calendar: SH000001)"] = {"error": str(e)}

  for tbl in ["stock_financial", "stock_shares", "stock_dividend"]:
    try:
      res = client.query_concat(
        f"SELECT ts FROM {tbl} WHERE obj='SZ000001' LIMIT 1; SELECT ts FROM {tbl}  WHERE obj='SZ000001' LIMIT -1;"
      )
      if tbl in res:
        t, _ = res[tbl]
        d = t.to_dict()["ts"]
        tables_info[tbl + " (sample: SZ000001)"] = {
          "min_ts": str(d[0])[:10],
          "max_ts": str(d[-1])[:10],
        }
    except Exception as e:
      tables_info[tbl] = {"error": str(e)}

  return {
    "connected": connected,
    "host": host,
    "latency_ms": latency_ms,
    "tables": tables_info,
  }


def check_stocks() -> dict[str, Any]:
  """Check stock universe and cached stock codes."""
  try:
    import mbt.shared.stocks as s

    return {
      "A_STOCKS": len(s.A_STOCKS),
      "A_STOCKS_EXCLUDE_ST": len(s.A_STOCKS_EXCLUDE_ST),
      "FOUNDS": len(s.FOUNDS),
    }
  except Exception as e:
    return {"error": str(e)}


def check_fee() -> dict[str, Any]:
  """Check transaction fee configuration."""
  from mbt import Fee

  fee_path = os.environ.get("MSD_BT_FEE")
  try:
    fee = Fee.load(fee_path)
    return {
      "path": fee_path or "etc/fee.json (default)",
      "rates": [vars(item) for item in fee.config] if hasattr(fee, "config") else "loaded",
    }
  except Exception as e:
    return {"error": str(e)}


def run_doctor(msd_host: str | None = None) -> int:
  """Run full diagnostics and print structured Markdown report."""
  host = msd_host or os.environ.get("MSD_HOST", "http://localhost:50510")
  os.environ.setdefault("MSD_HOST", host)
  env_info = check_env()
  msd_info = check_msd(host)
  stocks_info = check_stocks()
  fee_info = check_fee()

  lines = []
  lines.append("# 🩺 mbt doctor: 系统环境与数据自检报告\n")

  # 1. Environment
  lines.append("## 1. 运行时与依赖环境")
  lines.append(f"- **Python 版本**: `{env_info['python']}`")
  for pkg, ver in env_info["packages"].items():
    status = "✅" if ver else "❌ (未安装)"
    lines.append(f"- **{pkg}**: `{ver or 'None'}` {status}")
  lines.append("")

  # 2. MSD Database
  lines.append("## 2. MSD 数据库服务")
  if msd_info["connected"]:
    lines.append(f"- **状态**: ✅ 已连接 (`{msd_info['host']}`)")
    lines.append(f"- **探测延迟**: `{msd_info['latency_ms']:.2f} ms`")
    lines.append("- **数据表覆盖范围**:")
    for tbl, info in msd_info["tables"].items():
      if "error" in info:
        lines.append(f"  - `{tbl}`: ⚠️ {info['error']}")
      else:
        lines.append(f"  - `{tbl}`: **{info['min_ts']}** 至 **{info['max_ts']}**")
  else:
    lines.append(f"- **状态**: ❌ 无法连接 (`{msd_info['host']}`)")
    lines.append(f"- **错误**: `{msd_info.get('error')}`")
    lines.append("- **指引**: 请确认 MSD 服务是否已启动，或检查环境变量 `MSD_HOST`。")
  lines.append("")

  # 3. Stocks Universe
  lines.append("## 3. 股票与基金标的池")
  if "error" in stocks_info:
    lines.append(f"- ⚠️ 标的池加载异常: `{stocks_info['error']}`")
  else:
    lines.append(f"- **全市场 A 股 (A_STOCKS)**: `{stocks_info['A_STOCKS']} 只`")
    lines.append(f"- **剔除 ST/退市 (A_STOCKS_EXCLUDE_ST)**: `{stocks_info['A_STOCKS_EXCLUDE_ST']} 只`")
    lines.append(f"- **公募/ETF 基金 (FOUNDS)**: `{stocks_info['FOUNDS']} 只`")
  lines.append("")

  # 4. Fee Configuration
  lines.append("## 4. 手续费配置")
  lines.append(f"- **配置来源**: `{fee_info.get('path')}`")
  if "error" in fee_info:
    lines.append(f"- ⚠️ 费率解析失败: `{fee_info['error']}`")
  else:
    lines.append("- **状态**: ✅ 费率模型就绪")
  lines.append("")

  report = "\n".join(lines)
  print(report)

  return 0 if msd_info["connected"] else 1


if __name__ == "__main__":
  sys.exit(run_doctor())
