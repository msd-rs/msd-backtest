# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Subprocess execution worker for mbt GUI runs."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import pickle
import shlex
import time
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

import numpy as np
import polars as pl

from mbt import Fee, PortfolioAccount, StaticSelector
from mbt.gui.config import RUN_MODE_PORTFOLIO, RUN_MODE_SELECT, RUN_MODE_TIMING
from mbt.run.backend import SYSTEM_ARGS, RunRequest, parse_kv_args
from mbt.run.cli import build_selector, run_single_period
from mbt.run.report import (
  _format_date,
  compute_stock_attribution,
  extract_portfolio_trades,
)
from mbt.select import MsdSelectorDataProvider
from mbt.select.cli import expand_objs as select_expand_objs, load_selector, try_load_selector

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s",
  level=logging.INFO,
)
logger = logging.getLogger("mbt.gui.worker")


def split_objs_text(objs_text: str) -> list[str]:
  """Split user-entered stock/selector input by commas or whitespace."""
  tokens: list[str] = []
  for part in objs_text.replace(",", " ").split():
    p = part.strip()
    if p:
      tokens.append(p)
  return tokens


def build_kv_kwargs(
  strategy_args: dict[str, str] | None,
  extra_args: str,
) -> dict[str, Any]:
  """Combine structured parameter entries and extra key=value tokens."""
  tokens: list[str] = []
  if strategy_args:
    for k, v in strategy_args.items():
      v_str = str(v).strip()
      if k and v_str != "":
        tokens.append(f"{k}={v_str}")
  if extra_args and extra_args.strip():
    for tok in shlex.split(extra_args.strip()):
      tokens.append(tok)
  return parse_kv_args(tokens) if tokens else {}


def _format_dates(dates: np.ndarray | None) -> list[str]:
  if dates is None or len(dates) == 0:
    return []
  out: list[str] = []
  for i in range(len(dates)):
    s = _format_date(i, int(dates[i]) if isinstance(dates[i], (int, np.integer)) else 0, dates)
    if len(s) == 8 and s.isdigit():
      s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    out.append(s)
  return out


def _validate_kwargs(kv_kwargs: dict[str, Any], *classes: type | None) -> None:
  if not kv_kwargs:
    return
  valid_keys = set(SYSTEM_ARGS)
  names: list[str] = []
  for cls in classes:
    if cls is None:
      continue
    names.append(cls.__name__)
    if cls.__init__ is not object.__init__:
      valid_keys.update(
        p for p in inspect.signature(cls.__init__).parameters if p != "self"
      )
  orphan_keys = [k for k in kv_kwargs if k not in valid_keys]
  if orphan_keys:
    all_custom = sorted(valid_keys - set(SYSTEM_ARGS))
    hint = (
      f" Accepted custom parameters: {all_custom}."
      if all_custom
      else " Target class declares no custom __init__ parameters."
    )
    raise ValueError(
      f"Unrecognized argument(s): {orphan_keys} for {', '.join(names)}.{hint}"
    )


def execute_spec(spec: dict[str, Any]) -> dict[str, Any]:
  """Execute a GUI run specification and return a serializable result payload."""
  t0 = time.perf_counter()
  mode = spec["mode"]
  strategy_path = spec["strategy_path"]
  msd_host = (spec.get("msd_host") or os.environ.get("MSD_HOST") or "http://localhost:50510").strip()
  os.environ["MSD_HOST"] = msd_host
  raw_objs = split_objs_text(spec.get("objs") or "")
  kv_kwargs = build_kv_kwargs(spec.get("strategy_args"), spec.get("extra_args") or "")

  if mode == RUN_MODE_SELECT:
    selector = load_selector(strategy_path, **kv_kwargs)
    _validate_kwargs(kv_kwargs, selector.__class__)
    select_date = (spec.get("select_date") or "").strip() or None
    logger.info(f"Running selector {selector.__class__.__name__} on date={select_date or 'latest'}")
    dp = MsdSelectorDataProvider(msd_host, select_date)
    init_objs = select_expand_objs(raw_objs)
    stocks = list(selector.execute(dp, init_objs))
    stocks.sort()

    factors = getattr(selector, "factors", None)
    columns: list[str] = ["obj"]
    rows: list[list[Any]] = []
    if factors is not None and isinstance(factors, pl.DataFrame) and not factors.is_empty():
      if "obj" in factors.columns:
        try:
          factors = factors.sort("obj")
        except Exception:
          pass
      columns = list(factors.columns)
      rows = [list(r) for r in factors.iter_rows()]
    else:
      rows = [[s] for s in stocks]

    elapsed = time.perf_counter() - t0
    return {
      "mode": RUN_MODE_SELECT,
      "strategy_name": selector.__class__.__name__,
      "select_date": select_date or getattr(dp, "date", "") or "最新交易日",
      "stocks": stocks,
      "columns": columns,
      "rows": rows,
      "elapsed": elapsed,
    }

  if mode not in (RUN_MODE_TIMING, RUN_MODE_PORTFOLIO):
    raise ValueError(f"Unsupported execution mode: '{mode}'.")

  begin = (spec.get("begin") or os.environ.get("MSD_BT_BEGIN") or "2023-01-01").strip()
  end = (spec.get("end") or os.environ.get("MSD_BT_END") or "2024-12-31").strip()
  benchmark = (spec.get("benchmark") or os.environ.get("MSD_BT_BENCHMARK") or "SH000300").strip()
  cash_str = (spec.get("cash") or "").strip()
  cash = float(cash_str) if cash_str else None
  exec_mode_override = (spec.get("exec_mode") or "").strip() or None
  warmup_str = (spec.get("warmup_bars") or "").strip()
  warmup_override = int(warmup_str) if warmup_str else None

  use_internal_selector = bool(spec.get("use_internal_selector", False))

  with open(strategy_path, "r", encoding="utf-8") as f:
    req = RunRequest(
      f,
      raw_objs,
      start=begin,
      end=end,
      args=kv_kwargs if kv_kwargs else None,
    )

  selector_objs = list(raw_objs)
  if use_internal_selector:
    if try_load_selector(strategy_path, **kv_kwargs) is None:
      raise ValueError(
        f"已勾选“使用内置选股器”，但策略文件 '{strategy_path}' 中未定义 Selector 子类。"
      )
    if strategy_path not in selector_objs:
      selector_objs = [strategy_path] + selector_objs

  selector, init_objs = build_selector(
    selector_objs,
    selector_kwargs=kv_kwargs,
    strategy_path=strategy_path,
  )
  embedded_sel = None
  if isinstance(selector, StaticSelector) and strategy_path.endswith(".py"):
    try:
      embedded_sel = try_load_selector(strategy_path, **kv_kwargs)
    except Exception:
      pass

  _validate_kwargs(
    kv_kwargs,
    getattr(req.strategy, "__class__", None),
    selector.__class__,
    getattr(embedded_sel, "__class__", None),
  )

  use_portfolio = (mode == RUN_MODE_PORTFOLIO) or bool(getattr(req.strategy, "portfolio", False))
  if mode == RUN_MODE_TIMING and bool(getattr(req.strategy, "portfolio", False)):
    raise ValueError(
      f"Strategy '{req.strategy.__class__.__name__}' is a PortfolioStrategy (portfolio=True) "
      "and must be run in 资金账户型 (portfolio) mode."
    )
  if use_portfolio and len(selector_objs) == 0:
    raise ValueError(
      "资金账户型 (portfolio) 模式需要显式指定候选股票池（例如 A_STOCKS_EXCLUDE_ST），或勾选“使用本策略内置选股器”。"
    )

  selector_dp = MsdSelectorDataProvider(msd_host, date=begin)
  selected_objs = list(selector.execute(selector_dp, init_objs))
  if len(selected_objs) == 0:
    if isinstance(selector, StaticSelector):
      raise ValueError("未提供任何标的（请填写标的池，如 SH600000 或 A_STOCKS_EXCLUDE_ST）。")
    raise ValueError(
      f"Selector '{selector.__class__.__name__}' 在 {begin} 从 {len(init_objs)} 只候选中选出 0 只标的。"
    )

  try:
    selected_objs.remove(benchmark)
  except ValueError:
    pass
  selected_objs.insert(0, benchmark)

  req.objs = selected_objs
  req.portfolio = use_portfolio
  req.cash = cash  # type: ignore[attr-defined]
  req.benchmark = benchmark
  req.selector = selector
  req.raw_objs = raw_objs
  req.kv_kwargs = kv_kwargs
  req.strategy_path = strategy_path

  merged_fin: list[str] = []
  for src in [
    kv_kwargs.get("fin_fields"),
    getattr(req, "fin_fields", None),
    getattr(req.strategy, "fin_fields", None),
    getattr(selector, "fin_fields", None),
    getattr(embedded_sel, "fin_fields", None),
  ]:
    if not src:
      continue
    items = [x.strip() for x in src.split(",") if x.strip()] if isinstance(src, str) else list(src)
    for item in items:
      if item not in merged_fin:
        merged_fin.append(item)
  req.fin_fields = merged_fin if merged_fin else None
  req.fin_only_year = bool(
    kv_kwargs.get("fin_only_year")
    or getattr(req.strategy, "fin_only_year", False)
    or getattr(selector, "fin_only_year", False)
    or getattr(embedded_sel, "fin_only_year", False)
  )

  warmup_bars = int(
    warmup_override
    if warmup_override is not None
    else (kv_kwargs.get("warmup_bars") or getattr(req.strategy, "warmup_bars", 0))
  )
  exec_mode = (
    exec_mode_override
    or kv_kwargs.get("exec_mode")
    or getattr(req.strategy, "exec_mode", "close")
  )
  fin_lookback_days = kv_kwargs.get("fin_lookback_days") or getattr(
    req.strategy, "fin_lookback_days", None
  )

  fee_path = os.environ.get("MSD_BT_FEE", "")
  fee = Fee.load(fee_path, overrides=kv_kwargs)

  logger.info(
    f"Running {req.strategy.__class__.__name__} mode={mode} range={begin}~{end} objs={len(selected_objs)-1}"
  )
  reports, dp, acc = run_single_period(
    req,
    msd_host,
    fee,
    begin,
    end,
    warmup_bars=warmup_bars,
    exec_mode=exec_mode,
    fin_lookback_days=fin_lookback_days,
  )
  elapsed = time.perf_counter() - t0

  if use_portfolio or isinstance(acc, PortfolioAccount):
    r0 = reports[0]
    dates = _format_dates(r0.date)
    records_raw = getattr(acc, "records", None) or getattr(r0, "records", []) or []
    dp_dates = dp.dates[: dp.bars] if hasattr(dp, "dates") else None
    rebalance_records = []
    for rec in records_raw:
      rec_date = _format_date(rec.bar, rec.ts, dp_dates)
      if len(rec_date) == 8 and rec_date.isdigit():
        rec_date = f"{rec_date[:4]}-{rec_date[4:6]}-{rec_date[6:8]}"
      rebalance_records.append({
        "bar": int(rec.bar),
        "date": rec_date,
        "obj": str(rec.obj),
        "action": str(rec.action),
        "shares": float(rec.shares),
        "deal_price": float(rec.deal_price),
        "amount": float(rec.amount),
        "fee": float(rec.fee),
        "cash": float(rec.cash),
        "pos_shares": float(rec.pos_shares),
      })

    trades = getattr(r0, "trades", None) or extract_portfolio_trades(
      account=acc, dp=dp, records=records_raw
    )
    init_cash = float(getattr(acc, "initial_cash", 1000000.0))
    attribution = getattr(r0, "attribution", None) or (
      compute_stock_attribution(trades, initial_cash=init_cash) if trades else []
    )
    metrics = r0.metrics(with_obj=False)
    metrics["initial_cash"] = init_cash
    metrics["final_assets"] = float(acc.portfolio_assets[-1]) if len(getattr(acc, "portfolio_assets", [])) > 0 else init_cash
    metrics["pool_size"] = int(getattr(r0, "pool_size", len(selected_objs) - 1))
    metrics["traded_objs_count"] = int(getattr(r0, "traded_objs_count", 0))
    metrics["exec_mode"] = exec_mode

    return {
      "mode": RUN_MODE_PORTFOLIO,
      "strategy_name": req.strategy.__class__.__name__,
      "dates": dates,
      "ror": np.asarray(r0.ror, dtype=np.float64),
      "ror_original": np.asarray(r0.ror_original, dtype=np.float64),
      "benchmark_ror": (
        np.asarray(r0.benchmark_ror, dtype=np.float64)
        if getattr(r0, "benchmark_ror", None) is not None
        else np.zeros(len(dates), dtype=np.float64)
      ),
      "benchmark_obj": getattr(r0, "benchmark_obj", benchmark) or benchmark,
      "metrics": metrics,
      "rebalance_records": rebalance_records,
      "trades": trades,
      "attribution": attribution,
      "diagnostics": getattr(r0, "diagnostics", None),
      "elapsed": elapsed,
    }

  # Timing mode (single or multi-stock independent reports)
  report_items: list[dict[str, Any]] = []
  for rep in reports:
    m = rep.metrics(with_obj=True)
    report_items.append({
      "obj": rep.obj,
      "dates": _format_dates(rep.date),
      "ror": np.asarray(rep.ror, dtype=np.float64),
      "ror_original": np.asarray(rep.ror_original, dtype=np.float64),
      "benchmark_ror": (
        np.asarray(rep.benchmark_ror, dtype=np.float64)
        if getattr(rep, "benchmark_ror", None) is not None
        else np.zeros(len(rep.ror), dtype=np.float64)
      ),
      "benchmark_obj": getattr(rep, "benchmark_obj", benchmark) or benchmark,
      "metrics": m,
    })

  summary_curve: dict[str, Any] | None = None
  if len(report_items) > 1:
    base_len = len(report_items[0]["ror"])
    aligned = [it for it in report_items if len(it["ror"]) == base_len]
    if aligned:
      mean_ror = np.nanmean(np.vstack([it["ror"] for it in aligned]), axis=0)
      mean_orig = np.nanmean(np.vstack([it["ror_original"] for it in aligned]), axis=0)
      summary_curve = {
        "label": f"等权汇总 ({len(aligned)}只)",
        "dates": aligned[0]["dates"],
        "ror": mean_ror,
        "ror_original": mean_orig,
        "benchmark_ror": aligned[0]["benchmark_ror"],
        "benchmark_obj": aligned[0]["benchmark_obj"],
      }

  return {
    "mode": RUN_MODE_TIMING,
    "strategy_name": req.strategy.__class__.__name__,
    "reports": report_items,
    "summary_curve": summary_curve,
    "benchmark_obj": benchmark,
    "elapsed": elapsed,
  }


def main():
  parser = argparse.ArgumentParser(description="mbt GUI subprocess worker")
  parser.add_argument("--spec", required=True, help="Input JSON run spec file")
  parser.add_argument("--out", required=True, help="Output pickle result file")
  args = parser.parse_args()

  with open(args.spec, "r", encoding="utf-8") as f:
    spec = json.load(f)

  result = execute_spec(spec)
  tmp_out = f"{args.out}.tmp"
  with open(tmp_out, "wb") as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
  os.replace(tmp_out, args.out)


if __name__ == "__main__":
  main()
