# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Subprocess execution worker for ``mbt.server`` writing Parquet artifacts."""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

import numpy as np
import polars as pl

from mbt import Fee, PortfolioAccount, StaticSelector
from mbt.factor.evaluator import evaluate_factor
from mbt.factor.expr import ExpressionFactor, load_expression_file
from mbt.factor.factor import FACTOR_SYSTEM_ARGS, load_factor
from mbt.run.backend import SYSTEM_ARGS, RunRequest, parse_kv_args
from mbt.run.cli import build_selector, expand_objs as run_expand_objs, run_single_period
from mbt.run.report import (
  _format_date,
  compute_holdout_split,
  compute_stock_attribution,
  extract_portfolio_trades,
)
from mbt.select import MsdSelectorDataProvider
from mbt.select.cli import expand_objs as select_expand_objs, load_selector, try_load_selector
from mbt.server.storage import (
  format_dates_list,
  write_factor_result,
  write_portfolio_result,
  write_select_result,
  write_timing_result,
)
from mbt.shared.inspect_strategy import (
  RUN_MODE_FACTOR,
  RUN_MODE_PORTFOLIO,
  RUN_MODE_SELECT,
  RUN_MODE_TIMING,
)

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s",
  level=logging.INFO,
)
logger = logging.getLogger("mbt.server.worker")


def split_objs_text(objs_text: str) -> list[str]:
  """Split user-entered stock or selector input by commas or whitespace."""
  tokens: list[str] = []
  for part in objs_text.replace(",", " ").split():
    p = part.strip()
    if p:
      tokens.append(p)
  return tokens


def build_kv_kwargs(
  strategy_args: dict[str, Any] | None,
  extra_args: str,
) -> dict[str, Any]:
  """Combine structured parameter entries and extra ``key=value`` tokens."""
  tokens: list[str] = []
  typed_overrides: dict[str, Any] = {}
  if strategy_args:
    for k, v in strategy_args.items():
      if not k:
        continue
      if isinstance(v, (bool, int, float, list)):
        typed_overrides[str(k)] = v
      elif v is not None:
        v_str = str(v).strip()
        if v_str != "":
          tokens.append(f"{k}={v_str}")
  if extra_args and extra_args.strip():
    for tok in shlex.split(extra_args.strip()):
      tokens.append(tok)
  parsed = parse_kv_args(tokens) if tokens else {}
  return {**typed_overrides, **parsed}


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


def _parse_periods(raw: Any) -> tuple[int, ...]:
  if isinstance(raw, (list, tuple)):
    vals = tuple(sorted({int(p) for p in raw}))
  else:
    parts = [s.strip() for s in str(raw or "1,5,10,20").split(",") if s.strip()]
    if not parts:
      raise ValueError("factor_periods cannot be empty.")
    vals = tuple(sorted({int(p) for p in parts}))
  if any(p <= 0 for p in vals):
    raise ValueError(f"Invalid factor_periods {raw!r}: all periods must be > 0.")
  return vals


def _parse_neutralize(raw: Any) -> tuple[str, ...]:
  if not raw:
    return ()
  if isinstance(raw, (list, tuple)):
    items = [str(m).strip().lower() for m in raw]
  else:
    items = [m.strip().lower() for m in str(raw).split(",")]
  return tuple(dict.fromkeys(m for m in items if m and m != "none"))


def _str_opt(val: Any) -> str:
  return "" if val is None else str(val).strip()


def execute_and_store(spec: dict[str, Any], out_dir: Path | str) -> dict[str, Any]:
  """Execute a resolved run specification and persist Parquet artifacts to ``out_dir``."""
  t0 = time.perf_counter()
  target_dir = Path(out_dir)
  target_dir.mkdir(parents=True, exist_ok=True)

  mode = str(spec["mode"])
  strategy_path = str(spec["strategy_path"])
  if not os.path.isfile(strategy_path):
    raise FileNotFoundError(f"Strategy or factor file not found: '{strategy_path}'")

  msd_host = _str_opt(spec.get("msd_host")) or os.environ.get("MSD_HOST") or "http://localhost:50510"
  os.environ["MSD_HOST"] = msd_host
  raw_objs = split_objs_text(_str_opt(spec.get("objs")))
  kv_kwargs = build_kv_kwargs(spec.get("strategy_args"), _str_opt(spec.get("extra_args")))

  if mode == RUN_MODE_SELECT:
    selector = load_selector(strategy_path, **kv_kwargs)
    _validate_kwargs(kv_kwargs, selector.__class__)
    select_date = _str_opt(spec.get("select_date")) or None
    logger.info(
      f"Running selector {selector.__class__.__name__} on date={select_date or 'latest'}"
    )
    dp = MsdSelectorDataProvider(msd_host, select_date)
    init_objs = select_expand_objs(raw_objs)
    stocks = sorted(list(selector.execute(dp, init_objs)))
    factors = getattr(selector, "factors", None)
    eff_date = select_date or str(getattr(dp, "date", "") or "") or "最新交易日"
    elapsed = time.perf_counter() - t0
    return write_select_result(
      target_dir,
      strategy_name=selector.__class__.__name__,
      select_date=eff_date,
      stocks=stocks,
      factors=factors if isinstance(factors, pl.DataFrame) else None,
      elapsed=elapsed,
    )

  if mode == RUN_MODE_FACTOR:
    objs = run_expand_objs(raw_objs)
    if not objs:
      raise ValueError(
        "No candidate stocks provided for factor evaluation. "
        "Specify a stock pool such as 'A_STOCKS_EXCLUDE_ST' or stock codes."
      )

    ext = os.path.splitext(strategy_path)[1].lower()
    if ext == ".py":
      factor_inst = load_factor(strategy_path, kv_kwargs, strict_args=True)
    elif ext in (".txt", ".json"):
      bad_keys = [k for k in kv_kwargs if k not in FACTOR_SYSTEM_ARGS]
      if bad_keys:
        raise ValueError(
          f"Unrecognized argument(s) {bad_keys} passed with expression file '{strategy_path}'."
        )
      exprs, dirs = load_expression_file(strategy_path)
      factor_inst = ExpressionFactor(exprs, direction=dirs)
    else:
      raise ValueError(f"Unsupported factor file extension '{ext}' for '{strategy_path}'.")

    dir_override = spec.get("factor_direction")
    if dir_override in (1, -1, "1", "-1"):
      d_int = int(dir_override)
      if isinstance(factor_inst.direction, dict):
        factor_inst.direction = {k: d_int for k in factor_inst.direction}
      else:
        factor_inst.direction = d_int

    begin = _str_opt(spec.get("begin")) or os.environ.get("MSD_BT_BEGIN") or "2023-01-01"
    end = _str_opt(spec.get("end")) or os.environ.get("MSD_BT_END") or "2025-12-31"
    benchmark = _str_opt(spec.get("benchmark")) or os.environ.get("MSD_BT_BENCHMARK") or "SH000300"

    fin_raw = _str_opt(spec.get("fin_fields"))
    cli_fin = [f.strip() for f in fin_raw.split(",") if f.strip()] if fin_raw else None
    fin_only_year = bool(spec.get("fin_only_year", False))
    fin_lb_raw = _str_opt(spec.get("fin_lookback_days"))
    fin_lookback_days = int(fin_lb_raw) if fin_lb_raw else None
    warmup_raw = _str_opt(spec.get("warmup_bars"))
    warmup_bars = int(warmup_raw) if warmup_raw else None

    periods = _parse_periods(spec.get("factor_periods", "1,5,10,20"))
    prim_raw = _str_opt(spec.get("factor_primary_period"))
    primary_period = int(prim_raw) if prim_raw else None
    quantiles_raw = _str_opt(spec.get("factor_quantiles"))
    quantiles = int(quantiles_raw) if quantiles_raw else 5
    entry = _str_opt(spec.get("factor_entry")) or "next_open"
    exclude_limit = bool(spec.get("factor_exclude_limit", True))
    min_listed_raw = _str_opt(spec.get("factor_min_listed_days"))
    min_listed_days = int(min_listed_raw) if min_listed_raw else 0
    winsorize = _str_opt(spec.get("factor_winsorize")) or "none"
    standardize = _str_opt(spec.get("factor_standardize")) or "none"
    neutralize = _parse_neutralize(spec.get("factor_neutralize", "none"))
    cost_raw = _str_opt(spec.get("factor_cost_bps"))
    cost_bps = float(cost_raw) if cost_raw else 0.0
    holdout = _str_opt(spec.get("holdout")) or None
    causality_check = bool(spec.get("factor_causality_check", True))
    min_obs_raw = _str_opt(spec.get("factor_min_obs"))
    min_obs = int(min_obs_raw) if min_obs_raw else 10
    by_industry = bool(spec.get("factor_by_industry", False))
    industry_category = _str_opt(spec.get("factor_industry_category")) or "申万行业"

    logger.info(
      f"Evaluating factor {factor_inst.name or factor_inst.__class__.__name__} "
      f"range={begin}~{end} objs={len(objs)}"
    )
    report = evaluate_factor(
      factor_inst,
      objs,
      begin,
      end,
      msd_host=msd_host,
      benchmark=benchmark,
      fin_fields=cli_fin,
      fin_only_year=fin_only_year,
      fin_lookback_days=fin_lookback_days,
      warmup_bars=warmup_bars,
      periods=periods,
      primary_period=primary_period,
      quantiles=quantiles,
      entry=entry,
      exclude_limit=exclude_limit,
      min_listed_days=min_listed_days,
      winsorize=winsorize,
      standardize=standardize,
      neutralize=neutralize,
      cost_bps=cost_bps,
      holdout=holdout,
      causality_check=causality_check,
      min_obs=min_obs,
      by_industry=by_industry,
      industry_category=industry_category,
    )
    elapsed = time.perf_counter() - t0
    strat_label = factor_inst.name or factor_inst.__class__.__name__
    return write_factor_result(
      target_dir,
      report,
      strategy_name=strat_label,
      elapsed=elapsed,
    )

  if mode not in (RUN_MODE_TIMING, RUN_MODE_PORTFOLIO):
    raise ValueError(f"Unsupported execution mode: '{mode}'.")

  begin = _str_opt(spec.get("begin")) or os.environ.get("MSD_BT_BEGIN") or "2023-01-01"
  end = _str_opt(spec.get("end")) or os.environ.get("MSD_BT_END") or "2025-12-31"
  benchmark = _str_opt(spec.get("benchmark")) or os.environ.get("MSD_BT_BENCHMARK") or "SH000300"
  cash_str = _str_opt(spec.get("cash"))
  cash = float(cash_str) if cash_str else None
  exec_mode_override = _str_opt(spec.get("exec_mode")) or None
  warmup_str = _str_opt(spec.get("warmup_bars"))
  warmup_override = int(warmup_str) if warmup_str else None
  holdout = _str_opt(spec.get("holdout")) or None
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
        f"use_internal_selector is enabled, but '{strategy_path}' does not define a Selector subclass."
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
      "and must be run in portfolio mode."
    )
  if use_portfolio and len(selector_objs) == 0:
    raise ValueError(
      "Portfolio mode requires an explicit candidate stock pool (e.g. 'A_STOCKS_EXCLUDE_ST') "
      "or enabling 'use_internal_selector'."
    )

  selector_dp = MsdSelectorDataProvider(msd_host, date=begin)
  selected_objs = list(selector.execute(selector_dp, init_objs))
  if len(selected_objs) == 0:
    if isinstance(selector, StaticSelector):
      raise ValueError(
        "No target stocks provided. Specify stock codes or pool (e.g. 'SH600000' or 'A_STOCKS_EXCLUDE_ST')."
      )
    raise ValueError(
      f"Selector '{selector.__class__.__name__}' returned 0 stocks from "
      f"{len(init_objs)} candidates on '{begin}'."
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
    _str_opt(spec.get("fin_fields")),
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
    spec.get("fin_only_year")
    or kv_kwargs.get("fin_only_year")
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
  fin_lb_spec = _str_opt(spec.get("fin_lookback_days"))
  fin_lookback_days = (
    int(fin_lb_spec)
    if fin_lb_spec
    else (
      kv_kwargs.get("fin_lookback_days")
      or getattr(req.strategy, "fin_lookback_days", None)
    )
  )

  fee_path = _str_opt(spec.get("fee")) or os.environ.get("MSD_BT_FEE", "")
  fee = Fee.load(fee_path, overrides=kv_kwargs)

  logger.info(
    f"Running {req.strategy.__class__.__name__} mode={mode} "
    f"range={begin}~{end} objs={len(selected_objs) - 1}"
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

  holdout_info: dict[str, Any] | None = None
  if holdout and reports:
    holdout_info = compute_holdout_split(reports[0], holdout)

  if use_portfolio or isinstance(acc, PortfolioAccount):
    r0 = reports[0]
    dates = format_dates_list(r0.date)
    records_raw = getattr(acc, "records", None) or getattr(r0, "records", []) or []
    dp_dates = dp.dates[: dp.bars] if hasattr(dp, "dates") else None
    rebalance_records: list[dict[str, Any]] = []
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
        "pos_shares": float(rec.pos_shares),
        "cash": float(rec.cash),
      })

    trades = getattr(r0, "trades", None) or extract_portfolio_trades(
      account=acc, dp=dp, records=records_raw
    )
    init_cash = float(getattr(acc, "initial_cash", 10000000.0))
    attribution = getattr(r0, "attribution", None) or (
      compute_stock_attribution(trades, initial_cash=init_cash) if trades else []
    )
    metrics = r0.metrics(with_obj=False)
    metrics["initial_cash"] = init_cash
    p_assets = getattr(acc, "portfolio_assets", [])
    metrics["final_assets"] = float(p_assets[-1]) if len(p_assets) > 0 else init_cash
    metrics["pool_size"] = int(getattr(r0, "pool_size", len(selected_objs) - 1))
    metrics["traded_objs_count"] = int(getattr(r0, "traded_objs_count", 0))
    metrics["exec_mode"] = exec_mode

    bench_ror = (
      np.asarray(r0.benchmark_ror, dtype=np.float64)
      if getattr(r0, "benchmark_ror", None) is not None
      else np.zeros(len(dates), dtype=np.float64)
    )
    return write_portfolio_result(
      target_dir,
      strategy_name=req.strategy.__class__.__name__,
      dates=dates,
      ror=np.asarray(r0.ror, dtype=np.float64),
      ror_original=np.asarray(r0.ror_original, dtype=np.float64),
      benchmark_ror=bench_ror,
      nav=np.asarray(r0.price, dtype=np.float64),
      actions=np.asarray(r0.actions, dtype=np.int32),
      benchmark_obj=getattr(r0, "benchmark_obj", benchmark) or benchmark,
      metrics=metrics,
      rebalance_records=rebalance_records,
      trades=trades,
      attribution=attribution,
      holdout=holdout_info,
      diagnostics=getattr(r0, "diagnostics", None),
      elapsed=elapsed,
    )

  diag = getattr(reports[0], "diagnostics", None) if reports else None
  return write_timing_result(
    target_dir,
    strategy_name=req.strategy.__class__.__name__,
    reports=reports,
    benchmark_obj=benchmark,
    holdout=holdout_info,
    diagnostics=diag,
    elapsed=elapsed,
  )


def main() -> int:
  """CLI entry point for subprocess worker execution."""
  parser = argparse.ArgumentParser(description="mbt server subprocess execution worker")
  parser.add_argument("--spec", required=True, help="Path to resolved JSON spec file")
  parser.add_argument("--out-dir", required=True, help="Output directory for Parquet & manifest")
  args = parser.parse_args()

  with open(args.spec, "r", encoding="utf-8") as f:
    spec = json.load(f)

  execute_and_store(spec, Path(args.out_dir))
  return 0


if __name__ == "__main__":
  sys.exit(main())
