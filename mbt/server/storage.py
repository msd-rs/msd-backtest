# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Compressed Parquet result storage and on-demand Polars query/export helpers."""

from __future__ import annotations

import datetime
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence
import zipfile

import numpy as np
import orjson
import polars as pl

from mbt.factor.report import FactorReport, MultiFactorReport
from mbt.run.report import Report, _format_date
from mbt.shared.downsample import minmax_downsample


def sanitize_json(obj: Any) -> Any:
  """Recursively convert NumPy/datetime objects and non-finite floats into strict JSON values."""
  if obj is None:
    return None
  if isinstance(obj, dict):
    return {str(k): sanitize_json(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [sanitize_json(v) for v in obj]
  if isinstance(obj, np.ndarray):
    return [sanitize_json(v) for v in obj.tolist()]
  if isinstance(obj, (np.floating, float)):
    fv = float(obj)
    return None if not math.isfinite(fv) else fv
  if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
    return int(obj)
  if isinstance(obj, (np.bool_, bool)):
    return bool(obj)
  if isinstance(obj, np.datetime64):
    return str(obj)[:10]
  if isinstance(obj, (datetime.datetime, datetime.date)):
    return obj.isoformat()
  return obj


def format_dates_list(dates: np.ndarray | Sequence[Any] | None) -> list[str]:
  """Convert a date array of any mbt representation into ``YYYY-MM-DD`` strings."""
  if dates is None or len(dates) == 0:
    return []
  arr = np.asarray(dates)
  out: list[str] = []
  for i in range(len(arr)):
    val = arr[i]
    s = _format_date(i, int(val) if isinstance(val, (int, np.integer)) else 0, arr)
    if len(s) == 8 and s.isdigit():
      s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
    out.append(s)
  return out


def _compute_drawdown(ror: np.ndarray) -> np.ndarray:
  if len(ror) == 0:
    return np.empty(0, dtype=np.float64)
  equity = 1.0 + np.asarray(ror, dtype=np.float64)
  peaks = np.maximum.accumulate(equity)
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(peaks > 0, (equity - peaks) / peaks, 0.0)


CURVE_FIXED_SCALE = 10_000
_I32_LIMIT = 2_000_000_000


def _series_from_int_array(values: np.ndarray, valid: np.ndarray) -> pl.Series:
  if len(values) == 0:
    return pl.Series(values=np.empty(0, dtype=np.int32), dtype=pl.Int32)
  if not np.any(valid):
    return pl.Series(values=[None] * len(values), dtype=pl.Int32)

  v_valid = values[valid]
  max_abs = int(np.max(np.abs(v_valid)))
  all_valid = bool(np.all(valid))
  if max_abs <= _I32_LIMIT:
    out32 = np.zeros(len(values), dtype=np.int32)
    out32[valid] = v_valid.astype(np.int32)
    s = pl.Series(values=out32, dtype=pl.Int32)
  else:
    out64 = np.zeros(len(values), dtype=np.int64)
    out64[valid] = v_valid.astype(np.int64)
    s = pl.Series(values=out64, dtype=pl.Int64)

  if not all_valid:
    mask = pl.Series(values=valid, dtype=pl.Boolean)
    s = (
      pl.DataFrame({"v": s, "m": mask})
      .select(pl.when(pl.col("m")).then(pl.col("v")).otherwise(None))
      .to_series()
    )
  return s


def _encode_fixed_int_1d(arr: np.ndarray, scale: int = CURVE_FIXED_SCALE) -> pl.Series:
  """Quantize 1D float series to scaled fixed-point integers preserving nulls."""
  raw = np.asarray(arr, dtype=np.float64)
  valid = np.isfinite(raw)
  out = np.zeros(len(raw), dtype=np.int64)
  if np.any(valid):
    out[valid] = np.rint(raw[valid] * float(scale)).astype(np.int64)
  return _series_from_int_array(out, valid)


def _encode_delta_int_1d(arr: np.ndarray, scale: int = CURVE_FIXED_SCALE) -> pl.Series:
  """Quantize 1D float series to 4dp fixed-point integers and delta-encode finite samples."""
  raw = np.asarray(arr, dtype=np.float64)
  valid = np.isfinite(raw)
  out = np.zeros(len(raw), dtype=np.int64)
  if np.any(valid):
    q = np.rint(raw[valid] * float(scale)).astype(np.int64)
    deltas = np.empty_like(q)
    deltas[0] = q[0]
    if len(q) > 1:
      deltas[1:] = np.diff(q)
    out[valid] = deltas
  return _series_from_int_array(out, valid)


def _decode_curve_lf(lf: pl.LazyFrame, encoding: dict[str, Any] | None) -> pl.LazyFrame:
  """Decode delta- and fixed-point-encoded integer curve columns back to floats."""
  if not encoding:
    return lf
  kind = encoding.get("kind")
  if kind != "delta_fixed_int":
    raise ValueError(
      f"Unsupported curve encoding '{kind}'. Please restart or upgrade mbt-server."
    )
  scale = float(encoding.get("scale", CURVE_FIXED_SCALE))
  schema_names = set(lf.collect_schema().names())
  delta_cols = [c for c in (encoding.get("delta_columns") or []) if c in schema_names]
  fixed_cols = [c for c in (encoding.get("fixed_columns") or []) if c in schema_names]

  exprs: list[pl.Expr] = []
  for c in delta_cols:
    exprs.append(
      pl.when(pl.col(c).is_not_null())
      .then(pl.col(c).cast(pl.Int64).fill_null(0).cum_sum().cast(pl.Float64) / scale)
      .otherwise(None)
      .alias(c)
    )
  for c in fixed_cols:
    exprs.append((pl.col(c).cast(pl.Float64) / scale).alias(c))

  return lf.with_columns(exprs) if exprs else lf


def _write_parquet(df: pl.DataFrame, path: Path) -> dict[str, Any]:
  df.write_parquet(path, compression="zstd", compression_level=6)
  return {
    "file": path.name,
    "rows": int(df.height),
    "columns": [{"name": c, "dtype": str(dt)} for c, dt in df.schema.items()],
  }


def _write_manifest(out_dir: Path, manifest: dict[str, Any]) -> Path:
  out_dir.mkdir(parents=True, exist_ok=True)
  target = out_dir / "manifest.json"
  tmp = out_dir / "manifest.json.tmp"
  clean = sanitize_json(manifest)
  payload = orjson.dumps(clean, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY)
  tmp.write_bytes(payload + b"\n")
  os.replace(tmp, target)
  return target


def write_select_result(
  out_dir: Path,
  *,
  strategy_name: str,
  select_date: str,
  stocks: list[str],
  factors: pl.DataFrame | None,
  elapsed: float,
) -> dict[str, Any]:
  """Persist stock selector output as zstd Parquet + ``manifest.json``."""
  out_dir.mkdir(parents=True, exist_ok=True)
  if factors is not None and isinstance(factors, pl.DataFrame) and not factors.is_empty():
    df = factors.sort("obj") if "obj" in factors.columns else factors
  else:
    df = pl.DataFrame({"obj": pl.Series("obj", stocks, dtype=pl.Utf8)})

  tbl_info = _write_parquet(df, out_dir / "select_factors.parquet")
  manifest = {
    "version": 1,
    "mode": "select",
    "strategy_name": strategy_name,
    "elapsed": float(elapsed),
    "summary": {
      "strategy_name": strategy_name,
      "select_date": select_date,
      "selected_count": len(stocks),
      "stocks_preview": stocks[:20],
      "columns": list(df.columns),
      "elapsed": float(elapsed),
    },
    "tables": {"select_factors": tbl_info},
    "curves": {},
  }
  _write_manifest(out_dir, manifest)
  return manifest


def write_timing_result(
  out_dir: Path,
  *,
  strategy_name: str,
  reports: list[Report],
  benchmark_obj: str,
  holdout: dict[str, Any] | None,
  diagnostics: dict[str, Any] | None,
  elapsed: float,
) -> dict[str, Any]:
  """Persist timing strategy metrics and per-stock/summary curves to Parquet."""
  out_dir.mkdir(parents=True, exist_ok=True)
  metric_rows = [rep.metrics(with_obj=True) for rep in reports]
  if metric_rows:
    df_metrics = pl.DataFrame(metric_rows)
  else:
    df_metrics = pl.DataFrame(schema={"obj": pl.Utf8, "profit": pl.Float64})

  curve_frames: list[pl.DataFrame] = []
  for rep in reports:
    dates_str = format_dates_list(rep.date)
    n = len(dates_str)
    ror = np.asarray(rep.ror, dtype=np.float64)[:n]
    orig = np.asarray(rep.ror_original, dtype=np.float64)[:n] if len(rep.ror_original) >= n else np.zeros(n, dtype=np.float64)
    bench = (
      np.asarray(rep.benchmark_ror, dtype=np.float64)[:n]
      if rep.benchmark_ror is not None and len(rep.benchmark_ror) >= n
      else np.zeros(n, dtype=np.float64)
    )
    dd = _compute_drawdown(ror)
    acts = np.asarray(rep.actions, dtype=np.int32)[:n] if len(rep.actions) >= n else np.zeros(n, dtype=np.int32)
    acts_i8 = np.clip(acts, -128, 127).astype(np.int8)
    px = np.asarray(rep.price, dtype=np.float64)[:n] if len(rep.price) >= n else np.zeros(n, dtype=np.float64)
    curve_frames.append(
      pl.DataFrame({
        "obj": np.repeat(str(rep.obj), n),
        "bar_idx": np.arange(n, dtype=np.int32),
        "date": dates_str,
        "ror": _encode_delta_int_1d(ror),
        "ror_original": _encode_delta_int_1d(orig),
        "benchmark_ror": _encode_delta_int_1d(bench),
        "drawdown": _encode_delta_int_1d(dd),
        "actions": acts_i8,
        "price": _encode_delta_int_1d(px),
      })
    )

  if curve_frames:
    df_curves = pl.concat(curve_frames, how="vertical_relaxed").sort(["obj", "bar_idx"])
  else:
    df_curves = pl.DataFrame(
      schema={
        "obj": pl.Utf8,
        "bar_idx": pl.Int32,
        "date": pl.Utf8,
        "ror": pl.Int32,
        "ror_original": pl.Int32,
        "benchmark_ror": pl.Int32,
        "drawdown": pl.Int32,
        "actions": pl.Int8,
        "price": pl.Int32,
      }
    )
  _write_parquet(df_curves, out_dir / "curves.parquet")

  has_summary_curve = False
  summary_points = 0
  if len(reports) > 1:
    base_len = len(reports[0].ror)
    aligned = [r for r in reports if len(r.ror) == base_len]
    if aligned and base_len > 0:
      dates_str = format_dates_list(aligned[0].date)[:base_len]
      mean_ror = np.nanmean(
        np.vstack([np.asarray(r.ror, dtype=np.float64)[:base_len] for r in aligned]),
        axis=0,
      )
      mean_orig = np.nanmean(
        np.vstack([np.asarray(r.ror_original, dtype=np.float64)[:base_len] for r in aligned]),
        axis=0,
      )
      bench_arr = (
        np.asarray(aligned[0].benchmark_ror, dtype=np.float64)[:base_len]
        if aligned[0].benchmark_ror is not None and len(aligned[0].benchmark_ror) >= base_len
        else np.zeros(base_len, dtype=np.float64)
      )
      mean_dd = _compute_drawdown(mean_ror)
      df_sum = pl.DataFrame({
        "bar_idx": np.arange(base_len, dtype=np.int32),
        "date": dates_str,
        "ror": _encode_delta_int_1d(mean_ror),
        "ror_original": _encode_delta_int_1d(mean_orig),
        "benchmark_ror": _encode_delta_int_1d(bench_arr),
        "drawdown": _encode_delta_int_1d(mean_dd),
      })
      _write_parquet(df_sum, out_dir / "summary_curve.parquet")
      has_summary_curve = True
      summary_points = base_len

  tbl_metrics = _write_parquet(df_metrics, out_dir / "metrics.parquet")

  total = len(reports)
  profits = [float(r.profit) for r in reports]
  sharpes = [float(r.sharpe) for r in reports]
  mdds = [float(r.max_drawdown) for r in reports]
  active_cnt = sum(1 for r in reports if int(r.trade_count) > 0)
  pos_cnt = sum(1 for r in reports if float(r.profit) > 0)
  beat_bh = sum(1 for r in reports if float(r.profit) > float(r.profit_original))
  beat_bench = sum(1 for r in reports if float(r.profit) > float(r.benchmark_profit))
  bench_profit = float(reports[0].benchmark_profit) if reports else 0.0
  eff_bench = (reports[0].benchmark_obj or benchmark_obj) if reports else benchmark_obj

  manifest = {
    "version": 1,
    "mode": "timing",
    "strategy_name": strategy_name,
    "elapsed": float(elapsed),
    "summary": {
      "strategy_name": strategy_name,
      "total_objs": total,
      "active_objs": active_cnt,
      "positive_count": pos_cnt,
      "beat_buy_hold_count": beat_bh,
      "beat_benchmark_count": beat_bench,
      "benchmark_obj": eff_bench,
      "benchmark_profit": bench_profit,
      "mean_profit": float(np.mean(profits)) if profits else 0.0,
      "median_profit": float(np.median(profits)) if profits else 0.0,
      "mean_sharpe": float(np.mean(sharpes)) if sharpes else 0.0,
      "mean_max_drawdown": float(np.mean(mdds)) if mdds else 0.0,
      "has_summary_curve": has_summary_curve,
      "objs": [str(r.obj) for r in reports],
      "holdout": holdout,
      "diagnostics": diagnostics,
      "elapsed": float(elapsed),
    },
    "tables": {"metrics": tbl_metrics},
    "curves": {
      "timing_curves": {
        "file": "curves.parquet",
        "has_summary": has_summary_curve,
        "summary_file": "summary_curve.parquet" if has_summary_curve else None,
        "summary_points": summary_points,
        "objs": [str(r.obj) for r in reports],
        "benchmark_obj": eff_bench,
        "encoding": {
          "kind": "delta_fixed_int",
          "scale": CURVE_FIXED_SCALE,
          "delta_columns": ["ror", "ror_original", "benchmark_ror", "drawdown", "price"],
          "summary_delta_columns": ["ror", "ror_original", "benchmark_ror", "drawdown"],
        },
      }
    },
  }
  _write_manifest(out_dir, manifest)
  return manifest


def write_portfolio_result(
  out_dir: Path,
  *,
  strategy_name: str,
  dates: list[str],
  ror: np.ndarray,
  ror_original: np.ndarray,
  benchmark_ror: np.ndarray,
  nav: np.ndarray,
  actions: np.ndarray,
  benchmark_obj: str,
  metrics: dict[str, Any],
  rebalance_records: list[dict[str, Any]],
  trades: list[dict[str, Any]],
  attribution: list[dict[str, Any]],
  holdout: dict[str, Any] | None,
  diagnostics: dict[str, Any] | None,
  elapsed: float,
) -> dict[str, Any]:
  """Persist portfolio strategy curve, rebalances, trades, and attribution to Parquet."""
  out_dir.mkdir(parents=True, exist_ok=True)
  n = len(dates)
  ror_arr = np.asarray(ror, dtype=np.float64)[:n]
  orig_arr = np.asarray(ror_original, dtype=np.float64)[:n] if len(ror_original) >= n else np.zeros(n, dtype=np.float64)
  bench_arr = np.asarray(benchmark_ror, dtype=np.float64)[:n] if len(benchmark_ror) >= n else np.zeros(n, dtype=np.float64)
  nav_arr = np.asarray(nav, dtype=np.float64)[:n] if len(nav) >= n else (1.0 + ror_arr)
  act_arr = np.asarray(actions, dtype=np.int32)[:n] if len(actions) >= n else np.zeros(n, dtype=np.int32)
  act_i8 = np.clip(act_arr, -128, 127).astype(np.int8)
  dd_arr = _compute_drawdown(ror_arr)

  df_curve = pl.DataFrame({
    "bar_idx": np.arange(n, dtype=np.int32),
    "date": dates,
    "ror": _encode_delta_int_1d(ror_arr),
    "ror_original": _encode_delta_int_1d(orig_arr),
    "benchmark_ror": _encode_delta_int_1d(bench_arr),
    "drawdown": _encode_delta_int_1d(dd_arr),
    "nav": _encode_delta_int_1d(nav_arr),
    "actions": act_i8,
  })
  _write_parquet(df_curve, out_dir / "curve.parquet")

  if rebalance_records:
    df_reb = pl.DataFrame(rebalance_records)
  else:
    df_reb = pl.DataFrame(
      schema={
        "bar": pl.Int32,
        "date": pl.Utf8,
        "obj": pl.Utf8,
        "action": pl.Utf8,
        "shares": pl.Float64,
        "deal_price": pl.Float64,
        "amount": pl.Float64,
        "fee": pl.Float64,
        "pos_shares": pl.Float64,
        "cash": pl.Float64,
      }
    )

  if trades:
    df_trades = pl.DataFrame(trades)
  else:
    df_trades = pl.DataFrame(
      schema={
        "obj": pl.Utf8,
        "entry_bar": pl.Int32,
        "entry_date": pl.Utf8,
        "exit_bar": pl.Int32,
        "exit_date": pl.Utf8,
        "profit_rate": pl.Float64,
        "profit_amount": pl.Float64,
      }
    )

  if attribution:
    df_attrib = pl.DataFrame(attribution)
  else:
    df_attrib = pl.DataFrame(
      schema={
        "obj": pl.Utf8,
        "trade_count": pl.Int32,
        "win_rate": pl.Float64,
        "total_profit": pl.Float64,
        "capital_contribution": pl.Float64,
        "avg_return": pl.Float64,
      }
    )

  tbl_reb = _write_parquet(df_reb, out_dir / "rebalances.parquet")
  tbl_trades = _write_parquet(df_trades, out_dir / "trades.parquet")
  tbl_attrib = _write_parquet(df_attrib, out_dir / "attribution.parquet")

  manifest = {
    "version": 1,
    "mode": "portfolio",
    "strategy_name": strategy_name,
    "elapsed": float(elapsed),
    "summary": {
      "strategy_name": strategy_name,
      "benchmark_obj": benchmark_obj,
      "metrics": metrics,
      "pool_size": int(metrics.get("pool_size", 0)),
      "traded_objs_count": int(metrics.get("traded_objs_count", 0)),
      "rebalance_records_count": int(df_reb.height),
      "trades_count": int(df_trades.height),
      "exec_mode": str(metrics.get("exec_mode", "close")),
      "holdout": holdout,
      "diagnostics": diagnostics,
      "elapsed": float(elapsed),
    },
    "tables": {
      "rebalances": tbl_reb,
      "trades": tbl_trades,
      "attribution": tbl_attrib,
    },
    "curves": {
      "portfolio": {
        "file": "curve.parquet",
        "points": n,
        "benchmark_obj": benchmark_obj,
        "encoding": {
          "kind": "delta_fixed_int",
          "scale": CURVE_FIXED_SCALE,
          "delta_columns": ["ror", "ror_original", "benchmark_ror", "drawdown", "nav"],
        },
      }
    },
  }
  _write_manifest(out_dir, manifest)
  return manifest


def write_factor_result(
  out_dir: Path,
  report: FactorReport | MultiFactorReport,
  *,
  strategy_name: str,
  elapsed: float,
) -> dict[str, Any]:
  """Persist single or multi-factor evaluation report to zstd Parquet + ``manifest.json``."""
  out_dir.mkdir(parents=True, exist_ok=True)
  is_multi = isinstance(report, MultiFactorReport)
  reports = report.ranked_reports() if is_multi else [report]
  corr_matrix = report.correlation_matrix if is_multi else {}

  lb_rows = []
  ic_rows = []
  q_rows = []
  annual_rows = []
  industry_rows = []
  daily_frames: list[pl.DataFrame] = []
  max_q = max((r.quantiles for r in reports), default=5)

  for r in reports:
    lb_rows.append({
      **r.summary_row(),
      "direction": int(r.direction),
      "start_date": r.start_date,
      "end_date": r.end_date,
      "bars": int(r.bars),
      "n_objs": int(r.n_objs),
      "expression": r.expression,
    })

    for p in r.periods:
      ic_rows.append({"factor": r.name, **r.ic_summaries[p].to_dict()})
      qa = r.quantile_analyses[p]
      for qm in qa.quantiles:
        q_rows.append({"factor": r.name, "period": int(p), "kind": "bucket", **qm.to_dict()})
      q_rows.append({
        "factor": r.name,
        "period": int(p),
        "kind": "spread",
        **qa.long_excess_market.to_dict(),
      })
      q_rows.append({
        "factor": r.name,
        "period": int(p),
        "kind": "spread",
        **qa.long_short.to_dict(),
      })
      if qa.long_excess_benchmark is not None:
        q_rows.append({
          "factor": r.name,
          "period": int(p),
          "kind": "spread",
          **qa.long_excess_benchmark.to_dict(),
        })

      dates_str = format_dates_list(r.dates)
      n = len(dates_str)
      ic_s = np.asarray(r.ic_series[p], dtype=np.float64)[:n]
      ric_s = np.asarray(r.rank_ic_series[p], dtype=np.float64)[:n]
      cum_ric = np.cumsum(np.where(np.isfinite(ric_s), ric_s, 0.0))
      bench_nav = (
        np.asarray(qa.benchmark_nav, dtype=np.float64)[:n]
        if qa.benchmark_nav is not None and len(qa.benchmark_nav) >= n
        else np.full(n, np.nan, dtype=np.float64)
      )
      cols: dict[str, Any] = {
        "factor": np.repeat(r.name, n),
        "period": np.full(n, int(p), dtype=np.int32),
        "bar_idx": np.arange(n, dtype=np.int32),
        "date": dates_str,
        "ic": _encode_fixed_int_1d(ic_s),
        "rank_ic": _encode_fixed_int_1d(ric_s),
        "cum_rank_ic": _encode_delta_int_1d(cum_ric),
        "long_excess_nav": _encode_delta_int_1d(np.asarray(qa.long_excess_nav, dtype=np.float64)[:n]),
        "long_short_nav": _encode_delta_int_1d(np.asarray(qa.long_short_nav, dtype=np.float64)[:n]),
        "market_nav": _encode_delta_int_1d(np.asarray(qa.market_nav, dtype=np.float64)[:n]),
        "benchmark_nav": _encode_delta_int_1d(bench_nav),
      }
      for qi in range(1, max_q + 1):
        q_key = f"Q{qi}"
        arr = qa.quantile_navs.get(q_key)
        nav_vec = (
          np.asarray(arr, dtype=np.float64)[:n]
          if arr is not None and len(arr) >= n
          else np.full(n, np.nan, dtype=np.float64)
        )
        cols[f"{q_key}_nav"] = _encode_delta_int_1d(nav_vec)
      daily_frames.append(pl.DataFrame(cols))

    for row in r.annual_breakdown:
      annual_rows.append({"factor": r.name, **row})
    for row in r.industry_breakdown:
      industry_rows.append({"factor": r.name, **row})

  df_lb = pl.DataFrame(lb_rows) if lb_rows else pl.DataFrame(schema={"factor": pl.Utf8})
  df_ic = pl.DataFrame(ic_rows) if ic_rows else pl.DataFrame(schema={"factor": pl.Utf8, "period": pl.Int32})
  df_q = pl.DataFrame(q_rows) if q_rows else pl.DataFrame(schema={"factor": pl.Utf8, "period": pl.Int32, "label": pl.Utf8})
  df_ann = (
    pl.DataFrame(annual_rows)
    if annual_rows
    else pl.DataFrame(
      schema={
        "factor": pl.Utf8,
        "year": pl.Utf8,
        "bars": pl.Int64,
        "valid_ic_bars": pl.Int64,
        "ic_mean": pl.Float64,
        "rank_ic_mean": pl.Float64,
        "rank_icir": pl.Float64,
        "rank_ic_positive_ratio": pl.Float64,
        "top_excess_return": pl.Float64,
      }
    )
  )
  df_ind = (
    pl.DataFrame(industry_rows)
    if industry_rows
    else pl.DataFrame(
      schema={
        "factor": pl.Utf8,
        "industry": pl.Utf8,
        "n_stocks": pl.Int64,
        "avg_valid_stocks": pl.Float64,
        "valid_bars": pl.Int64,
        "rank_ic_mean": pl.Float64,
        "rank_icir": pl.Float64,
        "ic_mean": pl.Float64,
      }
    )
  )

  if daily_frames:
    df_daily = pl.concat(daily_frames, how="vertical_relaxed").sort(
      ["factor", "period", "bar_idx"]
    )
  else:
    df_daily = pl.DataFrame(
      schema={
        "factor": pl.Utf8,
        "period": pl.Int32,
        "bar_idx": pl.Int32,
        "date": pl.Utf8,
        "ic": pl.Int32,
        "rank_ic": pl.Int32,
        "cum_rank_ic": pl.Int32,
        "long_excess_nav": pl.Int32,
        "long_short_nav": pl.Int32,
        "market_nav": pl.Int32,
        "benchmark_nav": pl.Int32,
      }
    )
  _write_parquet(df_daily, out_dir / "daily_series.parquet")

  tbl_lb = _write_parquet(df_lb, out_dir / "leaderboard.parquet")
  tbl_ic = _write_parquet(df_ic, out_dir / "ic_summary.parquet")
  tbl_q = _write_parquet(df_q, out_dir / "quantiles.parquet")
  tbl_ann = _write_parquet(df_ann, out_dir / "annual.parquet")
  tbl_ind = _write_parquet(df_ind, out_dir / "industry.parquet")

  prim_rep = reports[0]
  factor_details = []
  for r in reports:
    factor_details.append({
      "name": r.name,
      "direction": int(r.direction),
      "expression": r.expression,
      "author": r.author,
      "start_date": r.start_date,
      "end_date": r.end_date,
      "bars": int(r.bars),
      "n_objs": int(r.n_objs),
      "quantiles": int(r.quantiles),
      "entry": r.entry,
      "winsorize": r.winsorize,
      "standardize": r.standardize,
      "neutralize": list(r.neutralize),
      "cost_bps": float(r.cost_bps),
      "benchmark": r.benchmark,
      "summary": r.summary_row(),
      "turnover": r.turnover,
      "holdout": r.holdout_breakdown,
      "diagnostics": r.diagnostics,
      "monotonicities": {
        str(p): float(r.quantile_analyses[p].monotonicity) for p in r.periods
      },
    })

  manifest = {
    "version": 1,
    "mode": "factor",
    "strategy_name": strategy_name,
    "elapsed": float(elapsed),
    "summary": {
      "strategy_name": strategy_name,
      "is_multi_factor": is_multi,
      "primary_period": int(prim_rep.primary_period),
      "periods": [int(p) for p in prim_rep.periods],
      "quantiles": int(prim_rep.quantiles),
      "factor_names": [r.name for r in reports],
      "factors": factor_details,
      "correlation_matrix": corr_matrix,
      "elapsed": float(elapsed),
    },
    "tables": {
      "leaderboard": tbl_lb,
      "ic_summary": tbl_ic,
      "quantiles": tbl_q,
      "annual": tbl_ann,
      "industry": tbl_ind,
    },
    "curves": {
      "factor_daily": {
        "file": "daily_series.parquet",
        "factors": [r.name for r in reports],
        "periods": [int(p) for p in prim_rep.periods],
        "quantiles": int(max_q),
        "encoding": {
          "kind": "delta_fixed_int",
          "scale": CURVE_FIXED_SCALE,
          "delta_columns": [
            "cum_rank_ic",
            "long_excess_nav",
            "long_short_nav",
            "market_nav",
            "benchmark_nav",
            *[f"Q{qi}_nav" for qi in range(1, max_q + 1)],
          ],
          "fixed_columns": ["ic", "rank_ic"],
        },
      }
    },
  }
  _write_manifest(out_dir, manifest)
  return manifest


def read_manifest(result_dir: Path | str) -> dict[str, Any]:
  """Read ``manifest.json`` from a run result directory."""
  m_path = Path(result_dir) / "manifest.json"
  if not m_path.is_file():
    raise FileNotFoundError(f"Run manifest not found in '{result_dir}'.")
  return json.loads(m_path.read_text(encoding="utf-8"))


def query_table(
  result_dir: Path | str,
  table_name: str,
  *,
  page: int = 1,
  page_size: int = 100,
  sort_by: str | None = None,
  sort_dir: str = "desc",
  q: str | None = None,
  factor: str | None = None,
  period: int | None = None,
) -> dict[str, Any]:
  """Lazily scan, filter, sort, and paginate a run Parquet table."""
  base_dir = Path(result_dir)
  manifest = read_manifest(base_dir)
  tables = manifest.get("tables") or {}
  if table_name not in tables:
    raise KeyError(
      f"Unknown table '{table_name}'. Available tables: {sorted(tables.keys())}."
    )

  tbl_meta = tables[table_name]
  pq_path = base_dir / tbl_meta["file"]
  if not pq_path.is_file():
    raise FileNotFoundError(f"Table file '{pq_path}' missing.")

  lf = pl.scan_parquet(pq_path)
  schema = lf.collect_schema()
  col_names = schema.names()

  if factor and "factor" in col_names:
    lf = lf.filter(pl.col("factor") == factor)
  if period is not None and "period" in col_names:
    lf = lf.filter(pl.col("period") == int(period))

  if q and q.strip():
    q_low = q.strip().lower()
    if col_names:
      pred = pl.any_horizontal([
        pl.col(c).cast(pl.Utf8).fill_null("").str.to_lowercase().str.contains(q_low, literal=True)
        for c in col_names
      ])
      lf = lf.filter(pred)

  total_rows = int(lf.select(pl.len()).collect().item())

  if sort_by:
    if sort_by not in col_names:
      raise ValueError(
        f"Invalid sort column '{sort_by}' for table '{table_name}'. Available: {col_names}."
      )
    desc = sort_dir.strip().lower() != "asc"
    lf = lf.sort(sort_by, descending=desc, nulls_last=True)

  safe_page = max(1, int(page))
  safe_size = max(1, min(2000, int(page_size)))
  offset = (safe_page - 1) * safe_size
  page_df = lf.slice(offset, safe_size).collect()
  rows = sanitize_json(page_df.to_dicts())

  return {
    "table": table_name,
    "columns": [{"name": c, "dtype": str(dt)} for c, dt in schema.items()],
    "total_rows": total_rows,
    "page": safe_page,
    "page_size": safe_size,
    "total_pages": max(1, math.ceil(total_rows / safe_size)) if total_rows > 0 else 1,
    "rows": rows,
  }


def _downsample_multi_series(
  df: pl.DataFrame,
  ds_keys: Sequence[str],
  max_points: int,
) -> tuple[pl.DataFrame, bool]:
  n = df.height
  if n <= max_points or max_points < 4:
    return df, False

  valid_keys = [k for k in ds_keys if k in df.columns]
  if not valid_keys:
    step = max(1, n // max_points)
    idx = np.arange(0, n, step, dtype=np.int64)
    if idx[-1] != n - 1:
      idx = np.append(idx, n - 1)
    return df[idx.tolist()], True

  per_key_budget = max(4, max_points // len(valid_keys))
  parts: list[np.ndarray] = [np.array([0, n - 1], dtype=np.int64)]
  for k in valid_keys:
    arr = df[k].to_numpy().astype(np.float64, copy=False)
    xs, _ = minmax_downsample(arr, 0, n, max_points=per_key_budget)
    if len(xs) > 0:
      parts.append(xs)

  combined = np.unique(np.concatenate(parts))
  return df[combined.tolist()], True


def query_curve(
  result_dir: Path | str,
  *,
  obj: str | None = None,
  factor: str | None = None,
  period: int | None = None,
  start_idx: int = 0,
  end_idx: int | None = None,
  max_points: int = 2000,
) -> dict[str, Any]:
  """Load and viewport-downsample time-series curves from Parquet on demand."""
  base_dir = Path(result_dir)
  manifest = read_manifest(base_dir)
  mode = manifest.get("mode")

  if mode == "select":
    raise ValueError("Select mode results do not contain time-series curves.")

  meta: dict[str, Any] = {"mode": mode}
  ds_keys: list[str]
  enc_meta: dict[str, Any] | None = None

  if mode == "timing":
    tc_meta = (manifest.get("curves") or {}).get("timing_curves") or {}
    tc_enc = tc_meta.get("encoding")
    has_summary = bool(tc_meta.get("has_summary", False))
    objs = list(tc_meta.get("objs") or [])
    use_summary = (obj is None or obj == "" or obj == "__SUMMARY__") and has_summary
    if use_summary:
      pq_path = base_dir / str(tc_meta["summary_file"])
      lf = pl.scan_parquet(pq_path)
      series_cols = ["ror", "ror_original", "benchmark_ror", "drawdown"]
      if isinstance(tc_enc, dict):
        enc_meta = {
          "kind": tc_enc.get("kind"),
          "scale": tc_enc.get("scale", CURVE_FIXED_SCALE),
          "delta_columns": tc_enc.get("summary_delta_columns", tc_enc.get("delta_columns", [])),
          "fixed_columns": [],
        }
      meta.update({"target": "__SUMMARY__", "benchmark_obj": tc_meta.get("benchmark_obj", "")})
    else:
      target_obj = obj if (obj and obj != "__SUMMARY__") else (objs[0] if objs else "")
      if objs and target_obj not in objs:
        raise KeyError(f"Unknown stock '{target_obj}' in timing curves.")
      pq_path = base_dir / str(tc_meta.get("file", "curves.parquet"))
      lf = pl.scan_parquet(pq_path).filter(pl.col("obj") == target_obj)
      series_cols = ["ror", "ror_original", "benchmark_ror", "drawdown", "actions", "price"]
      enc_meta = tc_enc if isinstance(tc_enc, dict) else None
      meta.update({"target": target_obj, "benchmark_obj": tc_meta.get("benchmark_obj", "")})
    ds_keys = ["ror", "drawdown"]

  elif mode == "portfolio":
    p_meta = (manifest.get("curves") or {}).get("portfolio") or {}
    pq_path = base_dir / str(p_meta.get("file", "curve.parquet"))
    lf = pl.scan_parquet(pq_path)
    series_cols = ["ror", "ror_original", "benchmark_ror", "drawdown", "nav", "actions"]
    ds_keys = ["ror", "drawdown"]
    p_enc = p_meta.get("encoding")
    enc_meta = p_enc if isinstance(p_enc, dict) else None
    meta.update({"target": "PORTFOLIO", "benchmark_obj": p_meta.get("benchmark_obj", "")})

  elif mode == "factor":
    f_meta = (manifest.get("curves") or {}).get("factor_daily") or {}
    factors = list(f_meta.get("factors") or [])
    periods = list(f_meta.get("periods") or [])
    target_factor = factor or (factors[0] if factors else "")
    if factors and target_factor not in factors:
      raise KeyError(f"Unknown factor '{target_factor}'. Available: {factors}.")
    target_period = int(period) if period is not None else int(
      (manifest.get("summary") or {}).get("primary_period") or (periods[0] if periods else 5)
    )
    if periods and target_period not in periods:
      raise KeyError(f"Unknown period '{target_period}'. Available: {periods}.")
    pq_path = base_dir / str(f_meta.get("file", "daily_series.parquet"))
    lf = pl.scan_parquet(pq_path).filter(
      (pl.col("factor") == target_factor) & (pl.col("period") == target_period)
    )
    schema = lf.collect_schema()
    series_cols = [
      c for c in schema.names() if c not in ("factor", "period", "bar_idx", "date")
    ]
    ds_keys = ["cum_rank_ic", "long_excess_nav"]
    f_enc = f_meta.get("encoding")
    enc_meta = f_enc if isinstance(f_enc, dict) else None
    meta.update({"factor": target_factor, "period": target_period})

  else:
    raise ValueError(f"Unsupported result mode '{mode}' for curve queries.")

  total_points = int(lf.select(pl.len()).collect().item())
  if total_points == 0:
    return {
      "total_points": 0,
      "start_idx": 0,
      "end_idx": 0,
      "downsampled": False,
      "indices": [],
      "dates": [],
      "series": {c: [] for c in series_cols},
      "meta": meta,
    }

  s = max(0, int(start_idx))
  e = total_points if end_idx is None else min(total_points, int(end_idx))
  if s >= e:
    s = 0
    e = total_points

  sub_df = _decode_curve_lf(lf.sort("bar_idx"), enc_meta).slice(s, e - s).collect()
  safe_max = max(4, min(20000, int(max_points)))
  sampled_df, downsampled = _downsample_multi_series(sub_df, ds_keys, safe_max)

  indices = [int(x) for x in sampled_df["bar_idx"].to_list()]
  dates = [str(x) for x in sampled_df["date"].to_list()]
  series = {
    col: sanitize_json(sampled_df[col].to_list())
    for col in series_cols
    if col in sampled_df.columns
  }

  return {
    "total_points": total_points,
    "start_idx": s,
    "end_idx": e,
    "downsampled": downsampled,
    "indices": indices,
    "dates": dates,
    "series": series,
    "meta": meta,
  }


def export_run_data(
  result_dir: Path | str,
  fmt: str,
  table: str | None = None,
) -> tuple[bytes, str, str]:
  """Export run results on demand as ``csv``, ``xlsx``, ``json``, or ``parquet``."""
  base_dir = Path(result_dir)
  manifest = read_manifest(base_dir)
  tables = manifest.get("tables") or {}
  norm_fmt = fmt.strip().lower()
  mode = str(manifest.get("mode") or "run")

  default_table_map = {
    "select": "select_factors",
    "timing": "metrics",
    "portfolio": "rebalances",
    "factor": "leaderboard",
  }

  if norm_fmt == "csv":
    target_tbl = table or default_table_map.get(mode) or (next(iter(tables.keys()), ""))
    if target_tbl not in tables:
      raise KeyError(f"Unknown table '{target_tbl}'. Available: {sorted(tables.keys())}.")
    df = pl.read_parquet(base_dir / tables[target_tbl]["file"])
    return df.write_csv().encode("utf-8-sig"), "text/csv; charset=utf-8", f"{target_tbl}.csv"

  if norm_fmt == "xlsx":
    import xlsxwriter

    buf = io.BytesIO()
    wb = xlsxwriter.Workbook(buf, {"in_memory": True, "nan_inf_to_errors": True})
    target_names = [table] if table else list(tables.keys())
    for tname in target_names:
      if tname not in tables:
        wb.close()
        raise KeyError(f"Unknown table '{tname}'. Available: {sorted(tables.keys())}.")
      df = pl.read_parquet(base_dir / tables[tname]["file"])
      df.write_excel(workbook=wb, worksheet=tname[:31])
    wb.close()
    fname = f"{table}.xlsx" if table else f"{mode}_report.xlsx"
    return (
      buf.getvalue(),
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
      fname,
    )

  if norm_fmt == "json":
    if table:
      if table not in tables:
        raise KeyError(f"Unknown table '{table}'. Available: {sorted(tables.keys())}.")
      df = pl.read_parquet(base_dir / tables[table]["file"])
      payload = sanitize_json({"table": table, "rows": df.to_dicts()})
      fname = f"{table}.json"
    else:
      tbl_data = {
        tname: sanitize_json(pl.read_parquet(base_dir / tinfo["file"]).to_dicts())
        for tname, tinfo in tables.items()
      }
      payload = sanitize_json({"manifest": manifest, "tables": tbl_data})
      fname = f"{mode}_report.json"
    return (
      orjson.dumps(payload, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY),
      "application/json",
      fname,
    )

  if norm_fmt == "parquet":
    if table:
      if table not in tables:
        raise KeyError(f"Unknown table '{table}'. Available: {sorted(tables.keys())}.")
      pq_bytes = (base_dir / tables[table]["file"]).read_bytes()
      return pq_bytes, "application/octet-stream", f"{table}.parquet"

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
      for p in sorted(base_dir.glob("*.parquet")):
        zf.write(p, arcname=p.name)
      m_path = base_dir / "manifest.json"
      if m_path.is_file():
        zf.write(m_path, arcname="manifest.json")
    return buf.getvalue(), "application/zip", f"{mode}_parquet.zip"

  raise ValueError(
    f"Unsupported export format '{fmt}'. Expected one of: csv, xlsx, json, parquet."
  )
