#!/usr/bin/env python3
"""mbt.factor 长周期全市场多因子评测性能剖析脚本.

复现场景:
  14 个经典因子 × 全市场非 ST 股票 (4,730 只) × 2005-01-04 ~ 2021-12-31 (4,132 根日 K 线)
并逐阶段精准计时:
  1. MSD load_data + Panel 构建与切片
  2. ExpressionFactor.compute(panel) 因子矩阵计算
  3. check_factor_causality 因果探针校验
  4. compute_forward_returns + compute_log_mktcap
  5. 逐因子评估循环 (preprocess_factor / IC / Quantile / Turnover / Annual)
  6. compute_factor_correlation_matrix 因子相关性矩阵

用法:
  uv run python scripts/bench_mbt_factor.py
  uv run python scripts/bench_mbt_factor.py --max-factors 5
  uv run python scripts/bench_mbt_factor.py --skip-causality
  uv run python scripts/bench_mbt_factor.py --run-cli
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dotenv import find_dotenv, load_dotenv
import numpy as np

from mbt.factor.expr import ExpressionFactor
from mbt.factor.factor import check_factor_causality, normalize_factor_output
from mbt.factor.metrics import (
  compute_annual_breakdown,
  compute_factor_correlation_matrix,
  compute_ic_series,
  compute_quantile_analysis,
  compute_turnover_and_autocorr,
  summarize_ic,
)
from mbt.factor.preprocess import compute_log_mktcap, preprocess_factor
from mbt.factor.returns import compute_forward_returns
from mbt.run.msd import load_data
from mbt.select import A_STOCKS_EXCLUDE_ST

DEFAULT_EXPRESSIONS: dict[str, str] = {
  "bp": "f003 / raw_close",
  "ep_ded": "f007 / raw_close",
  "roe_ann": "f205",
  "roic": "f207",
  "gross_margin": "f213",
  "ocf_np": "f234",
  "low_lev": "-1 * f223",
  "div_yield": "ts_sum(dividend, 250) / raw_close",
  "low_vol": "-1 * ts_std_dev(delta(close, 1) / ts_delay(close, 1), 60)",
  "mom60": "(close - ts_delay(close, 60)) / ts_delay(close, 60)",
  "mom120": "(close - ts_delay(close, 120)) / ts_delay(close, 120)",
  "low_turn": "-1 * ts_mean(volume / total_shares, 20)",
  "rev20": "-1 * (close - ts_delay(close, 20)) / ts_delay(close, 20)",
  "np_growth": "f197",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Benchmark mbt.factor evaluation stages.")
  parser.add_argument("-b", "--begin", default="2005-01-04", help="Start date (default: 2005-01-04)")
  parser.add_argument("-e", "--end", default="2021-12-31", help="End date (default: 2021-12-31)")
  parser.add_argument("--periods", default="20,60", help="Forward return periods (default: 20,60)")
  parser.add_argument("--primary-period", type=int, default=60, help="Primary period (default: 60)")
  parser.add_argument("--quantiles", type=int, default=5, help="Quantile buckets (default: 5)")
  parser.add_argument("--winsorize", default="mad", help="Winsorize method (default: mad)")
  parser.add_argument(
    "--max-factors",
    type=int,
    default=len(DEFAULT_EXPRESSIONS),
    help="Number of factors to benchmark (1..14, default: 14)",
  )
  parser.add_argument(
    "--skip-causality",
    action="store_true",
    help="Skip check_factor_causality probe",
  )
  parser.add_argument(
    "--run-cli",
    action="store_true",
    help="Also run `python -m mbt.factor` CLI subprocess end-to-end",
  )
  return parser.parse_args()


def build_cli_cmd(args: argparse.Namespace, expr_map: dict[str, str]) -> list[str]:
  cmd = [sys.executable, "-m", "mbt.factor"]
  for name, expr in expr_map.items():
    cmd.extend(["-E", f"{name}={expr}"])
  cmd.extend([
    "-s",
    "A_STOCKS_EXCLUDE_ST",
    "-b",
    args.begin,
    "-e",
    args.end,
    "--periods",
    args.periods,
    "--primary-period",
    str(args.primary_period),
    "--quantiles",
    str(args.quantiles),
    "--winsorize",
    args.winsorize,
    "-S",
  ])
  if args.skip_causality:
    cmd.append("--no-causality-check")
  return cmd


def main():
  load_dotenv(find_dotenv(), override=True)
  args = parse_args()

  items = list(DEFAULT_EXPRESSIONS.items())[: max(1, min(args.max_factors, len(DEFAULT_EXPRESSIONS)))]
  expr_map = dict(items)
  periods = tuple(sorted({int(p.strip()) for p in args.periods.split(",") if p.strip()}))
  prim_p = args.primary_period if args.primary_period in periods else periods[-1]

  print("=== mbt.factor Benchmark Configuration ===")
  print(f"Range: {args.begin} ~ {args.end} | Factors: {len(expr_map)} | Periods: {periods} | Winsorize: {args.winsorize}")
  cli_cmd = build_cli_cmd(args, expr_map)
  print("CLI equivalent:")
  print("  " + " ".join(f"'{c}'" if " " in c or "*" in c or "/" in c else c for c in cli_cmd))
  print()

  if args.run_cli:
    t_cli0 = time.perf_counter()
    proc = subprocess.run(cli_cmd, check=False)
    print(f"\n[CLI Subprocess] exit={proc.returncode}, wall={time.perf_counter() - t_cli0:.2f}s\n")
    return

  host = os.environ.get("MSD_HOST", "http://localhost:50510")
  bench_sym = os.environ.get("MSD_BT_BENCHMARK", "SH000300")
  factor_inst = ExpressionFactor(expr_map)

  # 1. Load data + Panel
  t0 = time.perf_counter()
  load_objs = [bench_sym] + [s for s in A_STOCKS_EXCLUDE_ST if s != bench_sym]
  dp, loaded_objs = load_data(
    host,
    load_objs,
    args.begin,
    args.end,
    fin_fields=factor_inst.fin_fields,
    fin_only_year=factor_inst.fin_only_year,
    warmup_bars=int(factor_inst.warmup_bars),
    fin_lookback_days=factor_inst.fin_lookback_days,
  )
  t_load = time.perf_counter() - t0

  t1 = time.perf_counter()
  full_panel = dp.to_panel()
  bench_close = None
  bench_open = None
  if bench_sym in full_panel.objs:
    b_idx = full_panel.pick(bench_sym)
    bench_close = np.array(full_panel.column("close")[b_idx, :], dtype=np.float64, copy=True)
    bench_open = np.array(full_panel.column("open")[b_idx, :], dtype=np.float64, copy=True)
    stock_objs = [s for s in loaded_objs if s != bench_sym]
    stock_panel = full_panel.slice(0, full_panel.bars, objs=stock_objs)
  else:
    stock_panel = full_panel
  t_panel = time.perf_counter() - t1

  print(f"Loaded Panel: {stock_panel.n_objs} stocks x {stock_panel.bars} bars (warmup={getattr(dp, 'warmup_bars', 0)})")

  factor_inst.bind_context_metadata(
    industry_codes=None,
    benchmark_close=bench_close,
    benchmark_open=bench_open,
  )

  # 2. Compute factor expressions
  t2 = time.perf_counter()
  raw_out = factor_inst.compute(stock_panel)
  full_outputs = normalize_factor_output(raw_out, stock_panel.n_objs, stock_panel.bars, "expr")
  t_compute = time.perf_counter() - t2
  print(f"[Stage 3] ExpressionFactor.compute: {t_compute:.2f}s", flush=True)

  # 3. Causality check
  w = int(getattr(dp, "warmup_bars", 0))
  t3 = time.perf_counter()
  if not args.skip_causality:
    check_factor_causality(factor_inst, stock_panel, full_outputs, warmup_bars=w, fail_fast=False)
  t_causality = time.perf_counter() - t3
  print(f"[Stage 4] check_factor_causality: {t_causality:.2f}s", flush=True)

  # 4. Forward returns & log_mktcap
  eval_panel = stock_panel.slice(w, stock_panel.bars) if w > 0 else stock_panel
  eval_outputs = (
    {k: np.ascontiguousarray(v[:, w:]) for k, v in full_outputs.items()} if w > 0 else full_outputs
  )
  dates = eval_panel.timeline()

  t4 = time.perf_counter()
  fwd_set = compute_forward_returns(eval_panel, periods=periods, entry="next_open", exclude_limit=True)
  log_mktcap = compute_log_mktcap(eval_panel)
  t_fwd = time.perf_counter() - t4
  print(f"[Stage 5] forward_returns + log_mktcap: {t_fwd:.2f}s", flush=True)

  # 5. Per-factor metric loop
  t_prep_sum = 0.0
  t_ic_mkt_sum = 0.0
  t_ic_fwd_sum = 0.0
  t_qa_sum = 0.0
  t_turn_sum = 0.0
  t_ann_sum = 0.0
  processed_map: dict[str, np.ndarray] = {}

  for idx_f, (fname, raw_mat) in enumerate(eval_outputs.items(), 1):
    t_f0 = time.perf_counter()
    direction = factor_inst.get_direction(fname)
    oriented = raw_mat * float(direction) if direction == -1 else raw_mat

    ta = time.perf_counter()
    proc_mat = preprocess_factor(
      oriented,
      signal_mask=fwd_set.signal_mask,
      winsorize_method=args.winsorize,
      neutralize_modes=(),
      standardize_method="none",
    )
    dt_prep = time.perf_counter() - ta
    t_prep_sum += dt_prep

    tb = time.perf_counter()
    compute_ic_series(proc_mat, log_mktcap, min_obs=10)
    dt_ic_m = time.perf_counter() - tb
    t_ic_mkt_sum += dt_ic_m

    ic_series_map: dict[int, np.ndarray] = {}
    ric_series_map: dict[int, np.ndarray] = {}
    qa_map = {}

    dt_ic_f = 0.0
    dt_qa = 0.0
    for p in periods:
      tc = time.perf_counter()
      ic_s, ric_s = compute_ic_series(proc_mat, fwd_set.forward_returns[p], min_obs=10)
      summarize_ic(ic_s, ric_s, period=p)
      dt_ic_f += time.perf_counter() - tc
      ic_series_map[p] = ic_s
      ric_series_map[p] = ric_s

      td = time.perf_counter()
      qa_map[p] = compute_quantile_analysis(
        proc_mat, fwd_set, p, n_quantiles=args.quantiles, cost_bps=0.0
      )
      dt_qa += time.perf_counter() - td
    t_ic_fwd_sum += dt_ic_f
    t_qa_sum += dt_qa

    te = time.perf_counter()
    compute_turnover_and_autocorr(proc_mat, n_quantiles=args.quantiles, min_obs=10)
    dt_turn = time.perf_counter() - te
    t_turn_sum += dt_turn

    tf = time.perf_counter()
    compute_annual_breakdown(
      dates, ic_series_map[prim_p], ric_series_map[prim_p], qa_map[prim_p].long_excess_nav, period=prim_p
    )
    dt_ann = time.perf_counter() - tf
    t_ann_sum += dt_ann

    print(
      f"  [{idx_f}/{len(eval_outputs)}] {fname}: total={time.perf_counter()-t_f0:.2f}s "
      f"(prep={dt_prep:.2f}s, ic_mkt={dt_ic_m:.2f}s, ic_fwd={dt_ic_f:.2f}s, qa={dt_qa:.2f}s, turn={dt_turn:.2f}s)",
      flush=True,
    )

    processed_map[fname] = proc_mat

  # 6. Correlation matrix
  t5 = time.perf_counter()
  if len(processed_map) > 1:
    compute_factor_correlation_matrix(processed_map, min_obs=10)
  t_corr = time.perf_counter() - t5
  print(f"[Stage 7] correlation_matrix: {t_corr:.2f}s", flush=True)

  stages = [
    ("1. MSD load_data", t_load),
    ("2. Panel to_panel + slice", t_panel),
    ("3. ExpressionFactor.compute", t_compute),
    ("4. check_factor_causality", t_causality),
    ("5. compute_forward_returns + mktcap", t_fwd),
    ("6a. preprocess_factor (sum)", t_prep_sum),
    ("6b. compute_ic_series mktcap (sum)", t_ic_mkt_sum),
    ("6c. compute_ic_series fwd (sum)", t_ic_fwd_sum),
    ("6d. compute_quantile_analysis (sum)", t_qa_sum),
    ("6e. compute_turnover_and_autocorr (sum)", t_turn_sum),
    ("6f. compute_annual_breakdown (sum)", t_ann_sum),
    ("7. compute_factor_correlation_matrix", t_corr),
  ]
  total = sum(dt for _, dt in stages)
  nf = len(expr_map)

  print("\n| Stage | Time (s) | Share | Per-Factor (s) |")
  print("| :--- | ---: | ---: | ---: |")
  for name, dt in stages:
    print(f"| {name} | {dt:7.2f}s | {dt / max(total, 1e-9) * 100:5.1f}% | {dt / nf:7.3f}s |")
  print(f"| **TOTAL** | **{total:7.2f}s** | **100.0%** | **{total / nf:7.3f}s** |")


if __name__ == "__main__":
  main()
