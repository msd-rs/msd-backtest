# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Command line interface for evaluating alpha factors (`python -m mbt.factor.cli`)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Sequence

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(), override=True)

import orjson

from mbt.factor.evaluator import evaluate_factor
from mbt.factor.expr import (
  ExpressionFactor,
  format_operator_catalog,
  load_expression_file,
  parse_inline_expressions,
)
from mbt.factor.factor import Factor, load_factor
from mbt.factor.report import export_report
from mbt.run.backend import parse_kv_args
from mbt.run.cli import expand_objs

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger("factor")


def _parse_periods(raw: str) -> tuple[int, ...]:
  parts = [s.strip() for s in raw.split(",") if s.strip()]
  if not parts:
    raise ValueError(f"Invalid --periods '{raw}': expected comma-separated positive integers.")
  out = tuple(sorted({int(p) for p in parts}))
  if any(p <= 0 for p in out):
    raise ValueError(f"Invalid --periods '{raw}': all periods must be > 0.")
  return out


def _parse_neutralize(raw: str) -> tuple[str, ...]:
  if not raw or raw.strip().lower() == "none":
    return ()
  return tuple(
    dict.fromkeys(
      m.strip().lower()
      for m in raw.split(",")
      if m.strip() and m.strip().lower() != "none"
    )
  )


def build_parser() -> argparse.ArgumentParser:
  """Construct the argument parser for ``mbt.factor.cli``."""
  parser = argparse.ArgumentParser(
    description="Evaluate alpha factors (Python Factor class or alpha.lang expressions)"
  )
  parser.add_argument(
    "factor",
    nargs="?",
    default=None,
    help="Factor Python file (.py) or batch expression file (.txt / .json)",
  )
  parser.add_argument(
    "-E",
    "--expr",
    action="append",
    default=[],
    help="Inline alpha.lang expression (repeatable; supports 'name=expr' or bare 'expr')",
  )
  parser.add_argument(
    "--list-ops",
    action="store_true",
    help="Print canonical expression operators, WQ101/GTJA191 aliases, and fields, then exit",
  )
  parser.add_argument(
    "--doctor",
    action="store_true",
    help="Run environment, MSD database, and data coverage diagnostics",
  )
  parser.add_argument(
    "-s",
    "--objs",
    nargs="+",
    action="extend",
    default=[],
    help="Candidate stock universe (codes, file path, A_STOCKS, A_STOCKS_EXCLUDE_ST, FOUNDS)",
  )
  parser.add_argument(
    "-B",
    "--benchmark",
    type=str,
    default=os.environ.get("MSD_BT_BENCHMARK", "SH000300"),
    help="Benchmark index obj (used as calendar baseline & excess benchmark), env $MSD_BT_BENCHMARK",
  )
  parser.add_argument(
    "-b",
    "--begin",
    type=str,
    default=os.environ.get("MSD_BT_BEGIN", "2022-01-01"),
    help="Evaluation start date (YYYY-MM-DD), env $MSD_BT_BEGIN",
  )
  parser.add_argument(
    "-e",
    "--end",
    type=str,
    default=os.environ.get("MSD_BT_END", "2025-12-31"),
    help="Evaluation end date (YYYY-MM-DD), env $MSD_BT_END",
  )
  parser.add_argument(
    "-a",
    "--args",
    nargs="*",
    action="extend",
    default=[],
    help="Factor __init__ arguments in key=value format (Fail-Fast on unrecognized keys)",
  )
  parser.add_argument(
    "-m",
    "--msd",
    type=str,
    default=os.environ.get("MSD_HOST", "http://localhost:50510"),
    help="MSD host URL, env $MSD_HOST",
  )
  parser.add_argument(
    "--periods",
    type=str,
    default="1,5,10,20",
    help="Comma-separated forward return holding periods in bars (default: 1,5,10,20)",
  )
  parser.add_argument(
    "--primary-period",
    type=int,
    default=None,
    help="Primary holding period for quantile/annual/holdout tables (default: 5 if present)",
  )
  parser.add_argument(
    "--quantiles",
    type=int,
    default=5,
    help="Number of cross-sectional quantile buckets (default: 5)",
  )
  parser.add_argument(
    "--entry",
    type=str,
    choices=["next_open", "open_close", "close"],
    default="next_open",
    help="Forward return alignment: 'next_open' (default), 'open_close' (al.FRET), or 'close'",
  )
  parser.add_argument(
    "--include-limit",
    action="store_true",
    default=False,
    help="Include limit-up bars (limited==1) in entry mask (default excludes limit-up)",
  )
  parser.add_argument(
    "--min-listed-days",
    type=int,
    default=0,
    help="Exclude stocks with fewer than N prior valid trading bars in the panel",
  )
  parser.add_argument(
    "--winsorize",
    type=str,
    choices=["none", "mad", "sigma", "percentile"],
    default="none",
    help="Cross-sectional outlier winsorization method (default: none)",
  )
  parser.add_argument(
    "--standardize",
    type=str,
    choices=["none", "zscore", "rank"],
    default="none",
    help="Cross-sectional standardization method (default: none)",
  )
  parser.add_argument(
    "--neutralize",
    type=str,
    default="none",
    help="Comma-separated neutralization factors: 'none', 'industry', 'mktcap', or 'industry,mktcap'",
  )
  parser.add_argument(
    "--cost-bps",
    type=float,
    default=0.0,
    help="One-way transaction cost in bps deducted from tranche portfolio returns",
  )
  parser.add_argument(
    "--direction",
    type=int,
    choices=[1, -1],
    default=None,
    help="Override factor direction (1: higher is better, -1: lower is better)",
  )
  parser.add_argument(
    "--fin-fields",
    type=str,
    default=None,
    help="Comma-separated financial fields (e.g. 'f003,f205') to preload",
  )
  parser.add_argument(
    "--fin-only-year",
    action="store_true",
    default=False,
    help="Only query annual (12-31) financial reports",
  )
  parser.add_argument(
    "--fin-lookback-days",
    type=int,
    default=None,
    help="Calendar-day lookback window for financial queries",
  )
  parser.add_argument(
    "--warmup-bars",
    type=int,
    default=None,
    help="Warmup bars loaded prior to --begin for rolling indicator initialization",
  )
  parser.add_argument(
    "--holdout",
    type=str,
    default=None,
    help="Holdout split date or duration (e.g. '2024-01-01', '1y', '180d') for IS vs OOS check",
  )
  parser.add_argument(
    "--by-industry",
    action="store_true",
    default=False,
    help="Include per-industry RankIC breakdown",
  )
  parser.add_argument(
    "--industry-category",
    type=str,
    default="申万行业",
    help="Block category in MSD stock_block used for industry neutralization/breakdown",
  )
  parser.add_argument(
    "--min-obs",
    type=int,
    default=10,
    help="Minimum valid cross-sectional stock count per bar for IC computation (default: 10)",
  )
  parser.add_argument(
    "--no-causality-check",
    action="store_true",
    default=False,
    help="Disable truncation causality probe (not recommended; marked SKIPPED in report)",
  )
  parser.add_argument(
    "-S",
    "--summary",
    action="store_true",
    help="Print Markdown factor evaluation report to stdout",
  )
  parser.add_argument(
    "-o",
    "--output",
    default="",
    help="Output file path (.json, .md, .csv, .xlsx, .png)",
  )
  parser.add_argument(
    "--overview",
    nargs="?",
    const="factor_overview.png",
    default=None,
    help="Generate high-DPI factor tear-sheet PNG chart (default: factor_overview.png)",
  )
  parser.add_argument("-V", "--verbose", action="store_true", help="Verbose debug logging")
  return parser


def _resolve_factor(args: argparse.Namespace, kv_kwargs: dict[str, Any]) -> Factor:
  if args.factor and args.expr:
    raise ValueError(
      "Cannot specify both a positional factor file and -E/--expr simultaneously. Use one or the other."
    )

  if args.expr:
    if kv_kwargs:
      raise ValueError(
        f"Unrecognized -a/--args {sorted(kv_kwargs.keys())} passed with inline -E/--expr. "
        "Inline expressions do not accept __init__ parameters."
      )
    expr_map = parse_inline_expressions(args.expr)
    direction: int | dict[str, int] = args.direction if args.direction is not None else 1
    if len(expr_map) == 1:
      fname, body = next(iter(expr_map.items()))
      return ExpressionFactor(body, name=fname, direction=direction)
    return ExpressionFactor(expr_map, direction=direction)

  if not args.factor:
    raise ValueError(
      "No factor provided. Pass a Factor .py file, a batch expression .txt/.json file, or -E/--expr."
    )

  ext = os.path.splitext(args.factor)[1].lower()
  if ext == ".py":
    inst = load_factor(args.factor, kv_kwargs, strict_args=True)
    if args.direction is not None:
      inst.direction = int(args.direction)
    return inst

  if ext in (".txt", ".json"):
    if kv_kwargs:
      raise ValueError(
        f"Unrecognized -a/--args {sorted(kv_kwargs.keys())} passed with expression file '{args.factor}'."
      )
    expr_map, dirs = load_expression_file(args.factor)
    if args.direction is not None:
      dirs = {k: int(args.direction) for k in expr_map}
    return ExpressionFactor(expr_map, direction=dirs)

  raise ValueError(
    f"Unsupported factor input '{args.factor}'. Expected .py, .txt, or .json file, or pass -E/--expr."
  )


def main(argv: Sequence[str] | None = None) -> int:
  parser = build_parser()
  args = parser.parse_args(argv)

  if args.list_ops:
    print(format_operator_catalog())
    return 0

  if args.doctor:
    from mbt.doctor import run_doctor

    return int(run_doctor(args.msd))

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  if args.msd and not os.environ.get("MSD_HOST"):
    os.environ["MSD_HOST"] = args.msd

  kv_kwargs = parse_kv_args(args.args) if args.args else {}
  factor_inst = _resolve_factor(args, kv_kwargs)

  objs = expand_objs(args.objs)
  if not objs:
    raise ValueError(
      "No candidate stocks provided via -s/--objs. "
      "Specify a stock pool such as `-s A_STOCKS_EXCLUDE_ST` or a stock list file."
    )

  periods = _parse_periods(args.periods)
  neutralize = _parse_neutralize(args.neutralize)
  cli_fin = (
    [f.strip() for f in args.fin_fields.split(",") if f.strip()]
    if args.fin_fields
    else None
  )

  report = evaluate_factor(
    factor_inst,
    objs,
    args.begin,
    args.end,
    msd_host=args.msd,
    benchmark=args.benchmark,
    fin_fields=cli_fin,
    fin_only_year=bool(args.fin_only_year),
    fin_lookback_days=args.fin_lookback_days,
    warmup_bars=args.warmup_bars,
    periods=periods,
    primary_period=args.primary_period,
    quantiles=args.quantiles,
    entry=args.entry,
    exclude_limit=not bool(args.include_limit),
    min_listed_days=args.min_listed_days,
    winsorize=args.winsorize,
    standardize=args.standardize,
    neutralize=neutralize,
    cost_bps=args.cost_bps,
    holdout=args.holdout,
    causality_check=not bool(args.no_causality_check),
    min_obs=args.min_obs,
    by_industry=bool(args.by_industry),
    industry_category=args.industry_category,
  )

  if args.output:
    export_report(report, args.output)

  if args.overview:
    from mbt.factor.overview import render_factor_overview

    render_factor_overview(report, args.overview)
    logger.info(f"factor overview chart saved to: {args.overview}")

  if args.summary:
    print(report.to_markdown())
  elif not args.output and not args.overview:
    payload = report.to_dict(include_series=False)
    print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode())

  return 0


if __name__ == "__main__":
  sys.exit(main())
