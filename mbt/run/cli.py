# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from __future__ import annotations

"""Command line interface for executing backtests.

Supports single-stock independent execution and multi-stock shared-cash portfolio execution.
"""

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

import datetime
from mbt import (
  DataProvider,
  Runner,
  Account,
  Fee,
  PortfolioAccount,
  PortfolioRunner,
  Selector,
  StaticSelector,
)
from mbt.run import (
  Report,
  build_report,
  build_portfolio_report,
  build_json_report,
  build_markdown_summary,
  build_multi_start_summary,
  RunRequest,
)
from mbt.run.msd import load_data
from mbt.select import MsdSelectorDataProvider
from mbt.select.cli import load_selector, try_load_selector, parse_args as parse_kv_args
from mbt.notify import load_notify_config, send_notifications
import mbt.shared.stocks
import logging
from typing import Any, Tuple
import argparse
import polars as pl
import orjson
import os
import sys
import inspect
import numpy as np
from mbt.run.backend import SYSTEM_ARGS

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)

logger = logging.getLogger("backtest")


def expand_objs(objs: list[str]) -> list[str]:
  """read as file when input is a file path, and expand it. otherwise, return the input as a list of objs"""
  expanded = []
  for s in objs:
    if os.path.isfile(s):
      with open(s, "r", encoding='utf-8') as f:
        lines = filter(
          lambda x: len(x) > 0 and not x.startswith("#"),
          map(lambda x: x.strip(), f.readlines()),
        )
        expanded.extend(lines)
    else:
      match s:
        case "A_STOCKS_EXCLUDE_ST":
          expanded.extend(mbt.shared.stocks.A_STOCKS_EXCLUDE_ST)
        case "A_STOCKS":
          expanded.extend(mbt.shared.stocks.A_STOCKS)
        case "FOUNDS":
          expanded.extend(mbt.shared.stocks.FOUNDS)
        case _:
          expanded.append(s)

  return expanded


def build_selector(
  objs: list[str],
  selector_kwargs: dict[str, Any] | None = None,
  strategy_path: str | None = None,
) -> Tuple[Selector, list[str]]:
  """Build a unified Selector and initial stock list from -s/--objs arguments.

  - If a .py file is passed in `objs`, load it as a `Selector` subclass, and
    expand any remaining arguments as its initial stock pool `init_objs`.
  - If `objs` is empty and `strategy_path` is a .py file that defines a
    `Selector` subclass (coexisting with `Strategy`), automatically use that `Selector`.
  - Otherwise, expand all arguments as stock codes and wrap them in a
    `StaticSelector` that always produces that stock list.
  """
  selector: Selector | None = None
  raw_stocks: list[str] = []

  for s in objs:
    if s.endswith(".py"):
      selector = load_selector(s, **(selector_kwargs or {}))
    else:
      raw_stocks.append(s)

  init_objs = expand_objs(raw_stocks)
  if selector is None and len(init_objs) == 0 and strategy_path and strategy_path.endswith(".py"):
    selector = try_load_selector(strategy_path, **(selector_kwargs or {}))

  if selector is None:
    selector = StaticSelector(init_objs)

  return selector, init_objs


def parse_args() -> Tuple[RunRequest, str, str, Fee]:
  parser = argparse.ArgumentParser(description="Run a strategy")
  parser.add_argument(
    "strategy",
    nargs="?",
    default=None,
    help="Strategy to run, can be a JSON file or a Python file",
  )
  parser.add_argument(
    "--doctor",
    action="store_true",
    help="run environment, MSD database and data coverage diagnostics",
  )
  parser.add_argument(
    "-s",
    "--objs",
    nargs="+",
    action="extend",
    default=[],
    help="objs or selector (.py) to run",
  )
  parser.add_argument("-B", "--benchmark", type=str, default=os.environ.get("MSD_BT_BENCHMARK", "SH000300"), help="Benchmark obj, env $MSD_BT_BENCHMARK")
  parser.add_argument(
    "-b",
    "--begin",
    action="append",
    default=None,
    help="Start date(s), can be specified multiple times or comma-separated (e.g. -b 2023-01-01 -b 2024-01-01 or -b 2023-01-01,2024-01-01), env $MSD_BT_BEGIN",
  )
  parser.add_argument("-e", "--end", type=str, default=os.environ.get("MSD_BT_END", "2020-12-31"), help="End date, env $MSD_BT_END")
  parser.add_argument(
    "-a",
    "--args",
    nargs="*",
    action="extend",
    default=[],
    help="Strategy/Selector arguments in key=value format",
  )
  parser.add_argument(
    "-m", "--msd", type=str, default=os.environ.get("MSD_HOST", "http://localhost:50510"), help="MSD host, env $MSD_HOST"
  )
  parser.add_argument(
    "-o",
    "--output",
    default="",
    help="output to, empty to stdout, or any supported file path, csv, json, excel",
  )
  parser.add_argument(
    "-f",
    "--fee",
    default=os.environ.get("MSD_BT_FEE", ""),
    help="fee config path, env $MSD_BT_FEE (optional, defaults to etc/fee.json)",
  )
  parser.add_argument(
    "-p",
    "--portfolio",
    action="store_true",
    help="run in shared-cash portfolio rebalancing mode (auto-enabled for PortfolioStrategy)",
  )
  parser.add_argument(
    "-c",
    "--cash",
    type=float,
    default=None,
    help="initial cash (default: 10000 for single-stock mode, 1000000 for portfolio mode)",
  )
  parser.add_argument(
    "-S",
    "--summary",
    action="store_true",
    help="print markdown statistical summary report to stdout",
  )
  parser.add_argument(
    "--fin-only-year",
    action="store_true",
    default=False,
    help="only query annual financial reports (ts.month=12) from stock_financial",
  )
  parser.add_argument(
    "--fin-fields",
    type=str,
    default=None,
    help="comma-separated financial fields (e.g. 'f001,f205') to preload for backtest",
  )
  parser.add_argument(
    "--fin-lookback-days",
    type=int,
    default=None,
    help="lookback window in calendar days for financial data (default 1200d for annual, 500d for quarterly)",
  )
  parser.add_argument(
    "--warmup-bars",
    type=int,
    default=0,
    help="warmup bars loaded before start date for indicators only (no trading)",
  )
  parser.add_argument(
    "--exec-mode",
    type=str,
    choices=["close", "next_open"],
    default=None,
    help="execution mode: 'close' (same-bar close) or 'next_open' (next-bar open)",
  )
  parser.add_argument(
    "--starts",
    type=str,
    default=None,
    help="comma-separated list of start dates for multi-start evaluation (deprecated, use -b instead)",
  )
  parser.add_argument(
    "--holdout",
    type=str,
    default=None,
    help="holdout date or duration (e.g. 2025-07-01, 180d, 1y) to evaluate In-Sample vs Out-of-Sample",
  )
  parser.add_argument(
    "--notify",
    type=str,
    default=None,
    help="path to JSON notification config (e.g. etc/notify.json) to push daily portfolio signals",
  )
  parser.add_argument(
    "--notify-on-trade",
    action="store_true",
    help="only send webhook notifications when the final bar has executed trades or pending signals",
  )
  parser.add_argument(
    "--notify-detail",
    type=str,
    choices=["compact", "minimal"],
    default=None,
    help="override notification message detail level ('compact' or 'minimal')",
  )
  parser.add_argument("-V", "--verbose", action="store_true", help="verbose output")
  args = parser.parse_args()

  if args.doctor:
    from mbt.doctor import run_doctor
    sys.exit(run_doctor(args.msd))

  if not args.strategy:
    parser.error("the following arguments are required: strategy")

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  if args.msd and not os.environ.get("MSD_HOST"):
    os.environ["MSD_HOST"] = args.msd

  raw_starts: list[str] = []
  if args.begin:
    for item in args.begin:
      raw_starts.extend([s.strip() for s in item.split(",") if s.strip()])
  if args.starts:
    raw_starts.extend([s.strip() for s in args.starts.split(",") if s.strip()])

  if not raw_starts:
    if os.environ.get("MSD_BT_BEGIN"):
      raw_starts.extend([s.strip() for s in os.environ["MSD_BT_BEGIN"].split(",") if s.strip()])
    else:
      raw_starts = ["2020-01-01"]

  seen_starts = set()
  deduped_starts = []
  for s in raw_starts:
    if s not in seen_starts:
      seen_starts.add(s)
      deduped_starts.append(s)
  raw_starts = deduped_starts

  primary_start = raw_starts[0]
  kv_kwargs = parse_kv_args(args.args) if args.args else None

  if args.strategy.endswith(".json"):
    with open(args.strategy, "r", encoding='utf-8') as f:
      req = RunRequest.from_json(
        f,
        objs=args.objs if len(args.objs) > 0 else None,
        start=primary_start,
        end=args.end,
        args=kv_kwargs if kv_kwargs else args.args,
      )
  elif args.strategy.endswith(".py"):
    with open(args.strategy, "r", encoding='utf-8') as f:
      req = RunRequest(
        f,
        args.objs,
        start=primary_start,
        end=args.end,
        args=kv_kwargs if kv_kwargs else args.args,
      )
  else:
    raise ValueError("Invalid strategy type")

  raw_objs = args.objs if len(args.objs) > 0 else (req.objs or [])
  selector, init_objs = build_selector(
    raw_objs,
    selector_kwargs=kv_kwargs,
    strategy_path=args.strategy,
  )

  selector_dp = MsdSelectorDataProvider(args.msd, date=req.start)
  selected_objs = list(selector.execute(selector_dp, init_objs))

  if len(selected_objs) == 0:
    if isinstance(selector, StaticSelector):
      raise ValueError(
        f"No target stocks provided for strategy '{args.strategy}'. "
        "Please specify stocks via -s/--objs (e.g. -s SH600000 or -s A_STOCKS), "
        "or define a Selector subclass in the strategy file."
      )
    else:
      raise ValueError(
        f"Selector '{selector.__class__.__name__}' returned 0 stocks from "
        f"{len(init_objs)} candidate stocks on date '{req.start}'. "
        "Please check if the start date is within the database's available date range, "
        "or if the selector filter conditions are too restrictive."
      )

  try:
    selected_objs.remove(args.benchmark)
  except ValueError:
    pass

  selected_objs.insert(0, args.benchmark)
  req.objs = selected_objs
  req.portfolio = bool(args.portfolio or getattr(req.strategy, "portfolio", False))  # type: ignore[attr-defined]

  # Fail-Fast: validate that all custom -a key=val parameters are recognized
  if kv_kwargs:
    valid_keys = set(SYSTEM_ARGS)
    strat_cls = getattr(req.strategy, "__class__", None)
    if strat_cls and strat_cls.__init__ is not object.__init__:
      valid_keys.update(
        p for p in inspect.signature(strat_cls.__init__).parameters if p != "self"
      )
    sel_cls = getattr(selector, "__class__", None)
    if sel_cls and sel_cls.__init__ is not object.__init__:
      valid_keys.update(
        p for p in inspect.signature(sel_cls.__init__).parameters if p != "self"
      )

    orphan_keys = [k for k in kv_kwargs if k not in valid_keys]
    if orphan_keys:
      all_custom = sorted(valid_keys - set(SYSTEM_ARGS))
      hint = f" Accepted custom parameters: {all_custom}." if all_custom else " Neither component declares custom __init__ parameters."
      raise ValueError(
        f"Unrecognized argument(s) passed via -a/--args: {orphan_keys}. "
        f"Neither Strategy ({strat_cls.__name__ if strat_cls else 'None'}) "
        f"nor Selector ({sel_cls.__name__ if sel_cls else 'None'}) accepts these arguments.{hint}"
      )
  if req.portfolio and len(raw_objs) == 0:
    raise ValueError(
      f"Portfolio strategy '{args.strategy}' requires an explicit candidate stock pool via -s/--objs "
      "(e.g. -s A_STOCKS or -s A_STOCKS_EXCLUDE_ST). Implicitly falling back to the initial day's "
      "selected stocks is disabled to prevent silent universe truncation."
    )
  def _normalize_fields(val: Any) -> list[str]:
    if not val:
      return []
    if isinstance(val, str):
      return [f.strip() for f in val.split(",") if f.strip()]
    return [str(f) for f in val]

  cli_fin = None
  if args.fin_fields:
    cli_fin = args.fin_fields
  elif kv_kwargs and "fin_fields" in kv_kwargs:
    cli_fin = kv_kwargs["fin_fields"]

  embedded_sel = None
  if isinstance(selector, StaticSelector) and args.strategy and args.strategy.endswith(".py"):
    try:
      embedded_sel = try_load_selector(args.strategy, **(kv_kwargs or {}))
    except Exception:
      pass

  merged_fin: list[str] = []
  for src in [
    cli_fin,
    getattr(req, "fin_fields", None),
    getattr(req.strategy, "fin_fields", None),
    getattr(selector, "fin_fields", None),
    getattr(embedded_sel, "fin_fields", None),
  ]:
    for f in _normalize_fields(src):
      if f not in merged_fin:
        merged_fin.append(f)
  req.fin_fields = merged_fin if merged_fin else None

  req.cash = args.cash  # type: ignore[attr-defined]
  req.summary = bool(args.summary)  # type: ignore[attr-defined]
  req.fin_only_year = bool(
    args.fin_only_year
    or (kv_kwargs.get("fin_only_year") if kv_kwargs else False)
    or getattr(req.strategy, "fin_only_year", False)
    or getattr(selector, "fin_only_year", False)
    or getattr(embedded_sel, "fin_only_year", False)
    or getattr(req, "fin_only_year", False)
  )
  req.selector = selector
  req.warmup_bars = int(args.warmup_bars or getattr(req.strategy, "warmup_bars", 0))
  req.exec_mode = (
    args.exec_mode
    or (kv_kwargs.get("exec_mode") if kv_kwargs else None)
    or getattr(req.strategy, "exec_mode", "close")
  )
  req.fin_lookback_days = (
    args.fin_lookback_days
    or (kv_kwargs.get("fin_lookback_days") if kv_kwargs else None)
    or getattr(req.strategy, "fin_lookback_days", None)
  )
  req.starts = raw_starts if len(raw_starts) > 1 else None
  req.holdout = args.holdout
  req.raw_objs = raw_objs
  req.kv_kwargs = kv_kwargs
  req.strategy_path = args.strategy
  req.benchmark = args.benchmark
  req.notify = args.notify  # type: ignore[attr-defined]
  req.notify_on_trade = bool(args.notify_on_trade)  # type: ignore[attr-defined]
  req.notify_detail = args.notify_detail  # type: ignore[attr-defined]

  fee = Fee.load(args.fee, overrides=kv_kwargs)

  return req, args.msd, args.output, fee


def float_format(f: float | None) -> str:
  if f is None or (isinstance(f, float) and np.isnan(f)):
    return ""
  s = f"{f:.6f}"
  if "." in s:
    s = s.rstrip("0").rstrip(".")
  return s

def json_default(self, o):
  if isinstance(o, (np.ndarray, np.generic)):
    return o.tolist()
  elif isinstance(o, np.datetime64):
    return np.datetime_as_string(o, 's').rstrip("T00:00:00")
  elif isinstance(o, datetime.datetime):
    return o.strftime("%Y-%m-%dT%X").rstrip("T00:00:00")
  elif isinstance(o, float):
    return round(o, 6)
  return o


def build_output(reports: list[Report], output: str):
  ext = output.split(".")[-1].lower()
  logger.info(f"output kind: {ext}")
  if ext == "csv":
    for report in reports:
      df = pl.DataFrame(report.detailed(with_obj=True))
      float_cols = [col for col, dtype in zip(df.columns, df.dtypes) if dtype in (pl.Float32, pl.Float64)]
      df = df.with_columns([
        pl.col(col).map_elements(float_format, return_dtype=pl.String)
        for col in float_cols
      ])
      df.write_csv(output)
  elif ext == "xlsx":
    import xlsxwriter
    wb = xlsxwriter.Workbook(output, options={
      "nan_inf_to_errors": True,
    })
    df_metrics = pl.DataFrame([r.metrics(True) for r in reports])
    df_metrics.write_excel(workbook=wb, worksheet="metrics")
    for report in reports:
      data = report.detailed()
      data['price'] = report.price
      data['fw_price'] = report.fw_price
      data['bw_price'] = report.bw_price
      df = pl.DataFrame(data)
      df.write_excel(workbook=wb, worksheet=report.obj)
      if hasattr(report, "attribution") and report.attribution:
        df_attrib = pl.DataFrame(report.attribution)
        df_attrib.write_excel(workbook=wb, worksheet="attribution")
      if hasattr(report, "trades") and report.trades:
        df_trades = pl.DataFrame(report.trades)
        df_trades.write_excel(workbook=wb, worksheet="trades")
    wb.close()
  elif ext == "json":
    data = build_json_report(reports)
    logger.info("build json")
    with open(output, "wb") as f:
      f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_OMIT_MICROSECONDS))
  elif ext == "md":
    md_text = build_markdown_summary(reports)
    with open(output, "w", encoding="utf-8") as f:
      f.write(md_text)
  else:
    raise ValueError(f"unsupported output format: {ext}")
  logger.info(f"output saved to: {output}")


def echo_reports(reports: list[Report]):
  metrics_list = [r.metrics(True) for r in reports]
  print(orjson.dumps(metrics_list, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode())


def run_single_period(
  req: RunRequest,
  msd_host: str,
  fee: Fee,
  start: str,
  end: str,
  warmup_bars: int = 0,
  exec_mode: str = "close",
  fin_lookback_days: int | None = None,
) -> tuple[list[Report], DataProvider, Any]:
  """Execute a backtest run for a single time period [start, end].

  Args:
    req: RunRequest containing strategy, objs, benchmark, and config.
    msd_host: MSD server host URL.
    fee: Fee configuration.
    start: Start date of trading.
    end: End date of trading.
    warmup_bars: Number of warmup bars prior to start date for indicator initialization.
    exec_mode: Execution mode ('close' or 'next_open').
    fin_lookback_days: Lookback window in calendar days for financial data history.

  Returns:
    tuple of (reports, data_provider, account).
  """
  strat_fin = getattr(req, "fin_fields", None) or getattr(
    req.strategy, "fin_fields", None
  )
  sel_fin = getattr(getattr(req, "selector", None), "fin_fields", None)
  merged_fin: list[str] = []
  for src in [strat_fin, sel_fin]:
    if src:
      fields = [f.strip() for f in src.split(",")] if isinstance(src, str) else list(src)
      for f in fields:
        if f not in merged_fin:
          merged_fin.append(f)
  fin_fields = merged_fin if merged_fin else None

  fin_only_year = bool(
    getattr(req, "fin_only_year", False)
    or getattr(req.strategy, "fin_only_year", False)
    or getattr(getattr(req, "selector", None), "fin_only_year", False)
  )
  if fin_lookback_days is None:
    fin_lookback_days = getattr(req, "fin_lookback_days", None) or getattr(
      req.strategy, "fin_lookback_days", None
    )

  selected_objs = req.objs
  if getattr(req, "strategy_path", None) and getattr(req, "starts", None):
    selector, init_objs = build_selector(
      req.raw_objs,
      selector_kwargs=req.kv_kwargs,
      strategy_path=req.strategy_path,
    )
    selector_dp = MsdSelectorDataProvider(msd_host, date=start)
    cur_objs = list(selector.execute(selector_dp, init_objs))
    if len(cur_objs) == 0:
      raise ValueError(
        f"Selector '{selector.__class__.__name__}' returned 0 stocks for start='{start}'."
      )
    try:
      cur_objs.remove(req.benchmark)
    except ValueError:
      pass
    cur_objs.insert(0, req.benchmark)
    selected_objs = cur_objs

  dp, objs = load_data(
    msd_host,
    selected_objs,
    start,
    end,
    fin_fields=fin_fields,
    fin_only_year=fin_only_year,
    warmup_bars=warmup_bars,
    fin_lookback_days=fin_lookback_days,
  )
  logger.info(
    f"data loaded, objs: {len(objs)}, bars per obj: {dp.bars}, total_bars: {dp.groups * dp.bars}"
  )
  use_portfolio = getattr(req, "portfolio", False) or getattr(
    req.strategy, "portfolio", False
  )
  custom_cash = getattr(req, "cash", None)

  bench_obj = getattr(req, "benchmark", None)
  if use_portfolio:
    init_cash = custom_cash if custom_cash is not None else 1000000.0
    p_account = PortfolioAccount(init_cash, dp, fee=fee, benchmark=bench_obj)
    logger.info(f"portfolio account created, initial_cash={init_cash}")
    p_runner = PortfolioRunner(p_account, exec_mode=exec_mode)
    logger.info(f"portfolio runner created, exec_mode={exec_mode}")
    p_runner.run(req.strategy)
    logger.info("portfolio strategy finished")
    diag = getattr(p_runner, "diagnostics", None) or getattr(req.strategy, "diagnostics", None)
    reports = build_portfolio_report(dp, p_account, diagnostics=diag)
    return reports, dp, p_account
  else:
    init_cash = custom_cash if custom_cash is not None else 10000.0
    account = Account(init_cash, dp, fee=fee)
    logger.info("account created")
    runner = Runner(account, exec_mode=exec_mode)
    logger.info(f"runner created, exec_mode={exec_mode}")
    runner.run(req.strategy)
    logger.info("strategy finished")
    diag = getattr(runner, "diagnostics", None) or getattr(req.strategy, "diagnostics", None)
    all_reports = build_report(dp, runner.ctx.accounts, diagnostics=diag)
    if bench_obj:
      reports = [r for r in all_reports if r.obj != bench_obj]
      if not reports:
        reports = all_reports
    else:
      reports = all_reports[1:] if len(all_reports) > 1 else all_reports
    return reports, dp, account


def main():
  req, msd_host, output, fee = parse_args()
  starts = getattr(req, "starts", None)
  warmup_bars = getattr(req, "warmup_bars", 0)
  exec_mode = getattr(req, "exec_mode", "close")
  fin_lookback_days = getattr(req, "fin_lookback_days", None)
  holdout = getattr(req, "holdout", None)
  show_summary = getattr(req, "summary", False)
  notify_path = getattr(req, "notify", None)
  notify_targets = load_notify_config(notify_path) if notify_path else None

  if starts and len(starts) > 1:
    logger.info(f"running multi-start backtest across {len(starts)} starts: {starts}")
    multi_runs = []
    for st in starts:
      logger.info(f"evaluating start: {st}")
      reports, dp, acc = run_single_period(
        req,
        msd_host,
        fee,
        st,
        req.end,
        warmup_bars=warmup_bars,
        exec_mode=exec_mode,
        fin_lookback_days=fin_lookback_days,
      )
      if reports:
        multi_runs.append({
          "start": st,
          "end": req.end,
          "report": reports[0],
          "objs": req.objs,
        })

    summary_md = build_multi_start_summary(multi_runs)
    if len(output) > 0:
      with open(output, "w", encoding="utf-8") as f:
        f.write(summary_md)
      logger.info(f"multi-start summary saved to {output}")
    print(summary_md, end="")
    return

  logger.info(f"load data from {msd_host}")
  reports, dp, acc = run_single_period(
    req,
    msd_host,
    fee,
    req.start,
    req.end,
    warmup_bars=warmup_bars,
    exec_mode=exec_mode,
    fin_lookback_days=fin_lookback_days,
  )
  logger.info("report built")

  summary_text = build_markdown_summary(reports, account=acc, holdout=holdout)
  if len(output) > 0:
    build_output(reports, output)
    if show_summary:
      print(summary_text, end="")
  else:
    if show_summary:
      print(summary_text, end="")
    else:
      echo_reports(reports)

  if notify_targets is not None:
    if not isinstance(acc, PortfolioAccount):
      raise ValueError("--notify is only supported in portfolio mode (-p)")
    snapshot = acc.daily_snapshot()
    strat_name = req.strategy.__class__.__name__ if req.strategy is not None else ""
    send_notifications(
      snapshot,
      notify_targets,
      strategy_name=strat_name,
      on_trade_only=(True if getattr(req, "notify_on_trade", False) else None),
      detail_override=getattr(req, "notify_detail", None),
    )


if __name__ == "__main__":
  main()
