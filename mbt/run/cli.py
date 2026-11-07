from dotenv import load_dotenv

load_dotenv(override=True)

import datetime
from mbt import (
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
  RunRequest,
)
from mbt.run.msd import load_data
from mbt.select import MsdSelectorDataProvider
from mbt.select.cli import load_selector, try_load_selector, parse_args as parse_kv_args
import mbt.shared.stocks
import logging
from typing import Any, Tuple
import argparse
import polars as pl
import orjson
import os
import numpy as np

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
    help="Strategy to run, can be a JSON file or a Python file",
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
    "-b", "--begin", type=str, default=os.environ.get("MSD_BT_BEGIN", "2020-01-01"), help="Start date, env $MSD_BT_BEGIN"
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
  parser.add_argument("-V", "--verbose", action="store_true", help="verbose output")
  args = parser.parse_args()

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

  if args.msd and not os.environ.get("MSD_HOST"):
    os.environ["MSD_HOST"] = args.msd

  kv_kwargs = parse_kv_args(args.args) if args.args else None

  if args.strategy.endswith(".json"):
    with open(args.strategy, "r", encoding='utf-8') as f:
      req = RunRequest.from_json(
        f,
        objs=args.objs if len(args.objs) > 0 else None,
        start=args.begin,
        end=args.end,
        args=kv_kwargs if kv_kwargs else args.args,
      )
  elif args.strategy.endswith(".py"):
    with open(args.strategy, "r", encoding='utf-8') as f:
      req = RunRequest(
        f,
        args.objs,
        start=args.begin,
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
    raise ValueError("No objs provided")

  try:
    selected_objs.remove(args.benchmark)
  except ValueError:
    pass

  selected_objs.insert(0, args.benchmark)
  req.objs = selected_objs
  req.portfolio = bool(args.portfolio or getattr(req.strategy, "portfolio", False))  # type: ignore[attr-defined]
  req.cash = args.cash  # type: ignore[attr-defined]
  req.summary = bool(args.summary)  # type: ignore[attr-defined]

  fee = Fee.load(args.fee)

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


def main():
  req, msd_host, output, fee = parse_args()
  logger.info(f"load data from {msd_host}")
  dp, objs = load_data(msd_host, req.objs, req.start, req.end)
  logger.info(
    f"data loaded, objs: {len(objs)}, bars per obj: {dp.bars}, total_bars: {dp.groups * dp.bars}"
  )
  use_portfolio = getattr(req, "portfolio", False) or getattr(
    req.strategy, "portfolio", False
  )
  custom_cash = getattr(req, "cash", None)

  if use_portfolio:
    init_cash = custom_cash if custom_cash is not None else 1000000.0
    p_account = PortfolioAccount(init_cash, dp, fee=fee, tradable_start=1)
    logger.info(f"portfolio account created, initial_cash={init_cash}")
    p_runner = PortfolioRunner(p_account)
    logger.info("portfolio runner created")
    p_runner.run(req.strategy)
    logger.info("portfolio strategy finished")
    reports = build_portfolio_report(dp, p_account)
  else:
    init_cash = custom_cash if custom_cash is not None else 10000.0
    account = Account(init_cash, dp, fee=fee)
    logger.info("account created")
    runner = Runner(account)
    logger.info("runner created")
    runner.run(req.strategy)
    logger.info("strategy finished")
    reports = build_report(dp, runner.ctx.accounts)[1:]

  logger.info("report built")
  show_summary = getattr(req, "summary", False)
  if len(output) > 0:
    build_output(reports, output)
    if show_summary:
      print(build_markdown_summary(reports), end="")
  else:
    if show_summary:
      print(build_markdown_summary(reports), end="")
    else:
      echo_reports(reports)


if __name__ == "__main__":
  main()
