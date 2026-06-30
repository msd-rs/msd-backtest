from dotenv import load_dotenv

load_dotenv(override=True)

import datetime
from mbt import Runner, Account, Fee
from mbt.run import Report, build_report, build_json_report, RunRequest
from mbt.run.msd import load_data
import logging
from typing import Tuple
import argparse
import pandas as pd
import orjson
import os
import numpy as np

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)

logger = logging.getLogger("backtest")


def expand_symbols(symbols: list[str]) -> list[str]:
  """read as file when input is a file path, and expand it. otherwise, return the input as a list of symbols"""
  expanded = []
  for s in symbols:
    if os.path.isfile(s):
      with open(s, "r") as f:
        lines = filter(
          lambda x: len(x) > 0 and not x.startswith("#"),
          map(lambda x: x.strip(), f.readlines()),
        )
        expanded.extend(lines)
    else:
      expanded.append(s)

  return expanded


def parse_args() -> Tuple[RunRequest, str, str, Fee]:
  parser = argparse.ArgumentParser(description="Run a strategy")
  parser.add_argument(
    "strategy",
    help="Strategy to run, can be a JSON file or a Python file",
  )
  parser.add_argument("-s", "--symbols", nargs="+", default=[], help="Symbols to run")
  parser.add_argument("-B", "--benchmark", type=str, default=os.environ.get("MSD_BT_BENCHMARK", "SH000300"), help="Benchmark symbol, env $MSD_BT_BENCHMARK")
  parser.add_argument(
    "-b", "--begin", type=str, default=os.environ.get("MSD_BT_BEGIN", "2020-01-01"), help="Start date, env $MSD_BT_BEGIN"
  )
  parser.add_argument("-e", "--end", type=str, default=os.environ.get("MSD_BT_END", "2020-12-31"), help="End date, env $MSD_BT_END")
  parser.add_argument("-a", "--args", nargs="*", help="Strategy arguments")
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
    "--fee_json",
    default=os.environ.get("MSD_BT_FEE", ""),
    help="fee config path, env $MSD_BT_FEE",
  )
  parser.add_argument("-V", "--verbose", action="store_true", help="verbose output")
  args = parser.parse_args()

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)


  args.symbols = expand_symbols(args.symbols)

  if args.strategy.endswith(".json"):
    with open(args.strategy, "r") as f:
      req = RunRequest.from_json(
        f,
        symbols=args.symbols,
        start=args.begin,
        end=args.end,
        args=args.args,
      )
  elif args.strategy.endswith(".py"):
    with open(args.strategy, "r") as f:
      req = RunRequest(
        f,
        args.symbols,
        start=args.begin,
        end=args.end,
        args=args.args,
      )
  else:
    raise ValueError("Invalid strategy type")

  if len(args.symbols) == 0:
    raise ValueError("No symbols provided")

  try:
    args.symbols.remove(args.benchmark)
  except ValueError:
    pass

  args.symbols.insert(0, args.benchmark)

  fee = Fee(args.fee_json) if len(args.fee_json) > 0 else Fee()

  return req, args.msd, args.output, fee


def float_format(f: float) -> str:
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
      df = pd.DataFrame(report.detailed(with_symbol=True))
      df.to_csv(output, index=False, float_format=float_format)  # type: ignore
  elif ext == "xlsx":
    with pd.ExcelWriter(output) as writer:
      df_metrics = pd.DataFrame([r.metrics(True) for r in reports])
      df_metrics.to_excel(writer, sheet_name="metrics", index=False)
      for report in reports:
        data = report.detailed()
        data['price'] = report.price
        data['fw_price'] = report.fw_price
        data['bw_price'] = report.bw_price
        df = pd.DataFrame(data)
        df.to_excel(
          writer,
          sheet_name=report.symbol,
          index=False,
          float_format="%.6f",
        )
  elif ext == "json":
    data = build_json_report(reports)
    logger.info("build json")
    with open(output, "wb") as f:
      f.write(orjson.dumps(data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_OMIT_MICROSECONDS))
  else:
    raise ValueError(f"unsupported output format: {ext}")
  logger.info(f"output saved to: {output}")


def echo_reports(reports: list[Report]):
  df_metrics = pd.DataFrame([r.metrics(True) for r in reports])
  print(df_metrics.to_json(orient='records', indent=2))


def main():
  req, msd_host, output, fee = parse_args()
  logger.info(f"load data from {msd_host}")
  dp, symbols = load_data(msd_host, req.symbols, req.start, req.end)
  logger.info(
    f"data loaded, symbols: {len(symbols)}, bars per symbol: {dp.bars}, total_bars: {dp.groups * dp.bars}"
  )
  account = Account(10000.0, dp, fee=fee)
  logger.info("account created")
  runner = Runner(account)
  logger.info("runner created")
  runner.run(req.strategy)
  logger.info("strategy finished")
  reports = build_report(dp, runner.ctx.accounts)[1:]
  logger.info("report built")
  if len(output) > 0:
    build_output(reports, output)
  else:
    echo_reports(reports)


if __name__ == "__main__":
  main()
