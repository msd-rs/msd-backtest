from mbt.run.report import build_report
from mbt.run.report import build_json_report
import logging
from typing import Tuple
from mbt import DataProvider
from mbt import Runner
from mbt import Account
from mbt.run.msd import load_data
import argparse
from mbt.run.backend import RunRequest
import pandas as pd
import numpy as np
import json
import logging

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)

logger = logging.getLogger("backtest")


def parse_args() -> Tuple[RunRequest, str, str]:
  parser = argparse.ArgumentParser(description="Run a strategy")
  parser.add_argument(
    "strategy",
    help="Strategy to run, can be a JSON file or a Python file",
  )
  parser.add_argument("-s", "--symbols", nargs="+", default=[], help="Symbols to run")
  parser.add_argument(
    "-b", "--begin", type=str, default="2020-01-01", help="Start date"
  )
  parser.add_argument("-e", "--end", type=str, default="2020-12-31", help="End date")
  parser.add_argument("-a", "--args", nargs="*", help="Strategy arguments")
  parser.add_argument(
    "-m", "--msd", type=str, default="http://localhost:50510", help="MSD host"
  )
  parser.add_argument(
    "-o",
    "--output",
    default="",
    help="output to, empty to stdout, or any supported file path, csv, excel",
  )
  parser.add_argument("-V", "--verbose", action="store_true", help="verbose output")
  args = parser.parse_args()

  if args.verbose:
    logging.getLogger().setLevel(logging.DEBUG)

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

  return req, args.msd, args.output


def float_format(f: float) -> str:
  s = f"{f:.6f}"
  if "." in s:
    s = s.rstrip("0").rstrip(".")
  return s


def build_output(dp: DataProvider, symbols: list[str], df: pd.DataFrame, output: str):
  ext = output.split(".")[-1].lower()
  logger.info(f"output kind: {ext}")
  if ext == "csv":
    df.to_csv(output, index=False, float_format=float_format)  # type: ignore
  elif ext == "xlsx":
    with pd.ExcelWriter(output) as writer:
      for i, symbol in enumerate(symbols):
        df.iloc[i * dp.bars : (i + 1) * dp.bars].to_excel(
          writer,
          sheet_name=symbol,
          index=False,
          float_format="%.6f",
        )
  elif ext == "json":
    data = build_json_report(dp, df)
    logger.info("build json")
    with open(output, "w") as f:
      json.dump(data, f, ensure_ascii=False)
  else:
    raise ValueError(f"unsupported output format: {ext}")
  logger.info(f"output saved to: {output}")


def main():
  req, msd_host, output = parse_args()
  logger.info(f"load data from {msd_host}")
  dp, symbols = load_data(msd_host, req.symbols, req.start, req.end)
  logger.info("data loaded")
  account = Account(10000.0, dp)
  logger.info("account created")
  runner = Runner(account)
  logger.info("runner created")
  runner.run(req.strategy)
  logger.info("strategy run")
  df = build_report(dp)
  logger.info("report built")
  if len(output) > 0:
    build_output(dp, symbols, df, output)
  else:
    print(df)


if __name__ == "__main__":
  main()
