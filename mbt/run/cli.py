from mbt import DataProvider
from mbt import Runner
from mbt import Account
from mbt.run.msd import load_data_msd
import argparse
from mbt.run.backend import RunRequest
import pandas as pd
import numpy as np


def parse_args() -> tuple[RunRequest, str]:
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
  args = parser.parse_args()

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

  return req, args.msd


def build_report(dp: DataProvider) -> pd.DataFrame:
  symbols = np.repeat(dp.symbols, dp.bars)
  dates = dp.dates
  df = pd.DataFrame(
    {
      "symbol": symbols,
      "date": dates,
      "price": dp.all("price"),
      "close": dp.all("close"),
      "ror": dp.all("ror"),
      "ror_hold": dp.all("ror_hold"),
      "actions": dp.all("actions"),
    }
  )
  return df


def main():
  req, msd_host = parse_args()
  dp, symbols = load_data_msd(msd_host, req.symbols, req.start, req.end)
  account = Account(10000.0, dp)
  runner = Runner(account)
  runner.run(req.strategy)
  df = build_report(dp)
  pd.set_option("display.max_rows", None)

  print(df)


if __name__ == "__main__":
  main()
