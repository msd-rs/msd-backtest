import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from mbt import Runner, Account, DataProvider
from mbt.run.msd import load_data
from mbt.run.backend import RunRequest

# Configuration
logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger("backtest-server")

DEFAULT_MSD_HOST = "http://localhost:50510"

app = FastAPI(title="MSD Backtest Server")


class RunRequestModel(BaseModel):
  strategy: str
  symbols: List[str]
  start: str
  end: str
  args: Optional[List] = None


def build_json_report(dp: DataProvider) -> dict:
  # Repeat logic from build_report in cli.py
  syms = np.repeat(dp.symbols, dp.bars)
  df = pd.DataFrame(
    {
      "symbol": syms,
      "date": dp.dates,
      "price": dp.all("price"),
      "close": dp.all("close"),
      "ror": dp.all("ror"),
      "ror_hold": dp.all("ror_hold"),
      "actions": dp.all("actions"),
    }
  )

  data = {}
  for i, symbol in enumerate(dp.symbols):
    d = df.iloc[i * dp.bars : (i + 1) * dp.bars].to_dict(orient="list")  # type: ignore
    del d["symbol"]
    # Format dates as strings
    d["date"] = [t.strftime("%Y-%m-%d %X").rstrip(" 00:00:00") for t in d["date"]]
    data[symbol] = d
  return data


@app.post("/backtest")
async def run_backtest(req_model: RunRequestModel):
  try:
    logger.info(f"Received backtest request for symbols: {req_model.symbols}")

    # Use RunRequest to load the strategy (can be source code or file path)
    # The RunRequest class in backend.py expects strategy, symbols, start, end, args
    # We'll pass the strategy string directly to load_strategy logic if needed,
    # but RunRequest handles it.

    # We need to load data first as in cli.py
    logger.info(f"Loading data from {DEFAULT_MSD_HOST} for {req_model.symbols}")
    dp, symbols = load_data(
      DEFAULT_MSD_HOST, req_model.symbols, req_model.start, req_model.end
    )

    if "ts" not in dp.data:
      raise ValueError(
        "No data found for the given symbols and date range. Please check your MSD server and parameters."
      )

    logger.info("Data loaded successfully")

    # Initialize Account and Runner
    account = Account(10000.0, dp)
    runner = Runner(account)

    # Load the strategy using backend logic
    # RunRequest constructor expects strategy as (str | IO | Strategy)
    strategy_obj = RunRequest.load_strategy(req_model.strategy, args=req_model.args)

    logger.info("Running strategy...")
    runner.run(strategy_obj)
    logger.info("Strategy run completed")

    # Build report
    report = build_json_report(dp)
    return report

  except Exception as e:
    logger.exception("Error during backtest execution")
    raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
  import uvicorn
  import argparse

  parser = argparse.ArgumentParser(description="MSD Backtest Server")
  parser.add_argument("--host", default="0.0.0.0", help="Listen host")
  parser.add_argument("--port", type=int, default=8000, help="Listen port")
  parser.add_argument(
    "--msd", default="http://localhost:50510", help="Default MSD host"
  )
  args = parser.parse_args()

  DEFAULT_MSD_HOST = args.msd

  uvicorn.run(app, host=args.host, port=args.port)
