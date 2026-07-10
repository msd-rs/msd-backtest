from dotenv import load_dotenv
load_dotenv(override=True)

from mbt import Runner, Account, Fee
from mbt.run import Report, build_report, build_json_report, RunRequest
from mbt.run.msd import load_data
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os

# Configuration
logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)
logger = logging.getLogger("backtest-server")

DEFAULT_MSD_HOST = "http://localhost:50510"

DEFAULT_FEE = Fee()

app = FastAPI(title="MSD Backtest Server")


class RunRequestModel(BaseModel):
  strategy: str
  symbols: List[str]
  start: str
  end: str
  args: Optional[List] = None




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
    account = Account(10000.0, dp, fee=DEFAULT_FEE)
    runner = Runner(account)

    # Load the strategy using backend logic
    # RunRequest constructor expects strategy as (str | IO | Strategy)
    strategy_obj = RunRequest.load_strategy(req_model.strategy, args=req_model.args)

    logger.info("Running strategy...")
    runner.run(strategy_obj)
    logger.info("Strategy run completed")

    # Build report
    reports = build_report(dp, runner.ctx.accounts)[1:]
    report = build_json_report(reports)
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
    "--msd", default=os.getenv("MSD_HOST", DEFAULT_MSD_HOST), help="Default MSD host"
  )
  parser.add_argument(
    "--fee",
    default=os.environ.get("MSD_BT_FEE", ""),
    help="fee config path, env $MSD_BT_FEE",
  )
  args = parser.parse_args()

  if len(args.fee) > 0:
    DEFAULT_FEE = Fee(args.fee)

  DEFAULT_MSD_HOST = args.msd

  uvicorn.run(app, host=args.host, port=args.port)
