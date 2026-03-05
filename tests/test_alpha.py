from mbt.strategy import Context
from mbt.strategy import Strategy
from mbt.runner import Runner
from mbt.account import Account
from mbt.data_provider import DataProvider
import pymsd
import alpha as al
import numpy as np

MSD_HOST = "http://localhost:50510"
SYMBOLS = [
  "SH600000",
  "SH600004",
  "SH600006",
  "SH600007",
  "SH600008",
  "SZ000001",
  "SZ000002",
  "SZ000004",
  "SZ000006",
  "SZ000007",
]


def load_data_msd(
  msd_host: str, symbols: list[str], start: str, end: str
) -> tuple[DataProvider, list[str]]:

  client = pymsd.create_msd_pandas(msd_host)

  dfs = client.load(
    objs=symbols,
    tables="stock_kline_1d",
    start=start,
    end=end,
    join="nearest",
  )

  data, symbols = client.concat(dfs, base="SZ000001", join="nan")
  data["price"] = data["close"]

  dp = DataProvider(data, symbols=symbols)

  return dp, symbols


def run_strategy(dp: DataProvider, strategy: Strategy):
  account = Account(10000.0, dp)
  runner = Runner(account)
  runner.run(strategy)


class DoubleMaStrategy(Strategy):
  def __init__(self, short_window: int, long_window: int):
    self.short_window = short_window
    self.long_window = long_window

  def execute(self, ctx: Context):
    short_ma: np.ndarray = al.MA(ctx.close, self.short_window)
    long_ma: np.ndarray = al.MA(ctx.close, self.long_window)

    ctx.buy(short_ma > long_ma, 1.0)
    ctx.sell(short_ma < long_ma, 1.0)


def main():
  dp, symbols = load_data_msd(MSD_HOST, SYMBOLS, "2020-01-01", "2020-12-31")
  strategy = DoubleMaStrategy(5, 20)
  run_strategy(dp, strategy)
  print(dp.all("assets"))
  print(dp.all("action_shares"))
  print(dp.all("actions"))


if __name__ == "__main__":
  main()
