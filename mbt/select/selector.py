from uvicorn import logging
import pandas as pd
import logging
import time

logger = logging.getLogger("selector")


class SelectorDataProvider:
  """
  Provide data to Selector
  """

  def load_kline(self, symbols: list[str], lastN: int = 100) -> dict[str, pd.DataFrame]:
    """
    load `lastN` kline bars for `symbols`

    Returns:
      A dict of [symbol] -> pl.DataFrame of lastN kline bars
      DataFrame must have columns 
        ['ts', 'open', 'high', 'low', 'close', 'amount', 'volume', 'total_shares', 'tradable_shares']
      All prices had been forward-adjusted (split, bonus, etc. already reflected).
      """
    pass

  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 12) -> dict[str, pd.DataFrame]:
    """
    load `lastN` financial data for `symbols`
    Financial data is provided on a quarterly basis, with dates of 03-30, 06-30, 09-30, and 12-31. 
    When the only_year parameter is set to true, only the annual data (12-31) is selected.

    Returns:
      A dict of [symbol] -> pl.DataFrame of lastN financial data
      DataFrame columns ['ts', *fields] 
      `ts` is the date of the financial data
    """
    pass

  def snapshot_last(self, dfs: list[dict[str, pd.DataFrame]]) -> pd.DataFrame:
    """
    given `dfs` a list of dict of symbol->DataFrame,  take the last row of each DataFrame for each symbol
    then build a DataFrame. It's index will be symbols, columns are the merge of all input DataFrames.
    """
    records = {}
    for data in dfs:
      for symbol, df in data.items():
        last = df.iloc[-1:].drop(columns=["ts"]).set_index(pd.Index([symbol], name="obj"))
        if symbol in records:
          records[symbol] = records[symbol].merge(last, left_index=True, right_index=True)
        else:
          records[symbol] = last

    return pd.concat(records.values()).sort_index()


class Selector:

  dp: SelectorDataProvider

  def __init__(self) :
    pass

  def execute(self, dp: SelectorDataProvider, init_stocks: list[str]) -> list[str]:
    self.dp = dp
    stocks = init_stocks
    for i in range(100):
      fn_name = f'step{i:02d}'
      fn = getattr(self, fn_name, None)
      if fn is None:
        break
      t1 = time.time()
      logger.info(f"start {fn_name} with {len(stocks)} stocks")
      if len(fn.__doc__.strip()) > 0:
        logger.info(fn.__doc__.strip())
      stocks = fn(stocks)
      logger.info(f"done {fn_name}, used {time.time() - t1:0.3f}s, remain {len(stocks)}")
    return stocks

    



