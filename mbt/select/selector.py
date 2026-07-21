from uvicorn import logging
import polars as pl
import logging
import time

logger = logging.getLogger("selector")


class SelectorDataProvider:

  date: str|None = None

  """
  Provide data to Selector
  """

  def load_kline(self, symbols: list[str], lastN: int = 100) -> dict[str, pl.DataFrame]:
    """
    load `lastN` kline bars for `symbols`

    Returns:
      A dict of [symbol] -> pl.DataFrame of lastN kline bars
      DataFrame must have columns 
        ['ts', 'open', 'high', 'low', 'close', 'amount', 'volume', 'total_shares', 'tradable_shares']
      All prices had been forward-adjusted (split, bonus, etc. already reflected).
      """
    pass

  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 12) -> dict[str, pl.DataFrame]:
    """
    load `lastN` financial data for `symbols`
    Financial data is provided on a quarterly basis, with dates of 03-30, 06-30, 09-30, and 12-31. 
    When the only_year parameter is set to true, only the annual data (12-31) is selected.

    Returns:
      A dict of [symbol] -> pl.DataFrame of lastN financial data
      DataFrame columns ['ts', *fields] 
    """
    pass

  def load_snapshot(self, symbols: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame:
    """
    Take a snapshot for the given symbols at 'snapshot_date', this is more faster than 
    load_kline and load_financial separately.

    User usually use this function to get the latest data for symbols.

    Args:
      symbols: symbols to snapshot
      fin_fields: fields to snapshot from financial data
      fin_only_year: whether to only snapshot annual financial data

    Returns:
      A DataFrame of snapshot for the given symbols, with columns combined from 
      `load_kline` and `load_financial` except the financial data's 'ts` column will be renamed to 'f_ts'
    """
    pass


class Selector:

  dp: SelectorDataProvider

  def __init__(self) :
    pass

  def execute(self, dp: SelectorDataProvider, init_stocks: list[str]) -> list[str]:
    self.dp = dp
    stocks = init_stocks
    t0 = time.time()
    logger.info(f"start execute selector, {self.__class__.__name__}")
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
    logger.info(f"done execute selector {self.__class__.__name__}, used {time.time() - t0:0.3f}s, remain {len(stocks)}")
    return stocks

    



