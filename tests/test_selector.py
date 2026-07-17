import pandas as pd
import polars as pl
import numpy as np
import mbt
from mbt.select import SelectorDataProvider
from strategy.growth_large_cap_ma15_selector import GrowthLargeCapMA15Selector

class MockSelectorDataProvider(SelectorDataProvider):
  def __init__(self, kline_data, financial_data):
    super().__init__()
    self.kline_data = kline_data
    self.financial_data = financial_data

  def load_kline(self, symbols: list[str], lastN: int = 100) -> dict[str, pl.DataFrame]:
    return {sym: self.kline_data[sym] for sym in symbols if sym in self.kline_data}

  def load_financial(self, symbols: list[str], fields: list[str], only_year: bool = True, lastN: int = 12) -> dict[str, pl.DataFrame]:
    return {sym: self.financial_data[sym] for sym in symbols if sym in self.financial_data}

def test_growth_large_cap_ma15_selector():
  # Define dates
  dates_fin = pd.to_datetime(["2023-12-31", "2024-12-31", "2025-12-31"])
  dates_k = pd.date_range(end="2026-03-06", periods=20, freq="D")

  # 1. SZ000001: All filters pass
  # Net Profit: 100 -> 120 -> 150 (growing & positive)
  df_fin_1 = pl.DataFrame({"ts": dates_fin, "f137": [100.0, 120.0, 150.0]})
  # Close price drops below MA15 on the last day
  close_1 = np.ones(20) * 12.0
  close_1[-2] = 13.0
  close_1[-1] = 8.0
  df_k_1 = pl.DataFrame({
    "ts": dates_k,
    "open": close_1,
    "high": close_1,
    "low": close_1,
    "close": close_1,
    "volume": np.ones(20) * 1000,
    "amount": np.ones(20) * 12000,
    "total_shares": np.ones(20) * 1000,
    "tradable_shares": np.ones(20) * 1000,
  })

  # 2. SZ000002: Fails financial growth (150 -> 120 -> 100)
  df_fin_2 = pl.DataFrame({"ts": dates_fin, "f137": [150.0, 120.0, 100.0]})
  df_k_2 = df_k_1.clone()

  # 3. SZ000003: Fails large cap (market cap is small)
  df_fin_3 = df_fin_1.clone()
  df_k_3 = df_k_1.clone()
  df_k_3 = df_k_3.with_columns(pl.lit(10.0).alias("total_shares"))

  # 4. SZ000004: Fails MA15 breakdown (stays at 12.0, close is never below MA15)
  df_fin_4 = df_fin_1.clone()
  close_4 = np.ones(20) * 12.0
  df_k_4 = pl.DataFrame({
    "ts": dates_k,
    "open": close_4,
    "high": close_4,
    "low": close_4,
    "close": close_4,
    "volume": np.ones(20) * 1000,
    "amount": np.ones(20) * 12000,
    "total_shares": np.ones(20) * 1000,
    "tradable_shares": np.ones(20) * 1000,
  })

  financial_data = {
    "SZ000001": df_fin_1,
    "SZ000002": df_fin_2,
    "SZ000003": df_fin_3,
    "SZ000004": df_fin_4,
  }

  kline_data = {
    "SZ000001": df_k_1,
    "SZ000002": df_k_2,
    "SZ000003": df_k_3,
    "SZ000004": df_k_4,
  }

  dp = MockSelectorDataProvider(kline_data, financial_data)
  
  # Use top 50% percentile
  selector = GrowthLargeCapMA15Selector(large_cap_percentile=0.5, break_days=3)
  
  selected = selector.execute(dp, ["SZ000001", "SZ000002", "SZ000003", "SZ000004"])
  
  assert selected == ["SZ000001"]
  
  # Check factors
  factors = selector.factors
  assert len(factors) == 1
  assert factors[0, "symbol"] == "SZ000001"
  assert factors[0, "net_profit_y3"] == 150.0
  assert factors[0, "market_cap"] == 8000.0
  assert factors[0, "close"] == 8.0
