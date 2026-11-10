import numpy as np
import pytest
from mbt.data_provider import DataProvider
from mbt.portfolio import PortfolioAccount, PortfolioContext
from mbt.strategy import Strategy


def test_portfolio_context_date_index_and_rebalance_objs():
  objs = ["SH000300", "SH600000", "SZ000001"]
  bars = 3
  ts_arr = np.array([
    np.datetime64("2024-01-02", "us"),
    np.datetime64("2024-01-03", "us"),
    np.datetime64("2024-01-04", "us"),
  ] * len(objs))

  data = {
    "ts": ts_arr,
    "close": np.full(len(objs) * bars, 10.0),
    "price": np.full(len(objs) * bars, 10.0),
  }
  dp = DataProvider(data, objs=objs)
  account = PortfolioAccount(100000.0, dp, tradable_start=1)
  ctx = PortfolioContext(account)

  # Check date
  dp.i = 0
  assert ctx.date == "2024-01-02"
  assert ctx.current_date == "2024-01-02"
  dp.i = 1
  assert ctx.date == "2024-01-03"

  # Check index_of
  assert ctx.index_of("SH000300") == 0
  assert ctx.index_of("SH600000") == 1
  assert ctx.index_of("SZ000001") == 2
  with pytest.raises(ValueError):
    ctx.index_of("NON_EXISTENT")

  # Check rebalance_objs
  ctx.rebalance_objs({"SH600000": 0.4, "SZ000001": 0.6})
  target_w = ctx.target_weights[dp.i]
  assert target_w[0] == 0.0  # benchmark is excluded
  assert np.isclose(target_w[1], 0.4)
  assert np.isclose(target_w[2], 0.6)

  # Rebalance with single obj; other tradable obj should become 0.0
  ctx.rebalance_objs({"SH600000": 0.5})
  target_w = ctx.target_weights[dp.i]
  assert np.isclose(target_w[1], 0.5)
  assert target_w[2] == 0.0
