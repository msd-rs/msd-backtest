import numpy as np
import orjson
import pytest
from mbt.shared import stocks as stocks_mod
from mbt.shared.stocks import (
  get_limit_ratio,
  ALL_STOCKS,
  get_industry_by_stocks,
  get_stock_by_industries,
)

def test_get_limit_ratio_order():
  objs = ["SH600000", "SZ300750", "SZ300477"]
  ratios = get_limit_ratio(objs)
  
  assert len(ratios) == 3
  # SH600000 is standard stock -> 0.1
  # SZ300750 is ChiNext stock -> 0.2
  # SZ300477 is ST stock -> 0.05
  np.testing.assert_array_equal(ratios, np.array([0.1, 0.2, 0.05]))

def test_get_limit_ratio_reverse_order():
  objs = ["SZ300477", "SZ300750", "SH600000"]
  ratios = get_limit_ratio(objs)
  
  np.testing.assert_array_equal(ratios, np.array([0.05, 0.2, 0.1]))

def test_get_limit_ratio_empty():
  ratios = get_limit_ratio([])
  assert len(ratios) == 0
  assert ratios.dtype == np.float64


class _FakeTable:
  def __init__(self, rows: list[tuple[str, str]]):
    self._rows = rows
    self.rows = len(rows)
    self.columns = 2 if rows else 0

  def __getitem__(self, idx: tuple[int, int]) -> str:
    r, c = idx
    return self._rows[r][c]


@pytest.fixture
def fake_stock_block_msd(monkeypatch):
  tree = {
    "申万行业": ["银行", "非银金融"],
    "概念板块": ["算力概念", "算力芯片", "半导体"],
    "中证指数": ["沪深300", "银行"],
  }
  data = {
    "__block_tree__": orjson.dumps(tree).decode(),
    "银行": "SZ000001,SH600000,SH601398",
    "沪深300": "SH600000,SH601398,SH600519",
    "算力概念": "SZ000977,SH603019",
    "SH600000": "上海市,沪深300,银行,上证50",
    "SH601398": "北京市,沪深300,银行,上证50",
    "SZ000001": "深圳市,深证成指,银行",
  }
  queries: list[str] = []

  class FakeConnection:
    def __init__(self, url: str):
      self.url = url

    def query_map(self, sql: str, handler):
      queries.append(sql)
      marker = "obj = '"
      start = sql.index(marker) + len(marker)
      end = sql.rindex("'")
      key = sql[start:end].replace("''", "'")
      if key in data:
        handler(_FakeTable([(key, data[key])]))
      else:
        handler(_FakeTable([]))

  monkeypatch.setattr(stocks_mod.pymsd, "MsdConnection", FakeConnection)
  monkeypatch.setenv("MSD_HOST", "http://localhost:50511")
  stocks_mod._BLOCK_TREE_CACHE = None
  stocks_mod._BLOCK_CATEGORIES_INDEX = None
  stocks_mod._BLOCK_TO_STOCKS_CACHE.clear()
  stocks_mod._STOCK_TO_BLOCKS_CACHE.clear()
  yield queries
  stocks_mod._BLOCK_TREE_CACHE = None
  stocks_mod._BLOCK_CATEGORIES_INDEX = None
  stocks_mod._BLOCK_TO_STOCKS_CACHE.clear()
  stocks_mod._STOCK_TO_BLOCKS_CACHE.clear()


def test_get_industry_by_stocks_single_and_category(fake_stock_block_msd):
  all_blocks = get_industry_by_stocks("SH600000")
  assert all_blocks == ["上海市", "沪深300", "银行", "上证50"]

  sw_only = get_industry_by_stocks("SH600000", category="申万行业")
  assert sw_only == ["银行"]


def test_get_industry_by_stocks_intersection_and_cache(fake_stock_block_msd):
  common = get_industry_by_stocks(["SH600000", "SH601398"])
  assert common == ["沪深300", "银行", "上证50"]

  common_sw = get_industry_by_stocks(["SH600000", "SZ000001"], category="申万行业")
  assert common_sw == ["银行"]

  query_count = len(fake_stock_block_msd)
  # Cached call should not trigger new MSD queries
  assert get_industry_by_stocks("SH600000", category="申万行业") == ["银行"]
  assert len(fake_stock_block_msd) == query_count


def test_get_stock_by_industries_single_and_intersection(fake_stock_block_msd):
  assert get_stock_by_industries("银行") == ["SZ000001", "SH600000", "SH601398"]
  assert get_stock_by_industries(["沪深300", "银行"]) == ["SH600000", "SH601398"]
  assert get_stock_by_industries(["银行", "算力概念"]) == []


def test_stock_block_fail_fast_errors(fake_stock_block_msd):
  with pytest.raises(ValueError, match="Did you mean:.*算力概念"):
    get_stock_by_industries("算力")

  with pytest.raises(ValueError, match="Unknown block category"):
    get_industry_by_stocks("SH600000", category="不存在分类")

  with pytest.raises(ValueError, match="not found in table 'stock_block'"):
    get_industry_by_stocks("600000")

  with pytest.raises(ValueError, match="non-empty"):
    get_stock_by_industries([])

  with pytest.raises(ValueError, match="non-empty"):
    get_industry_by_stocks([])
