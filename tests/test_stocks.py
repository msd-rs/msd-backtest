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


def test_get_limit_ratio_single_stock_returns_ndarray():
  ratios = get_limit_ratio(["SH600000"], repeat=1)
  assert isinstance(ratios, np.ndarray)
  assert ratios.shape == (1,)
  np.testing.assert_allclose(ratios, np.array([0.1]))


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
    "概念板块": ["算力概念", "算力芯片", "半导体"],
    "中证指数": ["沪深300", "银行"],
  }
  sw_records = [
    {"obj": "SZ000001", "l1": "银行", "l2": "股份制银行Ⅱ", "l3": "股份制银行Ⅲ"},
    {"obj": "SH600000", "l1": "银行", "l2": "股份制银行Ⅱ", "l3": "股份制银行Ⅲ"},
    {"obj": "SH601398", "l1": "银行", "l2": "国有大型银行Ⅱ", "l3": "国有大型银行Ⅲ"},
    {"obj": "SH600519", "l1": "食品饮料", "l2": "白酒Ⅱ", "l3": "白酒Ⅲ"},
  ]
  data = {
    "__block_tree__": orjson.dumps(tree).decode(),
    "__sw__": orjson.dumps(sw_records).decode(),
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

  def _reset_caches():
    stocks_mod._SW_DF_CACHE = None
    stocks_mod._BLOCK_TREE_CACHE = None
    stocks_mod._BLOCK_CATEGORIES_INDEX = None
    stocks_mod._BLOCK_TO_STOCKS_CACHE.clear()
    stocks_mod._STOCK_TO_BLOCKS_CACHE.clear()
    stocks_mod._CATEGORY_STOCK_MAP_CACHE.clear()

  monkeypatch.setattr(stocks_mod.pymsd, "MsdConnection", FakeConnection)
  monkeypatch.setenv("MSD_HOST", "http://localhost:50511")
  _reset_caches()
  yield queries, data
  _reset_caches()


def test_get_industry_by_stocks_single_and_category(fake_stock_block_msd):
  all_blocks = get_industry_by_stocks("SH600000")
  assert all_blocks == ["上海市", "沪深300", "银行", "上证50"]

  assert get_industry_by_stocks("SH600000", category="中证指数") == ["沪深300", "银行"]
  assert get_industry_by_stocks("SH600000", category="申万一级") == ["银行"]
  assert get_industry_by_stocks("SH600000", category="申万二级") == ["股份制银行Ⅱ"]
  assert get_industry_by_stocks("SH600000", category="申万行业") == ["股份制银行Ⅱ"]
  assert get_industry_by_stocks("SH600000", category="申万三级") == ["股份制银行Ⅲ"]


def test_get_industry_by_stocks_intersection_and_cache(fake_stock_block_msd):
  queries, _ = fake_stock_block_msd
  common = get_industry_by_stocks(["SH600000", "SH601398"])
  assert common == ["沪深300", "银行", "上证50"]

  assert get_industry_by_stocks(["SH600000", "SH601398"], category="申万一级") == ["银行"]
  assert get_industry_by_stocks(["SH600000", "SH601398"], category="申万二级") == []
  assert get_industry_by_stocks(["SH600000", "SZ000001"], category="申万行业") == ["股份制银行Ⅱ"]

  query_count = len(queries)
  # Cached call should not trigger new MSD queries
  assert get_industry_by_stocks("SH600000", category="申万二级") == ["股份制银行Ⅱ"]
  assert get_industry_by_stocks("SH600000", category="申万三级") == ["股份制银行Ⅲ"]
  assert len(queries) == query_count

  # refresh=True re-queries __sw__
  assert get_industry_by_stocks("SH600000", category="申万一级", refresh=True) == ["银行"]
  assert len(queries) == query_count + 1


def test_get_stock_by_industries_shenwan_and_general(fake_stock_block_msd):
  # Default category is "申万二级"; multiple Shenwan industries return union
  assert get_stock_by_industries("股份制银行Ⅱ") == ["SZ000001", "SH600000"]
  assert get_stock_by_industries("股份制银行Ⅱ", category="申万行业") == ["SZ000001", "SH600000"]
  assert get_stock_by_industries(["股份制银行Ⅱ", "国有大型银行Ⅱ"]) == [
    "SZ000001", "SH600000", "SH601398",
  ]
  assert get_stock_by_industries("银行", category="申万一级") == [
    "SZ000001", "SH600000", "SH601398",
  ]
  assert get_stock_by_industries(["股份制银行Ⅲ", "白酒Ⅲ"], category="申万三级") == [
    "SZ000001", "SH600000", "SH600519",
  ]

  # Non-Shenwan category and category=None use stock_block intersection
  assert get_stock_by_industries("算力概念", category="概念板块") == ["SZ000977", "SH603019"]
  assert get_stock_by_industries("银行", category=None) == ["SZ000001", "SH600000", "SH601398"]
  assert get_stock_by_industries(["沪深300", "银行"], category=None) == ["SH600000", "SH601398"]
  assert get_stock_by_industries(["银行", "算力概念"], category=None) == []


def test_stock_block_fail_fast_errors(fake_stock_block_msd):
  _, data = fake_stock_block_msd

  with pytest.raises(ValueError, match="Did you mean:.*'股份制银行Ⅱ'"):
    get_stock_by_industries("股份制银行")

  with pytest.raises(ValueError, match="Did you mean:.*算力概念"):
    get_stock_by_industries("算力", category=None)

  with pytest.raises(ValueError, match="under category '概念板块'"):
    get_stock_by_industries("沪深300", category="概念板块")

  with pytest.raises(ValueError, match="Unknown block category"):
    get_industry_by_stocks("SH600000", category="不存在分类")

  with pytest.raises(ValueError, match="Unknown block category"):
    get_stock_by_industries("银行", category="不存在分类")

  with pytest.raises(ValueError, match="not found in table 'stock_block'"):
    get_industry_by_stocks("600000")

  with pytest.raises(ValueError, match="not found in Shenwan industry data"):
    get_industry_by_stocks("600000", category="申万二级")

  with pytest.raises(ValueError, match="non-empty"):
    get_stock_by_industries([])

  with pytest.raises(ValueError, match="non-empty"):
    get_industry_by_stocks([])

  # Missing __sw__ record in MSD fails fast with RuntimeError
  del data["__sw__"]
  stocks_mod._SW_DF_CACHE = None
  with pytest.raises(RuntimeError, match="__sw__"):
    get_stock_by_industries("股份制银行Ⅱ", refresh=True)
