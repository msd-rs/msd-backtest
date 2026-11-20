# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for batch stock-to-industry mapping in mbt.shared.stocks."""

from __future__ import annotations

import orjson
import pytest

import mbt.shared.stocks as stocks_mod


class _FakeTable:
  def __init__(self, rows: list[tuple[str, str]]):
    self._rows = rows
    self.rows = len(rows)

  def __getitem__(self, idx: tuple[int, int]) -> str:
    r, c = idx
    return self._rows[r][c]


def test_get_stock_industry_map_shenwan_levels_and_cache(monkeypatch):
  monkeypatch.setattr(stocks_mod, "_SW_DF_CACHE", None)
  monkeypatch.setattr(stocks_mod, "_CATEGORY_STOCK_MAP_CACHE", {})

  sw_records = [
    {"obj": "SH600000", "l1": "银行", "l2": "股份制银行Ⅱ", "l3": "股份制银行Ⅲ"},
    {"obj": "SZ000001", "l1": "银行", "l2": "股份制银行Ⅱ", "l3": "股份制银行Ⅲ"},
    {"obj": "SH601398", "l1": "银行", "l2": "国有大型银行Ⅱ", "l3": "国有大型银行Ⅲ"},
    {"obj": "SH600519", "l1": "食品饮料", "l2": "白酒Ⅱ", "l3": "白酒Ⅲ"},
    {"obj": "SZ002415", "l1": "电子", "l2": "光学光电子", "l3": "安防设备"},
  ]
  sw_queries: list[str] = []

  class FakeConnection:
    def __init__(self, url: str):
      self.url = url

    def query_map(self, sql: str, handler):
      sw_queries.append(sql)
      handler(_FakeTable([("__sw__", orjson.dumps(sw_records).decode())]))

  monkeypatch.setattr(stocks_mod.pymsd, "MsdConnection", FakeConnection)
  monkeypatch.setenv("MSD_HOST", "http://localhost:50511")

  # Default category is "申万二级" and omits absent stocks like SH999999
  res_default = stocks_mod.get_stock_industry_map(["SH600000", "SH600519", "SH999999"])
  assert res_default == {"SH600000": "股份制银行Ⅱ", "SH600519": "白酒Ⅱ"}
  assert len(sw_queries) == 1

  # "申万行业" is equivalent to "申万二级" and reuses the cached map
  res_alias = stocks_mod.get_stock_industry_map(None, category="申万行业")
  assert len(res_alias) == 5
  assert res_alias["SH601398"] == "国有大型银行Ⅱ"
  assert len(sw_queries) == 1

  # Other Shenwan levels reuse the cached _SW_DF_CACHE without extra MSD queries
  assert stocks_mod.get_stock_industry_map(["SH601398"], category="申万一级") == {"SH601398": "银行"}
  assert stocks_mod.get_stock_industry_map(["SH601398"], category="申万三级") == {"SH601398": "国有大型银行Ⅲ"}
  assert len(sw_queries) == 1

  # refresh=True reloads __sw__
  stocks_mod.get_stock_industry_map(["SZ000001"], category="申万二级", refresh=True)
  assert len(sw_queries) == 2


def test_get_stock_industry_map_batch_and_cache(monkeypatch):
  # Reset caches
  monkeypatch.setattr(stocks_mod, "_BLOCK_TREE_CACHE", None)
  monkeypatch.setattr(stocks_mod, "_BLOCK_CATEGORIES_INDEX", None)
  monkeypatch.setattr(stocks_mod, "_BLOCK_TO_STOCKS_CACHE", {})
  monkeypatch.setattr(stocks_mod, "_CATEGORY_STOCK_MAP_CACHE", {})

  tree = {
    "地域板块": ["上海市", "深圳市", "四川省"],
    "概念板块": ["人工智能"],
  }
  block_members = {
    "上海市": ["SH600000", "SH600016"],
    "深圳市": ["SZ000001", "SZ002415"],
    "四川省": ["SH600519"],
  }
  query_calls: list[str] = []

  def fake_load_tree(refresh: bool = False):
    cats = {}
    for cat, blks in tree.items():
      for b in blks:
        cats.setdefault(b, set()).add(cat)
    return tree, cats

  def fake_query_block(obj: str):
    query_calls.append(obj)
    return block_members.get(obj)

  monkeypatch.setattr(stocks_mod, "_load_block_tree", fake_load_tree)
  monkeypatch.setattr(stocks_mod, "_query_stock_block", fake_query_block)

  res = stocks_mod.get_stock_industry_map(["SH600000", "SH600519", "SH999999"], category="地域板块")
  assert res == {"SH600000": "上海市", "SH600519": "四川省"}
  assert len(query_calls) == 3  # queried 3 blocks once, not per stock

  # Second call hits cache without extra queries
  res_all = stocks_mod.get_stock_industry_map(None, category="地域板块")
  assert len(res_all) == 5
  assert res_all["SZ002415"] == "深圳市"
  assert len(query_calls) == 3

  # refresh=True re-queries
  stocks_mod.get_stock_industry_map(["SZ000001"], category="地域板块", refresh=True)
  assert len(query_calls) == 6


def test_get_stock_industry_map_unknown_category_fails_fast(monkeypatch):
  monkeypatch.setattr(stocks_mod, "_CATEGORY_STOCK_MAP_CACHE", {})
  monkeypatch.setattr(
    stocks_mod,
    "_load_block_tree",
    lambda refresh=False: ({"概念板块": ["人工智能"]}, {"人工智能": {"概念板块"}}),
  )

  with pytest.raises(ValueError, match="Unknown block category"):
    stocks_mod.get_stock_industry_map(["SH600000"], category="不存在分类")

  with pytest.raises(ValueError, match="non-empty string"):
    stocks_mod.get_stock_industry_map(["SH600000"], category="  ")
