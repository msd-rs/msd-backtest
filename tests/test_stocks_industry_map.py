# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for batch stock-to-industry mapping in mbt.shared.stocks."""

from __future__ import annotations

import pytest

import mbt.shared.stocks as stocks_mod


def test_get_stock_industry_map_batch_and_cache(monkeypatch):
  # Reset caches
  monkeypatch.setattr(stocks_mod, "_BLOCK_TREE_CACHE", None)
  monkeypatch.setattr(stocks_mod, "_BLOCK_CATEGORIES_INDEX", None)
  monkeypatch.setattr(stocks_mod, "_BLOCK_TO_STOCKS_CACHE", {})
  monkeypatch.setattr(stocks_mod, "_CATEGORY_STOCK_MAP_CACHE", {})

  tree = {
    "申万行业": ["银行", "食品饮料", "电子"],
    "概念板块": ["人工智能"],
  }
  block_members = {
    "银行": ["SH600000", "SZ000001"],
    "食品饮料": ["SH600519", "SZ000858"],
    "电子": ["SZ002415"],
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

  res = stocks_mod.get_stock_industry_map(["SH600000", "SH600519", "SH999999"], category="申万行业")
  assert res == {"SH600000": "银行", "SH600519": "食品饮料"}
  assert len(query_calls) == 3  # queried 3 industries once, not per stock

  # Second call hits cache without extra queries
  res_all = stocks_mod.get_stock_industry_map(None, category="申万行业")
  assert len(res_all) == 5
  assert res_all["SZ002415"] == "电子"
  assert len(query_calls) == 3

  # refresh=True re-queries
  stocks_mod.get_stock_industry_map(["SZ000001"], category="申万行业", refresh=True)
  assert len(query_calls) == 6


def test_get_stock_industry_map_unknown_category_fails_fast(monkeypatch):
  monkeypatch.setattr(stocks_mod, "_CATEGORY_STOCK_MAP_CACHE", {})
  monkeypatch.setattr(
    stocks_mod,
    "_load_block_tree",
    lambda refresh=False: ({"申万行业": ["银行"]}, {"银行": {"申万行业"}}),
  )

  with pytest.raises(ValueError, match="Unknown block category"):
    stocks_mod.get_stock_industry_map(["SH600000"], category="不存在分类")

  with pytest.raises(ValueError, match="non-empty string"):
    stocks_mod.get_stock_industry_map(["SH600000"], category="  ")
