# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Stock universe, price-limit, and industry/block lookup utilities backed by MSD."""


from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from typing import TYPE_CHECKING
from collections.abc import Sequence
import difflib
import orjson
import polars as pl
import os
import numpy as np
import pymsd



if TYPE_CHECKING:
  ALL_STOCKS: pl.DataFrame
  A_STOCKS: list[str]
  A_STOCKS_EXCLUDE_ST: list[str]
  FOUNDS: list[str]


KIND_STOCK = 1
KIND_FUND = 2
KIND_INDEX = 3

STATUS_NORMAL = 0
STATUS_ST = 1
STATUS_STAR_ST = 2

JSON_SCHEMA = {
      'obj': pl.String,
      'name': pl.String,
      "volume_unit": pl.UInt8,
      "price_decimal": pl.String,
      'kind': pl.String,
    }

def parse_obj_kind(kind: str) -> int:
  match kind.lower():
    case 'stock':
      return KIND_STOCK
    case 'fund':
      return KIND_FUND
    case 'index':
      return KIND_INDEX
  return 0


def parse_obj_status(name: str | None) -> int:
  if not name:
    return STATUS_NORMAL
  if '*ST' in name:
    return STATUS_STAR_ST
  if 'ST' in name:
    return STATUS_ST
  return STATUS_NORMAL

def build_limit_ratio(row: dict[str, str]) -> float:
  obj = row.get('obj') or ''
  name = row.get('name') or ''
  if name.startswith("ST") or name.startswith("*ST"):
    return 0.05
  elif obj.startswith("SZ30") or obj.startswith("SH68"): #科创板和创业板
    return 0.2
  return 0.1

def _get_msd_connection() -> tuple[pymsd.MsdConnection, str]:
  base_url = os.environ.get("MSD_HOST", "http://localhost:50510")
  if base_url == "":
    raise ValueError("MSD_HOST env is empty, please set it")
  return pymsd.MsdConnection(base_url), base_url


def load_stocks_from_db() -> pl.DataFrame:
  conn, base_url = _get_msd_connection()
  df = None
  def handler(t: pymsd.Table):
    nonlocal df
    m : dict[str, dict] = orjson.loads(t[0, 1])
    for k, r in m.items():
      if "obj" not in r:
        r["obj"] = r.get("symbol", k)
    df = pl.from_records(list(m.values()), schema=JSON_SCHEMA, orient='row')

  conn.query_map("select * from stock_infos where obj='stocks'", handler)
  if df is None or len(df) == 0:
    raise RuntimeError(
      f"Failed to load stock universe from MSD ({base_url}): table 'stock_infos' returned no records for obj='stocks'. "
      "Please verify MSD service status and data initialization via `python -m mbt.doctor`."
    )
  return normalize_stocks(df)


def normalize_stocks(df: pl.DataFrame) -> pl.DataFrame:
  """Normalize raw stock info into typed columns: kind(int), status(int), limit_ratio(float).

  ``stock_infos`` stores ``kind`` as strings ("Stock"/"Fund"/"Index") while
  downstream constants are integers, so direct comparison would raise
  ``ComputeError``. The ``limit_ratio`` column must also be computed here,
  otherwise ``get_limit_ratio`` will fail due to a missing column.
  """
  kind = pl.col("kind")
  if df.schema["kind"] == pl.String:
    kind = kind.str.to_lowercase().replace_strict(
      {"stock": KIND_STOCK, "fund": KIND_FUND, "index": KIND_INDEX},
      default=0, return_dtype=pl.Int8,
    )
  else:
    kind = kind.cast(pl.Int8)
  return df.with_columns([
    kind.alias("kind"),
    pl.col("name").map_elements(parse_obj_status, return_dtype=pl.Int8).alias("status"),
    pl.struct([pl.col("obj"), pl.col("name")])
      .map_elements(build_limit_ratio, return_dtype=pl.Float64).alias("limit_ratio"),
  ])



def where_clause(objs: list[str], col: str = 'obj') -> str:
  sql = ','.join([f'"{s}"' for s in objs])
  sql = f"{col} IN ({sql})"
  return sql


def get_limit_ratio(objs: list[str], repeat: int = 1) -> np.ndarray | float:
  _ensure_stocks_loaded()
  if len(objs) == 1 and repeat <= 1:
    # fast path
    return ALL_STOCKS.filter(pl.col("obj") == objs[0]).get_column("limit_ratio").item()
  df = pl.DataFrame({"obj": objs}, schema={"obj": pl.String})
  if repeat <= 1:
    return df.join(ALL_STOCKS, on="obj", how="left").get_column("limit_ratio").to_numpy()
  else:
    return df.join(ALL_STOCKS, on="obj", how="left").get_column("limit_ratio").to_numpy().repeat(repeat)


_BLOCK_TREE_CACHE: dict[str, list[str]] | None = None
_BLOCK_CATEGORIES_INDEX: dict[str, set[str]] | None = None
_BLOCK_TO_STOCKS_CACHE: dict[str, list[str]] = {}
_STOCK_TO_BLOCKS_CACHE: dict[str, list[str]] = {}


def _load_block_tree(refresh: bool = False) -> tuple[dict[str, list[str]], dict[str, set[str]]]:
  global _BLOCK_TREE_CACHE, _BLOCK_CATEGORIES_INDEX
  if not refresh and _BLOCK_TREE_CACHE is not None and _BLOCK_CATEGORIES_INDEX is not None:
    return _BLOCK_TREE_CACHE, _BLOCK_CATEGORIES_INDEX

  conn, base_url = _get_msd_connection()
  raw_json: str | None = None

  def handler(t: pymsd.Table):
    nonlocal raw_json
    if t.rows > 0:
      raw_json = t[0, 1]

  conn.query_map("select * from stock_block where obj = '__block_tree__'", handler)
  if not raw_json:
    raise RuntimeError(
      f"Failed to load block tree from MSD ({base_url}): table 'stock_block' returned no records for obj='__block_tree__'."
    )

  tree: dict[str, list[str]] = orjson.loads(raw_json)
  if not isinstance(tree, dict):
    raise RuntimeError(
      f"Unexpected '__block_tree__' payload type from MSD ({base_url}): expected dict, got {type(tree).__name__}."
    )

  block_to_cats: dict[str, set[str]] = {}
  for cat, blocks in tree.items():
    for b in blocks:
      block_to_cats.setdefault(b, set()).add(cat)

  _BLOCK_TREE_CACHE = tree
  _BLOCK_CATEGORIES_INDEX = block_to_cats
  return tree, block_to_cats


def _query_stock_block(obj: str) -> list[str] | None:
  conn, _ = _get_msd_connection()
  safe_obj = obj.replace("'", "''")
  raw_val: str | None = None

  def handler(t: pymsd.Table):
    nonlocal raw_val
    if t.rows > 0:
      raw_val = t[0, 1]

  conn.query_map(f"select * from stock_block where obj = '{safe_obj}'", handler)
  if raw_val is None:
    return None
  items = [s.strip() for s in raw_val.split(",") if s.strip()]
  return list(dict.fromkeys(items))


def _normalize_names(items: str | Sequence[str], param_name: str) -> list[str]:
  if isinstance(items, str):
    names = [items]
  elif isinstance(items, Sequence):
    names = list(items)
  else:
    raise TypeError(f"{param_name} must be a str or Sequence[str], got {type(items).__name__}")

  if not names or any(not isinstance(s, str) or not s.strip() for s in names):
    raise ValueError(f"{param_name} must be a non-empty string or non-empty sequence of non-empty strings")
  return list(dict.fromkeys(s.strip() for s in names))


def _intersect_lists(lists: list[list[str]]) -> list[str]:
  if len(lists) == 1:
    return list(lists[0])
  common = set(lists[0]).intersection(*lists[1:])
  return [x for x in lists[0] if x in common]


def _suggest_blocks(
  name: str,
  block_to_cats: dict[str, set[str]],
  limit: int = 8,
) -> list[str]:
  all_blocks = sorted(block_to_cats.keys())
  lower_name = name.lower()
  substr_hits = [
    b for b in all_blocks
    if lower_name in b.lower() or b.lower() in lower_name
  ]
  fuzzy_hits = difflib.get_close_matches(name, all_blocks, n=limit, cutoff=0.4)
  merged = list(dict.fromkeys(substr_hits + fuzzy_hits))[:limit]
  return [f"'{b}' [{', '.join(sorted(block_to_cats[b]))}]" for b in merged]


def get_industry_by_stocks(
  stocks: str | Sequence[str],
  *,
  category: str | None = None,
  refresh: bool = False,
) -> list[str]:
  """Look up blocks (industries, concepts, indices, regions) for one or more stocks.

  When multiple stocks are provided, returns the **intersection** of their blocks
  while preserving the order from the first stock.

  Args:
    stocks: A single stock code (e.g. ``"SH600000"``) or a sequence of stock codes
      (e.g. ``["SH600000", "SH601398"]``) with market prefixes (``SH``/``SZ``/``BJ``).
    category: Optional top-level block category filter (e.g. ``"申万行业"``,
      ``"概念板块"``, ``"地区板块"``, ``"中证指数"``). When ``None``, all blocks
      are returned; unknown categories raise ``ValueError``.
    refresh: If ``True``, bypass the in-memory cache and re-query MSD (``MSD_HOST``).

  Returns:
    Deduplicated list of block names in order; returns ``[]`` if multiple stocks
    share no common blocks.

  Raises:
    ValueError: If ``stocks`` is empty, ``category`` is unknown, or any stock has
      no record in table ``stock_block``.

  Examples:
    >>> get_industry_by_stocks("SH600000", category="申万行业")
    ['银行']
    >>> get_industry_by_stocks(["SH600000", "SH601398"], category="申万行业")
    ['银行']
  """
  stock_list = _normalize_names(stocks, "stocks")

  allowed_blocks: set[str] | None = None
  if category is not None:
    tree, _ = _load_block_tree(refresh=refresh)
    if category not in tree:
      raise ValueError(
        f"Unknown block category {category!r}. Available categories: {list(tree.keys())}"
      )
    allowed_blocks = set(tree[category])

  per_stock_blocks: list[list[str]] = []
  for obj in stock_list:
    if refresh or obj not in _STOCK_TO_BLOCKS_CACHE:
      blocks = _query_stock_block(obj)
      if blocks is None:
        raise ValueError(
          f"Stock {obj!r} not found in table 'stock_block' "
          "(ensure market prefix like 'SH600000' or 'SZ000001')."
        )
      _STOCK_TO_BLOCKS_CACHE[obj] = blocks
    per_stock_blocks.append(_STOCK_TO_BLOCKS_CACHE[obj])

  result = _intersect_lists(per_stock_blocks)
  if allowed_blocks is not None:
    result = [b for b in result if b in allowed_blocks]
  return result


def get_stock_by_industries(
  industries: str | Sequence[str],
  *,
  refresh: bool = False,
) -> list[str]:
  """Look up constituent stocks for one or more blocks (industries, concepts, indices, regions).

  When multiple blocks are provided, returns the **intersection** of their
  constituents while preserving the order from the first block (e.g. stocks in
  both ``"沪深300"`` and ``"银行"``). Block names must match ``__block_tree__``
  exactly; unknown names fail fast with close-match suggestions instead of
  silently returning an empty list.

  Args:
    industries: A single block name (e.g. ``"银行"``) or a sequence of block names
      (e.g. ``["沪深300", "银行"]``).
    refresh: If ``True``, bypass the in-memory cache and re-query MSD (``MSD_HOST``).

  Returns:
    Deduplicated list of prefixed stock codes (e.g. ``["SZ000001", ...]``);
    returns ``[]`` if multiple blocks have no overlapping constituents.

  Raises:
    ValueError: If ``industries`` is empty or contains an unknown block name
      (error message includes close-match suggestions).
    RuntimeError: If a block exists in ``__block_tree__`` but ``stock_block``
      returns no record for it.

  Examples:
    >>> get_stock_by_industries("银行")
    ['SZ000001', ...]
    >>> get_stock_by_industries(["沪深300", "银行"])
    ['SZ000001', ...]
  """
  industry_list = _normalize_names(industries, "industries")
  _, block_to_cats = _load_block_tree(refresh=refresh)

  for ind in industry_list:
    if ind not in block_to_cats:
      suggestions = _suggest_blocks(ind, block_to_cats)
      hint = (
        f" Did you mean: {', '.join(suggestions)}?"
        if suggestions
        else ""
      )
      raise ValueError(f"Unknown block {ind!r} in 'stock_block'.{hint}")

  per_block_stocks: list[list[str]] = []
  for ind in industry_list:
    if refresh or ind not in _BLOCK_TO_STOCKS_CACHE:
      objs = _query_stock_block(ind)
      if objs is None:
        raise RuntimeError(
          f"Block {ind!r} exists in '__block_tree__' but returned no records in 'stock_block'."
        )
      _BLOCK_TO_STOCKS_CACHE[ind] = objs
    per_block_stocks.append(_BLOCK_TO_STOCKS_CACHE[ind])

  return _intersect_lists(per_block_stocks)


_LAZY_VARS = ("ALL_STOCKS", "A_STOCKS", "A_STOCKS_EXCLUDE_ST", "FOUNDS")


def _ensure_stocks_loaded():
  g = globals()
  if "ALL_STOCKS" not in g:
    all_stocks = load_stocks_from_db()
    g["ALL_STOCKS"] = all_stocks
    g["A_STOCKS"] = all_stocks.filter(pl.col("kind") == KIND_STOCK).get_column("obj").to_list()
    g["A_STOCKS_EXCLUDE_ST"] = all_stocks.filter(
      (pl.col("kind") == KIND_STOCK) &
      (pl.col("status") == STATUS_NORMAL)
    ).get_column("obj").to_list()
    g["FOUNDS"] = all_stocks.filter(pl.col("kind") == KIND_FUND).get_column("obj").to_list()


def __getattr__(name: str):
  if name in _LAZY_VARS:
    _ensure_stocks_loaded()
    return globals()[name]
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
  _ensure_stocks_loaded()
  #df = load_stocks_from_db()
  #print(df)
  print(ALL_STOCKS)
  print(len(A_STOCKS))
  print(len(FOUNDS))
  print(len(A_STOCKS_EXCLUDE_ST))

  r1 = get_limit_ratio(["SH600000", "SZ300750", "SZ300477"])  
  r2 = get_limit_ratio(["SH600000", "SZ300750", "SZ300477"], repeat=2)

  print(r1) # should be [0.1, 0.2, 0.05]
  print(r2) # should be [0.1, 0.1, 0.2, 0.2, 0.05, 0.05]

  with open("a.txt", "w") as fp:
    for obj in A_STOCKS_EXCLUDE_ST:
      fp.write(f"{obj}\n")