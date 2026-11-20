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


def get_limit_ratio(objs: list[str], repeat: int = 1) -> np.ndarray:
  _ensure_stocks_loaded()
  if len(objs) == 1 and repeat <= 1:
    # fast path
    arr = ALL_STOCKS.filter(pl.col("obj") == objs[0]).get_column("limit_ratio").to_numpy()
    if len(arr) == 1:
      return arr
  df = pl.DataFrame({"obj": objs}, schema={"obj": pl.String})
  if repeat <= 1:
    return df.join(ALL_STOCKS, on="obj", how="left").get_column("limit_ratio").to_numpy()
  else:
    return df.join(ALL_STOCKS, on="obj", how="left").get_column("limit_ratio").to_numpy().repeat(repeat)


_SW_CATEGORY_COL_MAP: dict[str, str] = {
  "申万一级": "l1",
  "申万二级": "l2",
  "申万三级": "l3",
  "申万行业": "l2",
}
_SW_CANONICAL_CACHE_KEY: dict[str, str] = {
  "申万一级": "申万一级",
  "申万二级": "申万二级",
  "申万三级": "申万三级",
  "申万行业": "申万二级",
}
_SW_SCHEMA: dict[str, pl.DataType] = {
  "obj": pl.String,
  "l1": pl.String,
  "l2": pl.String,
  "l3": pl.String,
}

_SW_DF_CACHE: pl.DataFrame | None = None
_BLOCK_TREE_CACHE: dict[str, list[str]] | None = None
_BLOCK_CATEGORIES_INDEX: dict[str, set[str]] | None = None
_BLOCK_TO_STOCKS_CACHE: dict[str, list[str]] = {}
_STOCK_TO_BLOCKS_CACHE: dict[str, list[str]] = {}
_CATEGORY_STOCK_MAP_CACHE: dict[str, dict[str, str]] = {}


def _resolve_sw_column(category: str | None) -> tuple[str, str] | None:
  if category is None:
    return None
  cat = category.strip()
  col = _SW_CATEGORY_COL_MAP.get(cat)
  if col is None:
    return None
  return col, _SW_CANONICAL_CACHE_KEY[cat]


def _load_sw_df(refresh: bool = False) -> pl.DataFrame:
  """Load and cache the global Shenwan 3-level industry DataFrame from MSD ``stock_block``.

  Shenwan industries are stored separately from ``__block_tree__`` under
  ``obj = '__sw__'`` as a JSON array of ``{"obj", "l1", "l2", "l3"}`` records
  (see ``etc/sw.json`` for reference). Unlike general block memberships, each
  stock belongs to at most one Shenwan L1/L2/L3 industry per level.
  """
  global _SW_DF_CACHE
  if not refresh and _SW_DF_CACHE is not None:
    return _SW_DF_CACHE

  if refresh:
    for key in _SW_CATEGORY_COL_MAP:
      _CATEGORY_STOCK_MAP_CACHE.pop(key, None)

  conn, base_url = _get_msd_connection()
  raw_json: str | None = None

  def handler(t: pymsd.Table):
    nonlocal raw_json
    if t.rows > 0:
      raw_json = t[0, 1]

  conn.query_map("select * from stock_block where obj = '__sw__'", handler)
  if not raw_json:
    raise RuntimeError(
      f"Failed to load Shenwan industry table from MSD ({base_url}): "
      "table 'stock_block' returned no records for obj='__sw__'."
    )

  records = orjson.loads(raw_json)
  if not isinstance(records, list) or not records:
    raise RuntimeError(
      f"Unexpected '__sw__' payload from MSD ({base_url}): "
      f"expected non-empty JSON list, got {type(records).__name__}."
    )

  required_keys = ("obj", "l1", "l2", "l3")
  normalized_rows: list[dict[str, str]] = []
  for idx, rec in enumerate(records):
    if not isinstance(rec, dict) or not all(k in rec for k in required_keys):
      raise RuntimeError(
        f"Invalid '__sw__' record at index {idx} from MSD ({base_url}): "
        f"expected dict with keys {required_keys}."
      )
    normalized_rows.append({
      k: "" if rec[k] is None else str(rec[k]).strip()
      for k in required_keys
    })

  df = (
    pl.DataFrame(normalized_rows, schema=_SW_SCHEMA)
    .filter(pl.col("obj") != "")
    .unique(subset=["obj"], keep="first", maintain_order=True)
  )
  _SW_DF_CACHE = df
  return df


def _get_sw_stock_map(
  col: str,
  cache_key: str,
  *,
  refresh: bool = False,
) -> dict[str, str]:
  if refresh or cache_key not in _CATEGORY_STOCK_MAP_CACHE:
    df = _load_sw_df(refresh=refresh)
    objs = df.get_column("obj").to_list()
    vals = df.get_column(col).to_list()
    _CATEGORY_STOCK_MAP_CACHE[cache_key] = {
      obj: val for obj, val in zip(objs, vals) if obj and val
    }
  return _CATEGORY_STOCK_MAP_CACHE[cache_key]


def _available_categories(tree: dict[str, list[str]]) -> list[str]:
  return list(dict.fromkeys([*tree.keys(), *_SW_CATEGORY_COL_MAP.keys()]))


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


def _rank_close_names(name: str, all_names: Sequence[str], limit: int = 8) -> list[str]:
  sorted_names = sorted(set(all_names))
  lower_name = name.lower()
  substr_hits = [
    b for b in sorted_names
    if lower_name in b.lower() or b.lower() in lower_name
  ]
  fuzzy_hits = difflib.get_close_matches(name, sorted_names, n=limit, cutoff=0.4)
  return list(dict.fromkeys(substr_hits + fuzzy_hits))[:limit]


def _suggest_blocks(
  name: str,
  block_to_cats: dict[str, set[str]],
  limit: int = 8,
) -> list[str]:
  merged = _rank_close_names(name, list(block_to_cats.keys()), limit=limit)
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

  Special handling for Shenwan (申万) industries:
    Unlike general blocks stored in ``__block_tree__`` and per-stock ``stock_block``
    rows, Shenwan industries are resolved from the global 3-level DataFrame loaded
    from ``stock_block`` record ``obj='__sw__'`` (``l1``/``l2``/``l3``):

    - ``"申万一级"`` -> Level 1 (``l1``, e.g. ``"银行"``)
    - ``"申万二级"`` -> Level 2 (``l2``, e.g. ``"股份制银行Ⅱ"``)
    - ``"申万三级"`` -> Level 3 (``l3``, e.g. ``"股份制银行Ⅲ"``)
    - ``"申万行业"`` -> Alias of ``"申万二级"`` (``l2``)

    When ``category`` is ``None``, only per-stock ``stock_block`` entries are
    queried and ``__sw__`` is not merged in. Because each stock belongs to a
    single Shenwan industry per level, multi-stock calls with a Shenwan
    ``category`` return ``[industry]`` when all stocks share that industry or
    ``[]`` otherwise.

  Args:
    stocks: A single stock code (e.g. ``"SH600000"``) or a sequence of stock codes
      (e.g. ``["SH600000", "SH601398"]``) with market prefixes (``SH``/``SZ``/``BJ``).
    category: Optional block category filter. Accepts Shenwan levels
      (``"申万一级"``, ``"申万二级"``, ``"申万三级"``, or ``"申万行业"`` alias) or
      any top-level category in ``__block_tree__`` (e.g. ``"概念板块"``,
      ``"地区板块"``, ``"中证指数"``). When ``None``, all ``stock_block`` blocks
      for the stock(s) are returned; unknown categories raise ``ValueError``.
    refresh: If ``True``, bypass the in-memory cache and re-query MSD (``MSD_HOST``).

  Returns:
    Deduplicated list of block/industry names in order; returns ``[]`` if multiple
    stocks share no common entry.

  Raises:
    ValueError: If ``stocks`` is empty, ``category`` is unknown, or any stock has
      no record in ``__sw__`` (for Shenwan categories) or ``stock_block`` (for
      general blocks).
    RuntimeError: If MSD ``__sw__`` or ``__block_tree__`` cannot be loaded.

  Examples:
    >>> get_industry_by_stocks("SH600000", category="申万一级")
    ['银行']
    >>> get_industry_by_stocks("SH600000", category="申万二级")
    ['股份制银行Ⅱ']
    >>> get_industry_by_stocks("SH600000", category="申万行业")
    ['股份制银行Ⅱ']
    >>> get_industry_by_stocks(["SH600000", "SH601398"], category="申万一级")
    ['银行']
  """
  stock_list = _normalize_names(stocks, "stocks")

  sw_info = _resolve_sw_column(category)
  if sw_info is not None:
    col, cache_key = sw_info
    sw_map = _get_sw_stock_map(col, cache_key, refresh=refresh)
    per_stock_sw: list[list[str]] = []
    for obj in stock_list:
      ind = sw_map.get(obj)
      if ind is None:
        raise ValueError(
          f"Stock {obj!r} not found in Shenwan industry data ('__sw__') "
          "(ensure market prefix like 'SH600000' or 'SZ000001')."
        )
      per_stock_sw.append([ind])
    return _intersect_lists(per_stock_sw)

  allowed_blocks: set[str] | None = None
  if category is not None:
    if not isinstance(category, str) or not category.strip():
      raise ValueError("category must be a non-empty string or None")
    cat = category.strip()
    tree, _ = _load_block_tree(refresh=refresh)
    if cat not in tree:
      raise ValueError(
        f"Unknown block category {cat!r}. Available categories: {_available_categories(tree)}"
      )
    allowed_blocks = set(tree[cat])

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
  category: str | None = "申万二级",
  refresh: bool = False,
) -> list[str]:
  """Look up constituent stocks for one or more industries or blocks.

  Special handling for Shenwan (申万) industries vs general ``stock_block`` blocks:
    - **Shenwan categories** (``"申万一级"``, ``"申万二级"`` [default],
      ``"申万三级"``, and ``"申万行业"`` aliased to ``"申万二级"``) resolve
      from the global ``__sw__`` DataFrame. Because every stock has a unique
      industry at each Shenwan level, passing multiple Shenwan industry names
      returns their **union** (e.g. ``["股份制银行Ⅱ", "国有大型银行Ⅱ"]`` returns
      stocks in either industry). Industry names must match ``__sw__`` **exactly**,
      including Unicode Roman numeral suffixes such as ``"Ⅱ"`` and ``"Ⅲ"``
      (e.g. ``"股份制银行Ⅱ"``, ``"IT服务Ⅱ"``); unknown names fail fast with
      close-match suggestions from that level.
    - **Non-Shenwan ``stock_block`` categories** (e.g. ``"概念板块"``,
      ``"中证指数"``) or ``category=None`` (all ``__block_tree__`` blocks)
      query ``stock_block`` and return the **intersection** of constituents
      while preserving the order from the first block (e.g. stocks in both
      ``"沪深300"`` and ``"银行"``).

  Args:
    industries: A single industry/block name (e.g. ``"股份制银行Ⅱ"``) or a
      sequence of names (e.g. ``["股份制银行Ⅱ", "城商行Ⅱ"]`` or
      ``["沪深300", "银行"]`` with ``category=None``).
    category: Category scope for lookup. Defaults to ``"申万二级"``. Accepts
      ``"申万一级"``, ``"申万二级"``, ``"申万三级"``, ``"申万行业"`` (alias of
      ``"申万二级"``), any top-level category in ``__block_tree__``, or ``None``
      to search across all ``__block_tree__`` blocks.
    refresh: If ``True``, bypass the in-memory cache and re-query MSD (``MSD_HOST``).

  Returns:
    Deduplicated list of prefixed stock codes (e.g. ``["SZ000001", ...]``);
    for intersection queries (non-Shenwan / ``None``), returns ``[]`` if blocks
    have no overlapping constituents.

  Raises:
    ValueError: If ``industries`` is empty, ``category`` is invalid/unknown, or
      any requested name does not exist in the selected category (error message
      includes close-match suggestions).
    RuntimeError: If MSD ``__sw__`` or ``__block_tree__`` cannot be loaded, or a
      ``__block_tree__`` block returns no record in ``stock_block``.

  Examples:
    >>> get_stock_by_industries("股份制银行Ⅱ")
    ['SH600000', 'SZ000001', ...]
    >>> get_stock_by_industries(["股份制银行Ⅱ", "国有大型银行Ⅱ"])
    ['SH600000', 'SH601398', 'SZ000001', ...]
    >>> get_stock_by_industries("银行", category="申万一级")
    ['SH600000', 'SH601398', 'SZ000001', ...]
    >>> get_stock_by_industries(["沪深300", "银行"], category=None)
    ['SH600000', 'SH601398']
  """
  industry_list = _normalize_names(industries, "industries")

  if category is not None:
    if not isinstance(category, str) or not category.strip():
      raise ValueError("category must be a non-empty string or None")
    cat: str | None = category.strip()
  else:
    cat = None

  sw_info = _resolve_sw_column(cat)
  if sw_info is not None:
    col, canonical_cat = sw_info
    df = _load_sw_df(refresh=refresh)
    valid_names = {
      val for val in df.get_column(col).to_list()
      if isinstance(val, str) and val
    }
    for ind in industry_list:
      if ind not in valid_names:
        suggestions = _rank_close_names(ind, list(valid_names))
        hint = (
          f" Did you mean: {', '.join(repr(s) for s in suggestions)}?"
          if suggestions
          else ""
        )
        raise ValueError(
          f"Unknown Shenwan industry {ind!r} under category {canonical_cat!r} "
          f"(exact match required, including 'Ⅱ'/'Ⅲ' suffixes if present).{hint}"
        )
    return df.filter(pl.col(col).is_in(industry_list)).get_column("obj").to_list()

  if cat is not None:
    tree, _ = _load_block_tree(refresh=refresh)
    if cat not in tree:
      raise ValueError(
        f"Unknown block category {cat!r}. Available categories: {_available_categories(tree)}"
      )
    cat_blocks = set(tree[cat])
    cat_block_index = {b: {cat} for b in cat_blocks}
    for ind in industry_list:
      if ind not in cat_blocks:
        suggestions = _suggest_blocks(ind, cat_block_index)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ValueError(f"Unknown block {ind!r} under category {cat!r} in 'stock_block'.{hint}")
  else:
    _, block_to_cats = _load_block_tree(refresh=refresh)
    for ind in industry_list:
      if ind not in block_to_cats:
        suggestions = _suggest_blocks(ind, block_to_cats)
        hint = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
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


def get_stock_industry_map(
  objs: Sequence[str] | None = None,
  *,
  category: str = "申万二级",
  refresh: bool = False,
) -> dict[str, str]:
  """Batch-resolve stock -> industry/block mapping for a given category.

  Special handling for Shenwan (申万) categories:
    When ``category`` is one of ``"申万一级"``, ``"申万二级"`` (default),
    ``"申万三级"``, or ``"申万行业"`` (equivalent to ``"申万二级"``), the mapping
    is built directly from the global ``__sw__`` DataFrame in a single MSD query
    and cached under the canonical Shenwan level (so ``"申万行业"`` and
    ``"申万二级"`` share one cache entry).

    For other categories in ``__block_tree__``, this function queries the
    constituent list of each block under ``category`` (avoiding ~5,000 per-stock
    queries) and caches the inverted map in memory.

  Args:
    objs: Optional sequence of stock codes to filter the returned mapping. When
      ``None``, returns the complete mapping for every stock in ``category``.
    category: Shenwan level (``"申万一级"``, ``"申万二级"``, ``"申万三级"``,
      ``"申万行业"``) or top-level block category in ``__block_tree__``
      (default ``"申万二级"``).
    refresh: If ``True``, bypass caches and re-query MSD.

  Returns:
    Dictionary mapping stock code to its industry/block name within ``category``.
    Stocks absent from ``__sw__`` or the target category are omitted from the
    returned dictionary.

  Raises:
    ValueError: If ``category`` is empty or not recognized.
    RuntimeError: If MSD ``__sw__`` or ``__block_tree__`` cannot be loaded.
  """
  if not isinstance(category, str) or not category.strip():
    raise ValueError("category must be a non-empty string")
  cat = category.strip()

  sw_info = _resolve_sw_column(cat)
  if sw_info is not None:
    col, cache_key = sw_info
    cached = _get_sw_stock_map(col, cache_key, refresh=refresh)
  else:
    if refresh or cat not in _CATEGORY_STOCK_MAP_CACHE:
      tree, _ = _load_block_tree(refresh=refresh)
      if cat not in tree:
        raise ValueError(
          f"Unknown block category {cat!r}. Available categories: {_available_categories(tree)}"
        )
      full_map: dict[str, str] = {}
      for block_name in tree[cat]:
        if refresh or block_name not in _BLOCK_TO_STOCKS_CACHE:
          members = _query_stock_block(block_name)
          if members is None:
            continue
          _BLOCK_TO_STOCKS_CACHE[block_name] = members
        for sym in _BLOCK_TO_STOCKS_CACHE[block_name]:
          if sym not in full_map:
            full_map[sym] = block_name
      _CATEGORY_STOCK_MAP_CACHE[cat] = full_map
    cached = _CATEGORY_STOCK_MAP_CACHE[cat]

  if objs is None:
    return dict(cached)
  return {sym: cached[sym] for sym in objs if sym in cached}


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
  print(ALL_STOCKS)  # type: ignore[unbound-name]
  print(len(A_STOCKS))  # type: ignore[unbound-name]
  print(len(FOUNDS))  # type: ignore[unbound-name]
  print(len(A_STOCKS_EXCLUDE_ST))  # type: ignore[unbound-name]

  r1 = get_limit_ratio(["SH600000", "SZ300750", "SZ300477"])  
  r2 = get_limit_ratio(["SH600000", "SZ300750", "SZ300477"], repeat=2)

  print(r1) # should be [0.1, 0.2, 0.05]
  print(r2) # should be [0.1, 0.1, 0.2, 0.2, 0.05, 0.05]

  with open("a.txt", "w") as fp:
    for obj in A_STOCKS_EXCLUDE_ST:  # type: ignore[unbound-name]
      fp.write(f"{obj}\n")