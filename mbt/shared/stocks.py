
from typing import TYPE_CHECKING
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

def load_stocks_from_db() -> pl.DataFrame:
  base_url = os.environ.get("MSD_HOST", "")
  if base_url == "":
    raise Exception("MSD_HOST env is empty, please set it")
  conn = pymsd.MsdConnection(base_url)
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
    raise Exception("stocks load failed")
  return normalize_stocks(df)


def normalize_stocks(df: pl.DataFrame) -> pl.DataFrame:
  """把原始股票信息规整为带类型的列: kind(int), status(int), limit_ratio(float)

  stock_infos 里的 kind 是字符串("Stock"/"Fund"/"Index"), 而下游常量是整数,
  直接比较会抛 ComputeError; 同时 limit_ratio 列必须在此生成,
  否则 get_limit_ratio 会因缺列而失败。
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