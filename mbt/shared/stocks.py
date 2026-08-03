import orjson
import polars as pl
import os
import numpy as np



KIND_STOCK = 1
KIND_FUND = 2
KIND_INDEX = 3

STATUS_NORMAL = 0
STATUS_ST = 1
STATUS_STAR_ST = 2

def parse_obj_kind(kind: str) -> int:
  match kind.lower():
    case 'stock':
      return KIND_STOCK
    case 'fund':
      return KIND_FUND
    case 'index':
      return KIND_INDEX
  return 0


def parse_obj_status(name: str) -> int:
  if 'ST' in name:
    return 1
  if '*ST' in name:
    return 2
  return 0

def build_limit_ratio(row: dict[str, str]) -> float:
  symbol = row['symbol']
  name = row['name']
  if name.startswith("ST") or name.startswith("*ST"):
    return 0.05
  elif symbol.startswith("SZ30") or symbol.startswith("SH68"): #科创板和创业板
    return 0.2
  return 0.1


def load_all_stocks(path: str) -> pl.DataFrame:
  with open(path, encoding='utf-8') as fp:
    content = orjson.loads(fp.read())
    df = pl.DataFrame(list(content.values()), schema={
      'symbol': pl.String,
      'name': pl.String,
      "volume_unit": pl.UInt8,
      "price_decimal": pl.String,
      'kind': pl.String,
    })

  df = df.with_columns([
    pl.col('kind').map_elements(parse_obj_kind, return_dtype=pl.Int8),
    pl.col('name').map_elements(parse_obj_status, return_dtype=pl.Int8).alias("status"),
    pl.struct([pl.col("symbol"), pl.col("name")]).map_elements(build_limit_ratio, return_dtype=pl.Float64).alias("limit_ratio")
  ])
  return df



def where_clause(symbols: list[str], col: str = 'obj') -> str:
  sql = ','.join([f'"{s}"' for s in symbols])
  sql = f"{col} IN ({sql})"
  return sql


def get_limit_ratio(symbols: list[str], repeat: int = 1) -> np.ndarray | float:
  if len(symbols) <= 1 and repeat <= 1:
    # fast path
    return ALL_STOCKS.filter(pl.col("symbol") == symbols[0]).get_column("limit_ratio").item()
  df = pl.DataFrame({"symbol": symbols}, schema={"symbol": pl.String})
  if repeat <= 1:
    return df.join(ALL_STOCKS, on="symbol", how="left").get_column("limit_ratio").to_numpy()
  else:
    return df.join(ALL_STOCKS, on="symbol", how="left").get_column("limit_ratio").to_numpy().repeat(repeat)


  

ALL_STOCKS = load_all_stocks(os.environ.get("MSD_STOCKS_FILE", 'etc/stocks.json'))
A_STOCKS = ALL_STOCKS.filter(pl.col("kind") == KIND_STOCK).get_column("symbol").to_list()
A_STOCKS_EXCLUDE_ST = ALL_STOCKS.filter(
  (pl.col("kind") == KIND_STOCK) &
  (pl.col("status") == STATUS_NORMAL)
).get_column("symbol").to_list()
FOUNDS = ALL_STOCKS.filter(pl.col("kind") == KIND_FUND).get_column("symbol").to_list()


if __name__ == "__main__":
  print(ALL_STOCKS)
  print(len(A_STOCKS))
  print(len(FOUNDS))
  print(len(A_STOCKS_EXCLUDE_ST))

  r1 = get_limit_ratio(["SH600000", "SZ300750", "SZ300477"])  
  r2 = get_limit_ratio(["SH600000", "SZ300750", "SZ300477"], repeat=2)

  print(r1) # should be [0.1, 0.2, 0.05]
  print(r2) # should be [0.1, 0.1, 0.2, 0.2, 0.05, 0.05]

  with open("a.txt", "w") as fp:
    for symbol in A_STOCKS_EXCLUDE_ST:
      fp.write(f"{symbol}\n")