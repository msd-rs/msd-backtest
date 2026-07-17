import orjson
import polars as pl
import os



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


def load_all_stocks(path: str) -> pl.DataFrame:
  with open(path) as fp:
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
  ])
  return df



def where_clause(symbols: list[str], col: str = 'obj') -> str:
  sql = ','.join([f'"{s}"' for s in symbols])
  sql = f"{col} IN ({sql})"
  return sql

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

  with open("a.txt", "w") as fp:
    for symbol in A_STOCKS_EXCLUDE_ST:
      fp.write(f"{symbol}\n")