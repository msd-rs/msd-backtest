import orjson
import pandas as pd
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


def load_all_stocks(path: str) -> pd.DataFrame:
  with open(path) as fp:
    content = orjson.loads(fp.read())
    df = pd.DataFrame.from_records(list(content.values())).drop(columns=['prev_close'])

  df['kind'] = df['kind'].map(parse_obj_kind)
  df['status'] = df['name'].map(parse_obj_status)
  return df



def where_clause(symbols: list[str], col: str = 'obj') -> str:
  sql = ','.join([f'"{s}"' for s in symbols])
  sql = f"{col} IN ({sql})"
  return sql

ALL_STOCKS = load_all_stocks(os.environ.get("MSD_STOCKS_FILE", 'etc/stocks.json'))
A_STOCKS = ALL_STOCKS.loc[ALL_STOCKS['kind'] == KIND_STOCK, 'symbol'].tolist()
A_STOCKS_EXCLUDE_ST = ALL_STOCKS.loc[
  (ALL_STOCKS['kind'] == KIND_STOCK) &
  (ALL_STOCKS['status'] == STATUS_NORMAL),
  'symbol'
].tolist()
FOUNDS = ALL_STOCKS.loc[ALL_STOCKS['kind'] == KIND_FUND, 'symbol'].tolist()


if __name__ == "__main__":
  print(ALL_STOCKS)
  print(len(A_STOCKS))
  print(len(FOUNDS))
  print(len(A_STOCKS_EXCLUDE_ST))

  