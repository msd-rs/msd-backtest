import logging
from dotenv import load_dotenv

load_dotenv(override=True)

from typing import Any
from mbt.select import Selector, SelectorDataProvider, MsdSelectorDataProvider
import mbt.select.stocks 
import logging
from typing import Tuple
import argparse
import os
import sys
import importlib.util

logging.basicConfig(
  format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO
)

logger = logging.getLogger("selector")
logger.setLevel(logging.DEBUG)


def expand_symbols(symbols: list[str]) -> list[str]:
  """read as file when input is a file path, and expand it. otherwise, return the input as a list of symbols"""
  expanded = set()
  for s in symbols:
    if os.path.isfile(s):
      with open(s, "r") as f:
        lines = filter(
          lambda x: len(x) > 0 and not x.startswith("#"),
          map(lambda x: x.strip(), f.readlines()),
        )
        expanded.update(lines)
    else:
      match s:
        case "A_STOCKS_EXCLUDE_ST":
          expanded.update(mbt.select.stocks.A_STOCKS_EXCLUDE_ST)
        case "A_STOCKS":
          expanded.update(mbt.select.stocks.A_STOCKS)
        case "FOUNDS":
          expanded.update(mbt.select.stocks.FOUNDS)
        case _:
          expanded.add(s)

  l = list(expanded)
  l.sort()
  return l


def load_selector(py_file: str, **kwargs) -> Selector:
  path = os.path.abspath(py_file)
  dir_path = os.path.dirname(path)
  if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

  module_name = os.path.splitext(os.path.basename(path))[0]
  spec = importlib.util.spec_from_file_location(module_name, path)
  if spec and spec.loader:
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for obj in vars(module).values():
      if isinstance(obj, type) and issubclass(obj, Selector) and obj is not Selector:
        return obj(**kwargs) if kwargs is not None else obj()
  raise ValueError(f"Could not find a Selector class in {py_file}")



def parse_args(args: list[str]) -> dict[str, Any]:
  if args is None:
    return {}
  kwargs = {}

  def try_bool(s: str) -> Tuple[bool, bool]:
    match s.lower():
      case "true" | "t" | "yes" | "y" | "on":
        return True, True
      case "false" | "f" | "no" | "n" | "off":
        return False, True
      case _:
        return False, False
    
  def try_int(s: str) -> Tuple[int, bool]:
    try:
      return int(s), True
    except ValueError:
      return 0, False

  def try_float(s: str) -> Tuple[float, bool]:
    try:
      return float(s), True
    except ValueError:
      return 0.0, False
    
  def try_float(s: str) -> Tuple[float, bool]:
    try:
      return float(s), True
    except ValueError:
      return 0.0, False
    
  def try_value(s: str) -> Any:
    for try_func in [try_bool, try_int, try_float]:
      v, ok = try_func(s)
      if ok:
        return v
    return s

  for arg in args:
    kv = arg.split(arg, 1)
    if len(kv) != 2:
      logger.warning(f'invalid arg "{arg}"')
      continue
    key, value = kv
    kwargs[key] = try_value(value)
  return kwargs

def main():
  parser = argparse.ArgumentParser(description="Run a selector")
  parser.add_argument("selector", help="Selector Python file")
  parser.add_argument("-a", "--arg", nargs='*', help="any init arguments with format key=value")
  parser.add_argument(
    "-m", "--msd", type=str, default=os.environ.get("MSD_HOST", "http://localhost:50510"), help="MSD host, env $MSD_HOST"
  )
  parser.add_argument("-s", "--symbols", nargs="+", default=[], help="Symbols to run")
  args = parser.parse_args()
  kwargs = parse_args(args.arg)
  selector = load_selector(args.selector, **kwargs)
  logger.info(f"selector loaded: {selector}")

  dp = MsdSelectorDataProvider(args.msd)
  logger.info("data provider created")

  init_symbols = expand_symbols(args.symbols)

  stocks = selector.execute(dp, init_symbols)
  logger.info("selector finished")
  factors = getattr(selector, 'factors', None)
  if factors is not None:
    print(factors)
  else:
    for stock in stocks:
      print(stock)

if __name__ == "__main__":
  main()
  






