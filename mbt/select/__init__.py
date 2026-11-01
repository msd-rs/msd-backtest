from typing import TYPE_CHECKING
from .selector import Selector, SelectorDataProvider
from .panel import Panel
from .msd import MsdSelectorDataProvider
from mbt.shared import stocks

if TYPE_CHECKING:
  from mbt.shared.stocks import ALL_STOCKS, A_STOCKS, A_STOCKS_EXCLUDE_ST, FOUNDS

__all__ = [
  'Selector',
  'SelectorDataProvider',
  'MsdSelectorDataProvider',
  'Panel',
  'stocks',
  'ALL_STOCKS',
  'A_STOCKS',
  'A_STOCKS_EXCLUDE_ST',
  'FOUNDS',
]


def __getattr__(name: str):
  if name in ('ALL_STOCKS', 'A_STOCKS', 'A_STOCKS_EXCLUDE_ST', 'FOUNDS'):
    return getattr(stocks, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

