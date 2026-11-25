# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause


from typing import TYPE_CHECKING
from . import stocks
from .stocks import (
  KIND_INDEX,
  KIND_FUND,
  KIND_STOCK,
  STATUS_NORMAL,
  STATUS_ST,
  STATUS_STAR_ST,
  get_limit_ratio,
  get_industry_by_stocks,
  get_stock_by_industries,
  get_stock_industry_map,
)
from .adjust import apply_price_adjustment
from .financial import to_statutory_disclosure_date
from .downsample import minmax_downsample
from .inspect_strategy import (
  ParamSpec,
  RUN_MODE_FACTOR,
  RUN_MODE_PORTFOLIO,
  RUN_MODE_SELECT,
  RUN_MODE_TIMING,
  StrategyMetadata,
  VALID_RUN_MODES,
  inspect_strategy_file,
)

if TYPE_CHECKING:
  from .stocks import ALL_STOCKS, A_STOCKS, A_STOCKS_EXCLUDE_ST, FOUNDS


__all__ = [
  'ALL_STOCKS',
  'A_STOCKS',
  'A_STOCKS_EXCLUDE_ST',
  'FOUNDS',
  'KIND_INDEX',
  'KIND_FUND',
  'KIND_STOCK',
  'ParamSpec',
  'RUN_MODE_FACTOR',
  'RUN_MODE_PORTFOLIO',
  'RUN_MODE_SELECT',
  'RUN_MODE_TIMING',
  'STATUS_NORMAL',
  'STATUS_ST',
  'STATUS_STAR_ST',
  'StrategyMetadata',
  'VALID_RUN_MODES',
  'apply_price_adjustment',
  'get_limit_ratio',
  'get_industry_by_stocks',
  'get_stock_by_industries',
  'get_stock_industry_map',
  'inspect_strategy_file',
  'minmax_downsample',
  'to_statutory_disclosure_date',
]


def __getattr__(name: str):
  if name in ('ALL_STOCKS', 'A_STOCKS', 'A_STOCKS_EXCLUDE_ST', 'FOUNDS'):
    return getattr(stocks, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
