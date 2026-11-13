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
)
from .financial import to_statutory_disclosure_date

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
  'STATUS_NORMAL',
  'STATUS_ST',
  'STATUS_STAR_ST',
  'get_limit_ratio',
  'get_industry_by_stocks',
  'get_stock_by_industries',
  'to_statutory_disclosure_date',
]


def __getattr__(name: str):
  if name in ('ALL_STOCKS', 'A_STOCKS', 'A_STOCKS_EXCLUDE_ST', 'FOUNDS'):
    return getattr(stocks, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")