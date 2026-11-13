# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

import polars as pl


def to_statutory_disclosure_date(ts_series: pl.Series) -> pl.Series:
  """Convert report period dates to A-share statutory disclosure deadlines.

  Statutory Deadlines:
  - Q1 (month == 3): year-04-30
  - Semi-annual / Q2 (month == 6): year-08-31
  - Q3 (month == 9): year-10-31
  - Annual / Q4 (month == 12): (year+1)-04-30
  - Non-standard: end of month (28th fallback)
  """
  y = ts_series.dt.year()
  m = ts_series.dt.month()
  df = pl.DataFrame({"report_ts": ts_series})
  ann = df.select(
    pl.when(m == 3).then(
      pl.date(y, 4, 30)
    ).when(m == 6).then(
      pl.date(y, 8, 31)
    ).when(m == 9).then(
      pl.date(y, 10, 31)
    ).when(m == 12).then(
      pl.date(y + 1, 4, 30)
    ).otherwise(
      pl.date(y, m, 28)
    ).cast(pl.Datetime("us")).alias("ts")
  )["ts"]
  return ann
