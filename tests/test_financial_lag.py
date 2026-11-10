import numpy as np
import polars as pl
from mbt.run.msd import to_statutory_disclosure_date


def test_statutory_disclosure_date_conversion():
  report_dates = np.array(
    [
      "2023-03-31T00:00:00.000000",  # Q1 -> 2023-04-30
      "2023-06-30T00:00:00.000000",  # Q2 -> 2023-08-31
      "2023-09-30T00:00:00.000000",  # Q3 -> 2023-10-31
      "2023-12-31T00:00:00.000000",  # Annual -> 2024-04-30
    ],
    dtype="datetime64[us]",
  )
  s = pl.Series("report_ts", report_dates)
  stat_dates = to_statutory_disclosure_date(s).to_numpy()

  expected = np.array(
    [
      "2023-04-30T00:00:00.000000",
      "2023-08-31T00:00:00.000000",
      "2023-10-31T00:00:00.000000",
      "2024-04-30T00:00:00.000000",
    ],
    dtype="datetime64[us]",
  )
  assert np.array_equal(stat_dates, expected)


def test_financial_alignment_no_lookahead():
  # Simulating quarterly financial announcements
  ann_ts = np.array(
    [
      "2023-10-31T00:00:00.000000",  # 2023 Q3 disclosed by 2023-10-31
      "2024-04-30T00:00:00.000000",  # 2023 Annual disclosed by 2024-04-30
    ],
    dtype="datetime64[us]",
  )
  report_ts = np.array(
    [
      "2023-09-30T00:00:00.000000",
      "2023-12-31T00:00:00.000000",
    ],
    dtype="datetime64[us]",
  )
  roe_vals = np.array([6.28, 7.50])

  # Daily bars spanning from Feb 2024 to May 2024
  kline_ts = np.array(
    [
      "2024-02-15T00:00:00.000000",
      "2024-04-29T00:00:00.000000",
      "2024-04-30T00:00:00.000000",
      "2024-05-02T00:00:00.000000",
    ],
    dtype="datetime64[us]",
  )

  idx = np.searchsorted(ann_ts, kline_ts, side="right") - 1
  valid = idx >= 0
  safe_idx = np.maximum(idx, 0)
  aligned_roe = np.where(valid, roe_vals[safe_idx], np.nan)
  aligned_rep_ts = np.where(valid, report_ts[safe_idx], np.datetime64("NaT", "us"))

  # In Feb 2024 and 2024-04-29 (before annual report disclosure on 04-30):
  # Must see 2023-09-30 Q3 data (ROE 6.28), NOT 2023-12-31!
  assert aligned_rep_ts[0] == np.datetime64("2023-09-30T00:00:00.000000", "us")
  assert np.isclose(aligned_roe[0], 6.28)
  assert aligned_rep_ts[1] == np.datetime64("2023-09-30T00:00:00.000000", "us")
  assert np.isclose(aligned_roe[1], 6.28)

  # On and after 2024-04-30:
  # Sees 2023-12-31 annual report (ROE 7.50)
  assert aligned_rep_ts[2] == np.datetime64("2023-12-31T00:00:00.000000", "us")
  assert np.isclose(aligned_roe[2], 7.50)
  assert aligned_rep_ts[3] == np.datetime64("2023-12-31T00:00:00.000000", "us")
  assert np.isclose(aligned_roe[3], 7.50)
