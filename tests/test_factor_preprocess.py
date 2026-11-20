# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for cross-sectional factor preprocessing (winsorize, neutralize, standardize)."""

from __future__ import annotations

import numpy as np
import pytest

from mbt.factor.preprocess import (
  compute_log_mktcap,
  neutralize,
  preprocess_factor,
  standardize,
  winsorize,
)
from mbt.select.panel import Panel


def test_winsorize_mad_sigma_percentile_and_zero_mad_guard():
  rng = np.random.default_rng(1)
  mat = rng.normal(0.0, 1.0, size=(30, 5))
  mat[0, :] = 100.0  # extreme upper outlier
  mat[1, :] = -100.0  # extreme lower outlier
  mat[2, 0] = np.nan

  w_mad = winsorize(mat, "mad", mad_k=3.0)
  assert np.isnan(w_mad[2, 0])
  assert np.nanmax(w_mad) < 10.0
  assert np.nanmin(w_mad) > -10.0

  w_sig = winsorize(mat, "sigma", sigma_k=2.0)
  assert np.nanmax(w_sig) < 60.0
  assert np.nanmin(w_sig) > -60.0

  w_pct = winsorize(mat, "percentile", pct_low=0.05, pct_high=0.95)
  assert np.nanmax(w_pct) < 10.0
  assert np.nanmin(w_pct) > -10.0

  # Zero-MAD cross-section (>50% zeros) must NOT turn finite values into NaN
  sparse = np.zeros((20, 2), dtype=np.float64)
  sparse[0, :] = 5.0
  w_sparse = winsorize(sparse, "mad")
  assert np.all(np.isfinite(w_sparse))
  assert w_sparse[0, 0] == 5.0


def test_standardize_zscore_and_rank():
  rng = np.random.default_rng(2)
  mat = rng.normal(10.0, 5.0, size=(25, 4))
  mat[0, 1] = np.nan

  z = standardize(mat, "zscore")
  assert np.isnan(z[0, 1])
  for t in range(4):
    col = z[:, t]
    fin = col[np.isfinite(col)]
    assert abs(float(np.mean(fin))) < 1e-6
    assert abs(float(np.std(fin)) - 1.0) < 0.05

  r = standardize(mat, "rank")
  assert np.isnan(r[0, 1])
  fin_r = r[np.isfinite(r)]
  assert np.min(fin_r) >= 0.0 and np.max(fin_r) <= 1.0


def test_neutralize_industry_mktcap_and_exact_fwl_joint_ols():
  rng = np.random.default_rng(3)
  n_objs, bars = 30, 8
  ind_codes = np.array([1] * 10 + [2] * 10 + [3] * 10, dtype=np.float64)
  log_cap = rng.normal(22.0, 1.5, size=(n_objs, bars))
  ind_effects = np.where(ind_codes == 1, 2.0, np.where(ind_codes == 2, -1.5, 0.5))[:, None]
  pure_alpha = rng.normal(0.0, 0.3, size=(n_objs, bars))
  raw = ind_effects + 0.8 * log_cap + pure_alpha
  # Inject NaNs at different places to test joint valid mask
  raw[0, 0] = np.nan
  log_cap[1, 0] = np.nan
  raw[15, 3] = np.nan

  # 1. Industry only -> zero mean within every industry per bar
  n_ind = neutralize(raw, modes=("industry",), industry_codes=ind_codes)
  for code in (1.0, 2.0, 3.0):
    sub = n_ind[ind_codes == code, :]
    for t in range(bars):
      col = sub[:, t]
      fin = col[np.isfinite(col)]
      assert abs(float(np.mean(fin))) < 1e-10

  # 2. Mktcap only -> zero correlation with log_cap per bar
  n_cap = neutralize(raw, modes=("mktcap",), log_mktcap=log_cap)
  for t in range(bars):
    ok = np.isfinite(n_cap[:, t]) & np.isfinite(log_cap[:, t])
    corr = float(np.corrcoef(n_cap[ok, t], log_cap[ok, t])[0, 1])
    assert abs(corr) < 1e-10

  # 3. Joint industry + mktcap via FWL must match explicit dummy-variable lstsq to machine precision
  n_joint = neutralize(raw, modes=("industry", "mktcap"), industry_codes=ind_codes, log_mktcap=log_cap)
  for t in range(bars):
    ok = np.isfinite(raw[:, t]) & np.isfinite(log_cap[:, t])
    y = raw[ok, t]
    X = np.column_stack([
      (ind_codes[ok] == 1).astype(np.float64),
      (ind_codes[ok] == 2).astype(np.float64),
      (ind_codes[ok] == 3).astype(np.float64),
      log_cap[ok, t],
    ])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    expected_resid = y - X @ coef
    assert np.allclose(n_joint[ok, t], expected_resid, atol=1e-10)
    assert np.all(np.isnan(n_joint[~ok, t]))


def test_compute_log_mktcap_uses_raw_close_not_adjusted_close():
  n_objs, bars = 4, 5
  close_adj = np.full((n_objs, bars), 10.0)
  raw_close = np.full((n_objs, bars), 25.0)
  shares = np.full((n_objs, bars), 1e8)
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-06"))
  panel = Panel(
    {
      "ts": np.tile(dates, (n_objs, 1)),
      "close": close_adj,
      "raw_close": raw_close,
      "total_shares": shares,
    },
    [f"SH60000{i}" for i in range(n_objs)],
    bars,
  )
  log_cap = compute_log_mktcap(panel)
  assert np.allclose(log_cap, np.log(25.0 * 1e8))

  with pytest.raises(ValueError, match="missing 'total_shares'"):
    compute_log_mktcap(
      Panel({"ts": np.tile(dates, (n_objs, 1)), "raw_close": raw_close}, panel.objs, bars)
    )


def test_preprocess_factor_pipeline_validation():
  mat = np.ones((5, 3), dtype=np.float64)
  with pytest.raises(ValueError, match="Invalid winsorize"):
    preprocess_factor(mat, winsorize_method="bad")
  with pytest.raises(ValueError, match="Invalid neutralize"):
    preprocess_factor(mat, neutralize_modes=("sector_unknown",))
  with pytest.raises(ValueError, match="Invalid standardize"):
    preprocess_factor(mat, standardize_method="minmax")
