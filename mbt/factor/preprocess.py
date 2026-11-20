# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Cross-sectional factor preprocessing: winsorization, neutralization, and standardization."""

from __future__ import annotations

from typing import Sequence

import alpha as al
import numpy as np

from mbt.select.panel import Panel

VALID_WINSORIZE = ("none", "mad", "sigma", "percentile")
VALID_STANDARDIZE = ("none", "zscore", "rank")
VALID_NEUTRALIZE = ("industry", "mktcap")


def compute_log_mktcap(panel: Panel) -> np.ndarray:
  """Compute ``log(raw_close * total_shares)`` of shape ``(n_objs, bars)``.

  Strictly uses unadjusted ``raw_close`` (never forward-adjusted ``close``) to
  preserve true point-in-time market capitalization semantics.
  """
  if "raw_close" in panel.columns:
    raw_close = np.asarray(panel.column("raw_close"), dtype=np.float64)
  elif "price" in panel.columns:
    raw_close = np.asarray(panel.column("price"), dtype=np.float64)
  else:
    raise ValueError(
      "Cannot compute market cap: Panel missing unadjusted price column ('raw_close' or 'price')."
    )

  if "total_shares" not in panel.columns:
    raise ValueError(
      "Cannot compute market cap: Panel missing 'total_shares' column."
    )
  shares = np.asarray(panel.column("total_shares"), dtype=np.float64)
  with np.errstate(divide="ignore", invalid="ignore"):
    cap = raw_close * shares
    ok = np.isfinite(cap) & (cap > 0.0)
    log_cap = np.where(ok, np.log(np.where(ok, cap, 1.0)), np.nan)
  return log_cap


def winsorize(
  mat: np.ndarray,
  method: str = "none",
  *,
  mad_k: float = 3.0,
  sigma_k: float = 3.0,
  pct_low: float = 0.01,
  pct_high: float = 0.99,
) -> np.ndarray:
  """Cross-sectionally winsorize a ``(n_objs, bars)`` factor matrix per bar.

  Args:
    mat: 2D factor matrix of shape ``(n_objs, bars)``.
    method: One of ``"none"``, ``"mad"``, ``"sigma"``, ``"percentile"``.
    mad_k: MAD multiplier (scaled by 1.4826 for normal consistency).
    sigma_k: Standard deviation multiplier for ``"sigma"``.
    pct_low: Lower quantile in ``(0, 1)`` for ``"percentile"``.
    pct_high: Upper quantile in ``(0, 1)`` for ``"percentile"``.

  Returns:
    New ``float64`` array of shape ``(n_objs, bars)``.
  """
  m = method.strip().lower()
  if m not in VALID_WINSORIZE:
    raise ValueError(f"Invalid winsorize method {method!r}. Supported: {VALID_WINSORIZE}.")

  out = np.array(mat, dtype=np.float64, copy=True)
  if m == "none" or out.size == 0:
    return out

  fin = np.isfinite(out)
  has_any = np.any(fin, axis=0)
  if not np.any(has_any):
    return out

  if m == "mad":
    valid_cols = np.flatnonzero(has_any)
    sub_valid = out[:, valid_cols]
    with np.errstate(all="ignore"):
      med = np.nanmedian(sub_valid, axis=0)
      abs_dev = np.abs(sub_valid - med[None, :])
      mad = np.nanmedian(abs_dev, axis=0)
    scaled = 1.4826 * mad * float(mad_k)
    # Guard zero-MAD cross-sections (e.g. >50% identical values) against collapsing signal
    active_local = np.isfinite(med) & np.isfinite(scaled) & (scaled > 1e-12)
    if np.any(active_local):
      target_cols = valid_cols[active_local]
      lo = med[active_local] - scaled[active_local]
      hi = med[active_local] + scaled[active_local]
      sub = out[:, target_cols]
      out[:, target_cols] = np.where(
        np.isfinite(sub), np.clip(sub, lo[None, :], hi[None, :]), np.nan
      )
    return out

  if m == "sigma":
    cnt = np.sum(fin, axis=0)
    sums = np.sum(np.where(fin, out, 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
      mean = np.where(cnt > 0, sums / cnt, np.nan)
      diff = np.where(fin, out - mean[None, :], 0.0)
      var = np.where(cnt > 1, np.sum(diff * diff, axis=0) / cnt, np.nan)
      std = np.sqrt(var)
    active = np.isfinite(mean) & np.isfinite(std) & (std > 1e-12)
    if np.any(active):
      lo = mean[active] - float(sigma_k) * std[active]
      hi = mean[active] + float(sigma_k) * std[active]
      sub = out[:, active]
      out[:, active] = np.where(np.isfinite(sub), np.clip(sub, lo[None, :], hi[None, :]), np.nan)
    return out

  # percentile
  if not (0.0 <= pct_low < pct_high <= 1.0):
    raise ValueError(f"Invalid percentile bounds: pct_low={pct_low}, pct_high={pct_high}")
  valid_cols = np.flatnonzero(has_any)
  with np.errstate(all="ignore"):
    lo_vals = np.nanpercentile(out[:, valid_cols], pct_low * 100.0, axis=0)
    hi_vals = np.nanpercentile(out[:, valid_cols], pct_high * 100.0, axis=0)
  sub = out[:, valid_cols]
  out[:, valid_cols] = np.where(
    np.isfinite(sub),
    np.clip(sub, lo_vals[None, :], hi_vals[None, :]),
    np.nan,
  )
  return out


def _demean_by_industry(mat: np.ndarray, ind_codes: np.ndarray) -> np.ndarray:
  """Subtract cross-sectional industry mean per bar, vectorized across all bars."""
  out = np.full_like(mat, np.nan, dtype=np.float64)
  valid_ind = np.isfinite(ind_codes)
  unique_inds = np.unique(ind_codes[valid_ind])
  for code in unique_inds:
    rows = valid_ind & (ind_codes == code)
    sub = mat[rows, :]
    fin = np.isfinite(sub)
    cnt = np.sum(fin, axis=0)
    sums = np.sum(np.where(fin, sub, 0.0), axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
      mean = np.where(cnt > 0, sums / cnt, np.nan)
    out[rows, :] = np.where(fin, sub - mean[None, :], np.nan)
  return out


def _ols_residual_1d(y: np.ndarray, x: np.ndarray, center: bool = True) -> np.ndarray:
  """Vectorized cross-sectional OLS residual of ``y`` on ``x`` across all bars."""
  valid = np.isfinite(y) & np.isfinite(x)
  cnt = np.sum(valid, axis=0)
  y_v = np.where(valid, y, 0.0)
  x_v = np.where(valid, x, 0.0)

  if center:
    with np.errstate(divide="ignore", invalid="ignore"):
      y_mean = np.where(cnt > 0, np.sum(y_v, axis=0) / cnt, np.nan)
      x_mean = np.where(cnt > 0, np.sum(x_v, axis=0) / cnt, np.nan)
    yc = np.where(valid, y - y_mean[None, :], 0.0)
    xc = np.where(valid, x - x_mean[None, :], 0.0)
  else:
    yc = y_v
    xc = x_v

  sxx = np.sum(xc * xc, axis=0)
  sxy = np.sum(xc * yc, axis=0)
  with np.errstate(divide="ignore", invalid="ignore"):
    beta = np.where((cnt >= 2) & (sxx > 1e-12), sxy / sxx, 0.0)

  resid = yc - beta[None, :] * xc
  return np.where(valid, resid, np.nan)


def neutralize(
  mat: np.ndarray,
  *,
  modes: Sequence[str] = (),
  industry_codes: np.ndarray | None = None,
  log_mktcap: np.ndarray | None = None,
) -> np.ndarray:
  """Cross-sectionally neutralize factor exposures against industry and/or log market cap.

  Joint ``("industry", "mktcap")`` neutralization uses the exact Frisch-Waugh-Lovell
  (FWL) projection: both factor and log market cap are demeaned within each industry
  on the common valid mask, followed by cross-sectional OLS residualization.

  Args:
    mat: 2D factor matrix of shape ``(n_objs, bars)``.
    modes: Subset of ``("industry", "mktcap")``.
    industry_codes: 1D numeric industry code array of shape ``(n_objs,)``.
    log_mktcap: 2D log market cap array of shape ``(n_objs, bars)``.

  Returns:
    Neutralized ``float64`` matrix of shape ``(n_objs, bars)``.
  """
  norm_modes = tuple(dict.fromkeys(m.strip().lower() for m in modes if m and m.strip().lower() != "none"))
  for m in norm_modes:
    if m not in VALID_NEUTRALIZE:
      raise ValueError(f"Invalid neutralize mode {m!r}. Supported: {VALID_NEUTRALIZE}.")

  if not norm_modes:
    return np.array(mat, dtype=np.float64, copy=True)

  n_objs, bars = mat.shape
  use_ind = "industry" in norm_modes
  use_cap = "mktcap" in norm_modes

  ind_arr: np.ndarray | None = None
  if use_ind:
    if industry_codes is None:
      raise ValueError(
        "Industry neutralization requested, but no industry_codes were provided."
      )
    ind_arr = np.asarray(industry_codes, dtype=np.float64)
    if ind_arr.shape != (n_objs,):
      raise ValueError(
        f"industry_codes shape mismatch: expected ({n_objs},), got {ind_arr.shape}."
      )

  cap_arr: np.ndarray | None = None
  if use_cap:
    if log_mktcap is None:
      raise ValueError(
        "Market-cap neutralization requested, but log_mktcap was not provided."
      )
    cap_arr = np.asarray(log_mktcap, dtype=np.float64)
    if cap_arr.shape != (n_objs, bars):
      raise ValueError(
        f"log_mktcap shape mismatch: expected ({n_objs}, {bars}), got {cap_arr.shape}."
      )

  if use_ind and not use_cap:
    assert ind_arr is not None
    return _demean_by_industry(np.asarray(mat, dtype=np.float64), ind_arr)

  if use_cap and not use_ind:
    assert cap_arr is not None
    return _ols_residual_1d(np.asarray(mat, dtype=np.float64), cap_arr, center=True)

  # Exact joint OLS via Frisch-Waugh-Lovell on common valid mask
  assert ind_arr is not None and cap_arr is not None
  y = np.asarray(mat, dtype=np.float64)
  valid = np.isfinite(y) & np.isfinite(cap_arr) & np.isfinite(ind_arr)[:, None]
  y_masked = np.where(valid, y, np.nan)
  x_masked = np.where(valid, cap_arr, np.nan)
  y_ind = _demean_by_industry(y_masked, ind_arr)
  x_ind = _demean_by_industry(x_masked, ind_arr)
  return _ols_residual_1d(y_ind, x_ind, center=False)


def standardize(mat: np.ndarray, method: str = "none") -> np.ndarray:
  """Cross-sectionally standardize a ``(n_objs, bars)`` factor matrix per bar.

  Args:
    mat: 2D factor matrix of shape ``(n_objs, bars)``.
    method: One of ``"none"``, ``"zscore"``, ``"rank"``.

  Returns:
    Standardized ``float64`` matrix of shape ``(n_objs, bars)``.
  """
  m = method.strip().lower()
  if m not in VALID_STANDARDIZE:
    raise ValueError(f"Invalid standardize method {method!r}. Supported: {VALID_STANDARDIZE}.")

  if m == "none" or mat.size == 0:
    return np.array(mat, dtype=np.float64, copy=True)

  n_objs, bars = mat.shape
  flat = np.ascontiguousarray(mat, dtype=np.float64).reshape(-1)
  with al.Context(groups=n_objs, flags=al.FLAG_SKIP_NAN, start=0, end=bars):
    if m == "zscore":
      out = al.CC_ZSCORE(flat)
    else:
      out = al.CC_RANK(flat)
  return np.asarray(out, dtype=np.float64).reshape(n_objs, bars)


def preprocess_factor(
  mat: np.ndarray,
  *,
  signal_mask: np.ndarray | None = None,
  winsorize_method: str = "none",
  neutralize_modes: Sequence[str] = (),
  standardize_method: str = "none",
  industry_codes: np.ndarray | None = None,
  log_mktcap: np.ndarray | None = None,
) -> np.ndarray:
  """Apply the complete cross-sectional preprocessing pipeline in deterministic order:

  ``mask -> winsorize -> neutralize -> standardize``.
  """
  arr = np.array(mat, dtype=np.float64, copy=True)
  if signal_mask is not None:
    arr = np.where(signal_mask, arr, np.nan)
  arr = winsorize(arr, method=winsorize_method)
  if neutralize_modes:
    arr = neutralize(
      arr,
      modes=neutralize_modes,
      industry_codes=industry_codes,
      log_mktcap=log_mktcap,
    )
  arr = standardize(arr, method=standardize_method)
  return arr
