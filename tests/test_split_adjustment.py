# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Tests for geometric factor-based corporate-action price adjustment."""

import numpy as np
import pytest

from mbt.factor.expr import ExpressionFactor
from mbt.factor.factor import check_factor_source
from mbt.factor.returns import build_tradability_masks
from mbt.select.panel import Panel
from mbt.shared import apply_price_adjustment


def _make_market_dict(
  raw_close: np.ndarray,
  dividend: np.ndarray,
  transfer_shares: np.ndarray,
  right_shares: np.ndarray,
  right_price: np.ndarray,
  *,
  raw_open: np.ndarray | None = None,
  raw_high: np.ndarray | None = None,
  raw_low: np.ndarray | None = None,
  volume: np.ndarray | None = None,
  amount: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
  if raw_open is None:
    raw_open = raw_close * 0.99
  if raw_high is None:
    raw_high = raw_close * 1.02
  if raw_low is None:
    raw_low = raw_close * 0.98
  if volume is None:
    volume = np.where(raw_close > 0.0, 1000.0, 0.0)
  if amount is None:
    amount = np.where(volume > 0.0, volume * raw_close * 1.005, np.nan)
  return {
    "open": raw_open.copy(),
    "high": raw_high.copy(),
    "low": raw_low.copy(),
    "close": raw_close.copy(),
    "volume": volume.copy(),
    "amount": amount.copy(),
    "dividend": dividend.copy(),
    "transfer_shares": transfer_shares.copy(),
    "right_shares": right_shares.copy(),
    "right_price": right_price.copy(),
  }


def test_long_horizon_high_dividend_remains_strictly_positive():
  # Simulate a high-dividend stock starting at 30.0 with cumulative cash dividends > 100.0 and splits
  n = 30
  raw_close = np.full(n, 30.0, dtype=np.float64)
  dividend = np.zeros(n, dtype=np.float64)
  transfer_shares = np.zeros(n, dtype=np.float64)
  right_shares = np.zeros(n, dtype=np.float64)
  right_price = np.zeros(n, dtype=np.float64)

  # 12 annual dividends of 10.0 each (total 120.0 > initial 30.0) + two 10-for-10 splits
  for idx in range(2, n, 2):
    dividend[idx] = 10.0
  transfer_shares[5] = 1.0
  transfer_shares[15] = 1.0

  data = _make_market_dict(raw_close, dividend, transfer_shares, right_shares, right_price)
  apply_price_adjustment(data, groups=1)

  for col in ("open", "high", "low", "close", "vwap", "bw_price", "fw_price"):
    assert np.all(np.isfinite(data[col]))
    assert np.all(data[col] > 0.0), f"Column {col} became non-positive"

  assert np.array_equal(data["close"], data["bw_price"])
  assert np.all(data["bw_factor"] >= 1.0)
  assert np.all((data["fw_factor"] > 0.0) & (data["fw_factor"] <= 1.0))


def test_pre_ipo_zeros_preserved_without_negative_backfill():
  # Regression for SH688001: pre-IPO bars have raw_close == 0.0, volume == 0.0, amount == NaN
  raw_close = np.array([0.0, 0.0, 0.0, 20.0, 22.0, 21.0, 23.0], dtype=np.float64)
  raw_open = np.array([0.0, 0.0, 0.0, 19.5, 21.0, 22.0, 22.5], dtype=np.float64)
  raw_high = np.array([0.0, 0.0, 0.0, 20.5, 22.5, 22.2, 23.5], dtype=np.float64)
  raw_low = np.array([0.0, 0.0, 0.0, 19.0, 20.5, 20.8, 22.0], dtype=np.float64)
  volume = np.array([0.0, 0.0, 0.0, 1000.0, 1200.0, 900.0, 1100.0], dtype=np.float64)
  amount = np.array([np.nan, np.nan, np.nan, 20000.0, 26000.0, 19000.0, 25000.0], dtype=np.float64)
  dividend = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.5], dtype=np.float64)
  transfer_shares = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0], dtype=np.float64)
  right_shares = np.zeros(7, dtype=np.float64)
  right_price = np.zeros(7, dtype=np.float64)

  data = _make_market_dict(
    raw_close,
    dividend,
    transfer_shares,
    right_shares,
    right_price,
    raw_open=raw_open,
    raw_high=raw_high,
    raw_low=raw_low,
    volume=volume,
    amount=amount,
  )
  apply_price_adjustment(data, groups=1)

  for col in ("open", "high", "low", "close", "bw_price", "fw_price"):
    assert np.all(data[col][:3] == 0.0), f"Pre-IPO {col} should remain 0.0, got {data[col][:3]}"
    assert np.all(data[col][3:] > 0.0)

  panel = Panel(data, objs=["SH688001"], bars=7)
  signal_mask, buy_mask, _, _ = build_tradability_masks(panel)
  assert not np.any(signal_mask[0, :3])
  assert not np.any(buy_mask[0, :3])
  assert np.all(signal_mask[0, 3:])


def test_total_return_and_ohlc_ratio_invariance():
  # Bar 1: split (10送10), Bar 2: cash dividend, Bar 3: rights issue, Bar 4: normal trading day
  raw_close = np.array([20.0, 11.0, 10.0, 9.6, 10.56], dtype=np.float64)
  raw_open = np.array([19.8, 10.5, 10.2, 9.5, 10.0], dtype=np.float64)
  raw_high = np.array([20.4, 11.2, 10.5, 9.9, 10.8], dtype=np.float64)
  raw_low = np.array([19.5, 10.2, 9.8, 9.2, 9.9], dtype=np.float64)
  volume = np.array([1000.0, 2000.0, 1500.0, 1800.0, 1600.0], dtype=np.float64)
  amount = volume * np.array([20.1, 10.8, 10.1, 9.55, 10.3], dtype=np.float64)

  dividend = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
  transfer_shares = np.array([0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
  right_shares = np.array([0.0, 0.0, 0.0, 0.2, 0.0], dtype=np.float64)
  right_price = np.array([0.0, 0.0, 0.0, 5.0, 0.0], dtype=np.float64)

  data = _make_market_dict(
    raw_close,
    dividend,
    transfer_shares,
    right_shares,
    right_price,
    raw_open=raw_open,
    raw_high=raw_high,
    raw_low=raw_low,
    volume=volume,
    amount=amount,
  )
  apply_price_adjustment(data, groups=1)

  # Exact ex-action total return: close[t] / close[t-1] == raw_close[t] / ex_ref_price[t]
  for t in (1, 2, 3):
    denom = 1.0 + transfer_shares[t] + right_shares[t]
    ex_ref = (raw_close[t - 1] - dividend[t] + right_shares[t] * right_price[t]) / denom
    expected_ret = raw_close[t] / ex_ref
    actual_ret = data["close"][t] / data["close"][t - 1]
    assert np.isclose(actual_ret, expected_ret, rtol=1e-12)

  # Post-action normal bar return matches raw return exactly
  assert np.isclose(data["close"][4] / data["close"][3], raw_close[4] / raw_close[3], rtol=1e-12)

  # Intraday OHLC & VWAP ratios and ordering are preserved identically
  assert np.allclose(data["open"] / data["close"], raw_open / raw_close, rtol=1e-12)
  assert np.allclose(data["high"] / data["low"], raw_high / raw_low, rtol=1e-12)
  assert np.allclose(data["vwap"] / data["close"], data["raw_vwap"] / raw_close, rtol=1e-12)
  assert np.all(data["low"] <= data["open"])
  assert np.all(data["low"] <= data["close"])
  assert np.all(data["open"] <= data["high"])
  assert np.all(data["close"] <= data["high"])


def test_strict_causality_under_future_truncation():
  raw_close = np.array([10.0, 11.0, 10.5, 12.0, 11.0, 13.0, 12.5, 14.0], dtype=np.float64)
  dividend = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 1.0, 0.0, 2.0], dtype=np.float64)
  transfer_shares = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 1.0, 0.0], dtype=np.float64)
  right_shares = np.zeros(8, dtype=np.float64)
  right_price = np.zeros(8, dtype=np.float64)

  t1 = 5
  d_prefix = _make_market_dict(
    raw_close[:t1],
    dividend[:t1],
    transfer_shares[:t1],
    right_shares[:t1],
    right_price[:t1],
  )
  d_full = _make_market_dict(
    raw_close,
    dividend,
    transfer_shares,
    right_shares,
    right_price,
  )
  apply_price_adjustment(d_prefix, groups=1)
  apply_price_adjustment(d_full, groups=1)

  for col in ("open", "high", "low", "close", "vwap", "bw_price", "bw_factor"):
    assert np.array_equal(d_prefix[col], d_full[col][:t1]), f"{col} changed under future truncation!"


def test_multi_group_isolation():
  bars = 4
  raw_close = np.array([10.0, 10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 23.0], dtype=np.float64)
  dividend = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
  transfer_shares = np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
  right_shares = np.zeros(2 * bars, dtype=np.float64)
  right_price = np.zeros(2 * bars, dtype=np.float64)

  data = _make_market_dict(raw_close, dividend, transfer_shares, right_shares, right_price)
  apply_price_adjustment(data, groups=2)

  # Group 0 was adjusted
  assert np.all(data["bw_factor"][bars - 1] > 1.0)
  # Group 1 had no corporate actions: factor stays 1.0 and adjusted prices equal raw prices
  assert np.all(data["bw_factor"][bars:] == 1.0)
  assert np.all(data["fw_factor"][bars:] == 1.0)
  assert np.array_equal(data["close"][bars:], raw_close[bars:])
  assert np.array_equal(data["open"][bars:], data["raw_open"][bars:])


def test_factor_expr_and_tradability_guardrails():
  raw_close = np.array([10.0, 10.0, 11.0, 12.0], dtype=np.float64)
  dividend = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
  transfer_shares = np.zeros(4, dtype=np.float64)
  right_shares = np.zeros(4, dtype=np.float64)
  right_price = np.zeros(4, dtype=np.float64)

  data = _make_market_dict(raw_close, dividend, transfer_shares, right_shares, right_price)
  apply_price_adjustment(data, groups=1)
  data["f003"] = np.full(4, 5.0, dtype=np.float64)
  panel = Panel(data, objs=["SH600000"], bars=4)

  # Factor expressions can inspect bw_factor, fw_factor, and adj_factor alias
  for expr in ("bw_factor", "fw_factor", "adj_factor"):
    fac = ExpressionFactor(expr, name=f"test_{expr}")
    out = np.asarray(fac.compute(panel)).reshape(-1)
    expected = data["bw_factor"] if expr == "adj_factor" else data[expr]
    assert np.allclose(out, expected)

  # Anti-pattern AST guard catches division of per-share financials with bw_price / fw_price
  for bad_expr in ("f003 / bw_price", "f003 / fw_price"):
    with pytest.raises(ValueError, match="ANTI-PATTERN DETECTED"):
      ExpressionFactor(bad_expr)

  bad_src = """
from mbt.factor import Factor
class BadFactor(Factor):
  def compute(self, panel):
    p = panel.column("bw_price")
    bps = panel.column("f003")
    return bps / p
"""
  with pytest.raises(ValueError, match="ANTI-PATTERN DETECTED"):
    check_factor_source(bad_src, fail_fast=True)

  # build_tradability_masks rejects raw_close == 0.0 even if close were artificially non-zero
  panel_data = dict(data)
  panel_data["raw_close"] = np.array([0.0, 10.0, 11.0, 12.0], dtype=np.float64)
  panel_data["close"] = np.array([5.0, 10.0, 11.0, 12.0], dtype=np.float64)
  p2 = Panel(panel_data, objs=["SH600000"], bars=4)
  sig_mask, buy_mask, _, _ = build_tradability_masks(p2)
  assert sig_mask[0, 0] is np.bool_(False)
  assert buy_mask[0, 0] is np.bool_(False)
  assert np.all(sig_mask[0, 1:])
