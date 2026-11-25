# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Geometric corporate-action price adjustment helpers."""

from __future__ import annotations

import alpha as al
import numpy as np


def apply_price_adjustment(data: dict[str, np.ndarray], groups: int) -> None:
  """Apply geometric backward/forward split and dividend factors in-place.

  Preserves unadjusted ``raw_open``, ``raw_high``, ``raw_low``, ``raw_close``,
  and ``price``, computes geometric ``bw_factor`` and ``fw_factor`` once from
  ``raw_close``, and populates ``open``, ``high``, ``low``, ``close``,
  ``bw_price``, ``fw_price`` (plus ``raw_vwap`` and ``vwap`` when volume and
  amount are available).

  ``open``, ``high``, ``low``, ``close``, and ``vwap`` use backward geometric
  adjustment (``raw * bw_factor``), ensuring strictly positive adjusted prices,
  zero pre-IPO preservation, exact total-return / OHLC ratio invariance, and
  strict causality without lookahead bias.

  Args:
    data: Flat obj-major market data dictionary containing at least ``open``,
      ``high``, ``low``, ``close``, ``dividend``, ``transfer_shares``,
      ``right_shares``, and ``right_price``.
    groups: Number of stock groups in ``data``.
  """
  if "raw_open" not in data:
    data["raw_open"] = data["open"].copy()
  if "raw_high" not in data:
    data["raw_high"] = data["high"].copy()
  if "raw_low" not in data:
    data["raw_low"] = data["low"].copy()
  if "raw_close" not in data:
    data["raw_close"] = data["close"].copy()
  if "price" not in data:
    data["price"] = data["raw_close"].copy()

  raw_open = data["raw_open"]
  raw_high = data["raw_high"]
  raw_low = data["raw_low"]
  raw_close = data["raw_close"]

  with al.Context(groups=groups, flags=al.FLAG_SKIP_NAN):
    bw_factor = al.BW_SPLIT_FACTOR(
      raw_close,
      data["dividend"],
      data["transfer_shares"],
      data["right_shares"],
      data["right_price"],
    )
    fw_factor = al.FW_SPLIT_FACTOR(
      raw_close,
      data["dividend"],
      data["transfer_shares"],
      data["right_shares"],
      data["right_price"],
    )

  data["bw_factor"] = bw_factor
  data["fw_factor"] = fw_factor
  data["bw_price"] = raw_close * bw_factor
  data["fw_price"] = raw_close * fw_factor
  data["open"] = raw_open * bw_factor
  data["high"] = raw_high * bw_factor
  data["low"] = raw_low * bw_factor
  data["close"] = data["bw_price"]

  if "volume" in data and "amount" in data:
    with np.errstate(divide="ignore", invalid="ignore"):
      raw_vwap = np.where(data["volume"] > 0, data["amount"] / data["volume"], np.nan)
    data["raw_vwap"] = raw_vwap
    data["vwap"] = raw_vwap * bw_factor
