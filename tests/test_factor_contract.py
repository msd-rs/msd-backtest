# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for Factor contract, AST validation, loading, and causality probes."""

from __future__ import annotations

import alpha as al
import numpy as np
import pytest

from mbt.factor import (
  Factor,
  check_factor_causality,
  check_factor_source,
  load_factor,
  normalize_factor_output,
)
from mbt.select.panel import Panel


def _make_panel(n_objs: int = 12, bars: int = 60, seed: int = 42) -> Panel:
  rng = np.random.default_rng(seed)
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + np.timedelta64(bars, "D"))
  rets = rng.normal(0.0005, 0.02, size=(n_objs, bars))
  close = 10.0 * np.cumprod(1.0 + rets, axis=1)
  open_ = close * (1.0 + rng.normal(0.0, 0.003, size=(n_objs, bars)))
  high = np.maximum(open_, close) * 1.01
  low = np.minimum(open_, close) * 0.99
  vol = rng.uniform(1e5, 1e6, size=(n_objs, bars))
  objs = [f"SH60{i:04d}" for i in range(n_objs)]
  return Panel(
    {
      "ts": np.tile(dates, (n_objs, 1)),
      "open": open_,
      "high": high,
      "low": low,
      "close": close,
      "raw_open": open_,
      "raw_high": high,
      "raw_low": low,
      "raw_close": close,
      "volume": vol,
      "amount": vol * close,
      "limited": np.zeros((n_objs, bars), dtype=np.int32),
      "f003": np.full((n_objs, bars), 5.0),
    },
    objs,
    bars,
  )


def test_check_factor_source_catches_lookahead_and_price_basis_mixing():
  bad_fret = """
import alpha as al
from mbt.factor import Factor
class F(Factor):
  def compute(self, panel):
    return panel.compute(al.FRET, "open", "close", "volume", 1, 5)
"""
  with pytest.raises(ValueError, match="LOOKAHEAD DETECTED"):
    check_factor_source(bad_fret, fail_fast=True)

  bad_neg_ref = """
import alpha as al
from mbt.factor import Factor
class F(Factor):
  def compute(self, panel):
    return panel.compute(al.REF, "close", -1)
"""
  with pytest.raises(ValueError, match="LOOKAHEAD DETECTED"):
    check_factor_source(bad_neg_ref, fail_fast=True)

  bad_valuation = """
from mbt.factor import Factor
class F(Factor):
  def compute(self, panel):
    c = panel.column("close")
    bvps = panel.column("f003")
    return bvps / c
"""
  with pytest.raises(ValueError, match="ANTI-PATTERN DETECTED"):
    check_factor_source(bad_valuation, fail_fast=True)

  good_valuation = """
from mbt.factor import Factor
class F(Factor):
  def compute(self, panel):
    rc = panel.column("raw_close")
    bvps = panel.column("f003")
    return bvps / rc
"""
  assert check_factor_source(good_valuation, fail_fast=True) == []


def test_causality_probe_passes_causal_and_rejects_lookahead():
  panel = _make_panel(n_objs=10, bars=50)

  class CausalMomentum(Factor):
    name = "causal_mom"
    def compute(self, p: Panel) -> np.ndarray:
      ref = p.compute(al.REF, "close", 5)
      return p.column("close") / ref - 1.0

  f_ok = CausalMomentum()
  out_ok = normalize_factor_output(f_ok.compute(panel), panel.n_objs, panel.bars, f_ok.name)
  res = check_factor_causality(f_ok, panel, out_ok, warmup_bars=5, fail_fast=True)
  assert res["status"] == "PASS"
  assert len(res["probes"]) == 2

  class SneakyLookahead(Factor):
    name = "sneaky_global_mean"
    def compute(self, p: Panel) -> np.ndarray:
      c = p.column("close")
      # Leaks future mean along time axis!
      return c - np.mean(c, axis=1, keepdims=True)

  f_bad = SneakyLookahead()
  out_bad = normalize_factor_output(f_bad.compute(panel), panel.n_objs, panel.bars, f_bad.name)
  with pytest.raises(ValueError, match="LOOKAHEAD BIAS DETECTED"):
    check_factor_causality(f_bad, panel, out_bad, warmup_bars=5, fail_fast=True)


def test_load_factor_and_orphan_kwargs_fail_fast(tmp_path):
  f_path = tmp_path / "my_fac.py"
  f_path.write_text(
    """
from mbt.factor import Factor
M_AUTHOR = "TestAuthor"
class SimpleFactor(Factor):
  def __init__(self, window: int = 10):
    self.window = int(window)
  def compute(self, panel):
    return panel.column("close")
""",
    encoding="utf-8",
  )

  inst = load_factor(str(f_path), {"window": 20})
  assert getattr(inst, "window") == 20
  assert getattr(inst, "_author") == "TestAuthor"

  with pytest.raises(ValueError, match="Unrecognized argument"):
    load_factor(str(f_path), {"windwo": 20})


def test_normalize_factor_output_validation():
  arr_1d = np.arange(12, dtype=np.float64)
  out = normalize_factor_output(arr_1d, 3, 4, "f1")
  assert out["f1"].shape == (3, 4)

  with pytest.raises(ValueError, match="shape mismatch"):
    normalize_factor_output(np.zeros((2, 4)), 3, 4, "f1")

  with pytest.raises(ValueError, match="empty dict"):
    normalize_factor_output({}, 3, 4, "f1")
