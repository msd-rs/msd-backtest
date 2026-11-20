# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Unit tests for alpha.lang expression transpilation, PanelExecContext, and dialect compatibility."""

from __future__ import annotations

import json
import numpy as np
import pytest

from mbt.factor import (
  ExpressionFactor,
  PanelExecContext,
  check_factor_causality,
  compile_expression,
  format_operator_catalog,
  load_expression_file,
  normalize_factor_output,
  parse_inline_expressions,
)
from mbt.select.panel import Panel


def _make_rich_panel(n_objs: int = 12, bars: int = 90, seed: int = 7) -> Panel:
  rng = np.random.default_rng(seed)
  dates = np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + np.timedelta64(bars, "D"))
  rets = rng.normal(0.0004, 0.018, size=(n_objs, bars))
  close = 20.0 * np.cumprod(1.0 + rets, axis=1)
  open_ = close * (1.0 + rng.normal(0.0, 0.004, size=(n_objs, bars)))
  high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.015, size=(n_objs, bars)))
  low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.015, size=(n_objs, bars)))
  vol = rng.uniform(1e5, 5e6, size=(n_objs, bars))
  amount = vol * close * 1.05
  vwap = amount / vol
  raw_close = close * 2.0  # intentionally distinct from forward-adjusted close
  total_shares = rng.uniform(1e8, 5e9, size=(n_objs, 1)) * np.ones((1, bars))
  objs = [f"SH60{i:04d}" for i in range(n_objs)]
  return Panel(
    {
      "ts": np.tile(dates, (n_objs, 1)),
      "open": open_,
      "high": high,
      "low": low,
      "close": close,
      "vwap": vwap,
      "raw_open": open_ * 2.0,
      "raw_high": high * 2.0,
      "raw_low": low * 2.0,
      "raw_close": raw_close,
      "raw_vwap": vwap * 2.0,
      "volume": vol,
      "amount": amount,
      "total_shares": total_shares,
      "tradable_shares": total_shares * 0.8,
      "limited": np.zeros((n_objs, bars), dtype=np.int32),
      "f003": rng.uniform(2.0, 12.0, size=(n_objs, bars)),
      "f205": rng.uniform(5.0, 25.0, size=(n_objs, bars)),
    },
    objs,
    bars,
  )


def test_wq101_user_example_expression_compiles_runs_and_is_causal():
  panel = _make_rich_panel(n_objs=10, bars=90)
  expr = "((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))"
  fac = ExpressionFactor(expr, name="wq101_alpha7")
  assert fac.warmup_bars == 60
  out = fac.compute(panel)
  assert isinstance(out, np.ndarray)
  assert out.shape == (10, 90)
  assert np.sum(np.isfinite(out[:, 65:])) > 0

  norm = normalize_factor_output(out, panel.n_objs, panel.bars, fac.name)
  res = check_factor_causality(fac, panel, norm, warmup_bars=60, fail_fast=True)
  assert res["status"] == "PASS"


def test_gtja191_and_brain_dialects_and_scale_causality():
  panel = _make_rich_panel(n_objs=10, bars=70)
  ind_codes = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2], dtype=np.float64)

  exprs = {
    "brain_canonical": "group_neutralize(rank(ts_delta(close, 5) / ts_std_dev(returns, 10)), indclass.industry)",
    "gtja_sma3_decay": "decaylinear(sma(close, 5, 1) - tsmin(low, 10), 5)",
    "gtja_self_scan": "(close > open) ? (1 + SELF) : SELF",
    "scale_fixed": "scale(ts_delta(close, 3), 2)",
    "amount_vs_vol": "adv_amt5 / (adv5 + 1)",
    "val_bp": "f003 / raw_close",
    "log_cap": "log(cap)",
  }
  fac = ExpressionFactor(exprs)
  assert fac.fin_fields == ["f003"]
  assert fac.needs_industry is True

  fac.bind_context_metadata(industry_codes=ind_codes)
  outputs = fac.compute(panel)
  assert isinstance(outputs, dict)
  assert set(outputs.keys()) == set(exprs.keys())

  # Verify SCALE is strictly cross-sectional per bar (sum(|x_t|) == 2.0 on every valid bar)
  scaled = outputs["scale_fixed"]
  for t in range(10, 70):
    col_sum = float(np.nansum(np.abs(scaled[:, t])))
    assert np.isclose(col_sum, 2.0, atol=1e-6)

  # Verify ALL expressions (especially scale_fixed!) pass causality probe with zero lookahead
  norm = normalize_factor_output(outputs, panel.n_objs, panel.bars, "batch")
  res = check_factor_causality(fac, panel, norm, warmup_bars=15, fail_fast=True)
  assert res["status"] == "PASS"

  # Verify AMOUNT != VOLUME in PanelExecContext
  ctx = PanelExecContext(panel)
  assert not np.allclose(ctx("AMOUNT"), ctx("VOLUME"))
  assert np.allclose(ctx("AMOUNT"), panel.buffer("amount"))
  assert np.allclose(ctx("VOLUME"), panel.buffer("volume"))


def test_expression_guardrails_fail_fast():
  with pytest.raises(ValueError, match="Invalid power operator"):
    compile_expression("close ** 2", name="bad_pow")

  with pytest.raises(ValueError, match="Unknown operator 'TS_STD'"):
    compile_expression("ts_std(close, 10)", name="typo_op")

  with pytest.raises(ValueError, match="Unknown variable 'CLOS'"):
    compile_expression("ts_mean(clos, 10)", name="typo_var")

  with pytest.raises(ValueError, match="LOOKAHEAD DETECTED"):
    compile_expression("fret(open, close, volume, 1, 5)", name="bad_fret")

  with pytest.raises(ValueError, match="LOOKAHEAD DETECTED"):
    compile_expression("ts_delay(close, -2)", name="bad_lag")

  with pytest.raises(ValueError, match="ANTI-PATTERN DETECTED"):
    compile_expression("f003 / close", name="bad_bp")


def test_parse_inline_and_batch_expression_files(tmp_path):
  inline = parse_inline_expressions([
    "bp=f003 / raw_close",
    "(close == open) ? 0 : (close - open)",
  ])
  assert list(inline.keys()) == ["bp", "expr_01"]
  assert inline["bp"] == "f003 / raw_close"

  txt_path = tmp_path / "pool.txt"
  txt_path.write_text(
    "# Comment line\n"
    "mom5: ts_delta(close, 5) / ts_delay(close, 5)\n"
    "bp = f003 / raw_close\n"
    "rank(volume)\n",
    encoding="utf-8",
  )
  exprs_txt, dirs_txt = load_expression_file(str(txt_path))
  assert list(exprs_txt.keys()) == ["mom5", "bp", "expr_01"]
  assert dirs_txt["mom5"] == 1

  json_path = tmp_path / "pool.json"
  json_path.write_text(
    json.dumps({
      "rev": {"expr": "ts_delta(close, 5)", "direction": -1},
      "bp": "f003 / raw_close",
    }),
    encoding="utf-8",
  )
  exprs_js, dirs_js = load_expression_file(str(json_path))
  assert exprs_js["rev"] == "ts_delta(close, 5)"
  assert dirs_js["rev"] == -1
  assert dirs_js["bp"] == 1

  catalog = format_operator_catalog()
  assert "ts_mean" in catalog and "raw_close" in catalog and "group_neutralize" in catalog
