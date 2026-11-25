# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Factor base class, dynamic loader, AST static checks, and causality probe."""

from __future__ import annotations

import ast
import difflib
import importlib.util
import inspect
import logging
import os
import sys
from typing import Any

import numpy as np

from mbt.run.backend import instantiate_class
from mbt.select.panel import Panel

logger = logging.getLogger("factor")

FACTOR_SYSTEM_ARGS: set[str] = {
  "fin_fields",
  "fin_only_year",
  "fin_lookback_days",
  "warmup_bars",
  "periods",
  "quantiles",
  "entry",
  "winsorize",
  "standardize",
  "neutralize",
  "holdout",
  "cost_bps",
  "min_listed_days",
  "direction",
}

_LOOKAHEAD_CALLS = {"FRET", "BACKSET"}
_PER_SHARE_FIN_FIELDS = {"f001", "f002", "f003", "f004", "f005", "f006"}


class Factor:
  """Base class for vectorized cross-sectional alpha factors.

  Subclasses implement :meth:`compute`, which accepts a rectangular
  :class:`mbt.select.Panel` of shape ``(n_objs, bars)`` and returns either:

  * A single 2D ``np.ndarray`` of shape ``(n_objs, bars)`` (or 1D block buffer
    of length ``n_objs * bars``), or
  * A ``dict[str, np.ndarray]`` mapping factor names to matrices of the same
    shape for multi-factor batch evaluation and correlation analysis.

  Class Attributes:
    name: Default factor name when ``compute()`` returns a single array.
      Defaults to the class name when empty.
    fin_fields: Financial columns to preload from ``stock_financial`` (aligned
      strictly by statutory disclosure deadlines).
    fin_only_year: When ``True``, only annual (12-31) financial reports are loaded.
    fin_lookback_days: Explicit calendar-day lookback window for financial queries.
    warmup_bars: Number of trading bars loaded prior to the evaluation start date
      for rolling window initialization. These bars are excluded from evaluation
      metrics.
    direction: Expected factor direction: ``1`` (higher is better) or ``-1``
      (lower is better, e.g. PE/PB). For multi-factor families, may be a
      ``dict[str, int]`` mapping each factor name to ``1`` or ``-1``.
    needs_industry: Whether ``compute()`` expects an ``industry_code`` column or
      Shenwan industry metadata in the panel.
  """

  name: str = ""
  fin_fields: list[str] | None = None
  fin_only_year: bool = False
  fin_lookback_days: int | None = None
  warmup_bars: int = 0
  direction: int | dict[str, int] = 1
  needs_industry: bool = False

  def compute(self, panel: Panel) -> np.ndarray | dict[str, np.ndarray]:
    """Compute factor values across the full historical panel.

    Args:
      panel: Rectangular ``(n_objs, bars)`` market and financial data panel.
        Use geometric backward-adjusted columns (``open``, ``high``, ``low``,
        ``close``, ``vwap`` via ``BW_SPLIT_FACTOR``) for price-volume indicators,
        and unadjusted raw columns (``raw_close``, ``raw_open``, ...) for
        valuation ratios and market cap.

    Returns:
      Either a single factor array of shape ``(n_objs, bars)`` (or flat
      ``n_objs * bars``) or a dictionary ``{factor_name: array}``.
    """
    raise NotImplementedError(
      f"{self.__class__.__name__}.compute(panel) must be implemented by subclasses."
    )

  def get_direction(self, factor_name: str) -> int:
    """Return the configured direction (``1`` or ``-1``) for ``factor_name``."""
    if isinstance(self.direction, dict):
      d = int(self.direction.get(factor_name, 1))
    else:
      d = int(self.direction)
    if d not in (1, -1):
      raise ValueError(
        f"Invalid factor direction {d!r} for '{factor_name}'. Expected 1 or -1."
      )
    return d

  def finalize(self) -> dict[str, Any]:
    """Optional hook returning structured diagnostics for the factor report."""
    return {}


class _FactorAntiPatternVisitor(ast.NodeVisitor):
  """Static AST visitor catching lookahead primitives and price-basis mixing."""

  def __init__(self, filename: str = "<factor>"):
    self.filename = filename
    self.errors: list[str] = []

  def visit_FunctionDef(self, node: ast.FunctionDef):
    self._check_function(node)
    self.generic_visit(node)

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
    self._check_function(node)
    self.generic_visit(node)

  def _check_function(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef):
    fw_close_vars: set[str] = set()
    per_share_vars: set[str] = set()

    for stmt in ast.walk(func_node):
      if isinstance(stmt, ast.Call):
        self._check_call(stmt)
      if isinstance(stmt, ast.Assign):
        col = self._extract_panel_column_name(stmt.value)
        if col in ("close", "open", "high", "low", "vwap", "bw_price", "fw_price"):
          for t in stmt.targets:
            if isinstance(t, ast.Name):
              fw_close_vars.add(t.id)
        elif col in _PER_SHARE_FIN_FIELDS:
          for t in stmt.targets:
            if isinstance(t, ast.Name):
              per_share_vars.add(t.id)

    for stmt in ast.walk(func_node):
      if isinstance(stmt, ast.BinOp) and isinstance(stmt.op, ast.Div):
        left_fw = self._is_fw_price(stmt.left, fw_close_vars)
        right_fw = self._is_fw_price(stmt.right, fw_close_vars)
        left_fin = self._is_per_share_fin(stmt.left, per_share_vars)
        right_fin = self._is_per_share_fin(stmt.right, per_share_vars)
        if (left_fw and right_fin) or (right_fw and left_fin):
          lineno = getattr(stmt, "lineno", "?")
          msg = (
            f"{self.filename}:{lineno}: [ANTI-PATTERN DETECTED] Dividing adjusted "
            "price ('close/open/high/low/vwap/bw_price/fw_price') with per-share financial field ('f001'-'f006'). "
            "Per-share financials are reported on unadjusted share bases; mixing them with "
            "adjusted prices distorts historical valuation multiples after stock splits. "
            "Fix: Use 'raw_close' (e.g. panel.column('raw_close')) instead."
          )
          if msg not in self.errors:
            self.errors.append(msg)

  def _check_call(self, call: ast.Call):
    func = call.func
    attr_name: str | None = None
    if isinstance(func, ast.Attribute):
      attr_name = func.attr
    elif isinstance(func, ast.Name):
      attr_name = func.id

    if attr_name in _LOOKAHEAD_CALLS:
      lineno = getattr(call, "lineno", "?")
      msg = (
        f"{self.filename}:{lineno}: [LOOKAHEAD DETECTED] Forbidden lookahead function "
        f"'{attr_name}' used inside Factor definition. Factors must be strictly causal."
      )
      if msg not in self.errors:
        self.errors.append(msg)

    # Check negative shift in REF(..., -k) or panel.compute(al.REF, ..., -k) or np.roll(..., -k)
    if attr_name in ("REF", "roll"):
      neg_arg = self._find_negative_int_arg(call.args[1:])
      if neg_arg is not None:
        lineno = getattr(call, "lineno", "?")
        msg = (
          f"{self.filename}:{lineno}: [LOOKAHEAD DETECTED] Negative shift ({neg_arg}) "
          f"passed to '{attr_name}'. Shifting future bars backward introduces lookahead bias."
        )
        if msg not in self.errors:
          self.errors.append(msg)
    elif attr_name == "compute" and len(call.args) >= 3:
      first = call.args[0]
      first_name = (
        first.attr if isinstance(first, ast.Attribute)
        else (first.id if isinstance(first, ast.Name) else None)
      )
      if first_name in _LOOKAHEAD_CALLS:
        lineno = getattr(call, "lineno", "?")
        msg = (
          f"{self.filename}:{lineno}: [LOOKAHEAD DETECTED] Forbidden lookahead function "
          f"'{first_name}' passed to panel.compute()."
        )
        if msg not in self.errors:
          self.errors.append(msg)
      elif first_name == "REF":
        neg_arg = self._find_negative_int_arg(call.args[2:])
        if neg_arg is not None:
          lineno = getattr(call, "lineno", "?")
          msg = (
            f"{self.filename}:{lineno}: [LOOKAHEAD DETECTED] Negative shift ({neg_arg}) "
            "passed to panel.compute(al.REF, ...)."
          )
          if msg not in self.errors:
            self.errors.append(msg)

  @staticmethod
  def _find_negative_int_arg(args: list[ast.expr]) -> int | None:
    for arg in args:
      if (
        isinstance(arg, ast.UnaryOp)
        and isinstance(arg.op, ast.USub)
        and isinstance(arg.operand, ast.Constant)
        and isinstance(arg.operand.value, (int, float))
        and arg.operand.value > 0
      ):
        return -int(arg.operand.value)
      if (
        isinstance(arg, ast.Constant)
        and isinstance(arg.value, (int, float))
        and arg.value < 0
      ):
        return int(arg.value)
    return None

  @staticmethod
  def _extract_panel_column_name(node: ast.AST) -> str | None:
    if (
      isinstance(node, ast.Call)
      and isinstance(node.func, ast.Attribute)
      and node.func.attr in ("column", "buffer")
      and len(node.args) >= 1
      and isinstance(node.args[0], ast.Constant)
      and isinstance(node.args[0].value, str)
    ):
      return node.args[0].value
    return None

  def _is_fw_price(self, node: ast.AST, fw_vars: set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in fw_vars:
      return True
    col = self._extract_panel_column_name(node)
    return col in ("close", "open", "high", "low", "vwap", "bw_price", "fw_price")

  def _is_per_share_fin(self, node: ast.AST, fin_vars: set[str]) -> bool:
    if isinstance(node, ast.Name) and node.id in fin_vars:
      return True
    col = self._extract_panel_column_name(node)
    return col in _PER_SHARE_FIN_FIELDS


def check_factor_source(
  source: str,
  filename: str = "<factor>",
  fail_fast: bool = True,
) -> list[str]:
  """Statically inspect Factor Python source code for lookahead and price-basis anti-patterns.

  Args:
    source: Python source text.
    filename: File path or identifier for error reporting.
    fail_fast: When ``True``, raises ``ValueError`` on the first detected violation.

  Returns:
    List of diagnostic error/warning messages.
  """
  try:
    tree = ast.parse(source, filename=filename)
  except Exception as e:
    logger.debug(f"Could not parse factor AST for validation: {e}")
    return []

  visitor = _FactorAntiPatternVisitor(filename=filename)
  visitor.visit(tree)
  if fail_fast and visitor.errors:
    raise ValueError(visitor.errors[0])
  return visitor.errors


def normalize_factor_output(
  raw: np.ndarray | dict[str, np.ndarray],
  n_objs: int,
  bars: int,
  default_name: str,
) -> dict[str, np.ndarray]:
  """Validate and normalize ``Factor.compute()`` output to ``{name: float64[n_objs, bars]}``.

  Args:
    raw: Raw output returned by ``Factor.compute(panel)``.
    n_objs: Expected number of stocks (rows).
    bars: Expected number of bars (columns).
    default_name: Name used when ``raw`` is a single array.

  Returns:
    Dictionary mapping each factor name to a C-contiguous ``float64`` 2D array
    of shape ``(n_objs, bars)`` with ``+/-inf`` converted to ``np.nan``.

  Raises:
    TypeError: If ``raw`` is neither an ndarray nor a dict of ndarrays.
    ValueError: If any array has an incompatible shape or empty name.
  """
  if isinstance(raw, dict):
    if len(raw) == 0:
      raise ValueError("Factor.compute() returned an empty dict; expected at least one factor.")
    items = raw
  else:
    name = default_name.strip() or "factor"
    items = {name: raw}

  normalized: dict[str, np.ndarray] = {}
  expected_flat = n_objs * bars
  for name, val in items.items():
    if not isinstance(name, str) or not name.strip():
      raise ValueError(f"Factor name must be a non-empty string, got {name!r}")
    arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 2:
      if arr.shape != (n_objs, bars):
        raise ValueError(
          f"Factor '{name}' shape mismatch: expected ({n_objs}, {bars}), got {arr.shape}."
        )
    elif arr.ndim == 1:
      if arr.size != expected_flat:
        raise ValueError(
          f"Factor '{name}' length mismatch: expected {expected_flat} "
          f"({n_objs} objs x {bars} bars), got {arr.size}."
        )
      arr = arr.reshape(n_objs, bars)
    else:
      raise ValueError(
        f"Factor '{name}' must be 1D or 2D array, got {arr.ndim}D with shape {arr.shape}."
      )

    clean = np.where(np.isfinite(arr), arr, np.nan)
    normalized[name.strip()] = np.ascontiguousarray(clean, dtype=np.float64)

  return normalized


def check_factor_causality(
  factor: Factor,
  panel: Panel,
  full_outputs: dict[str, np.ndarray],
  *,
  warmup_bars: int = 0,
  n_probes: int = 2,
  rtol: float = 1e-5,
  atol: float = 1e-6,
  fail_fast: bool = True,
) -> dict[str, Any]:
  """Verify strict causality by truncating the panel at probe bars and comparing outputs.

  For each probe bar ``t``, slices ``panel.slice(0, t + 1)`` (removing all future
  bars ``> t``), recomputes ``factor.compute(sub_panel)``, and verifies that the
  cross-section at bar ``t`` matches the full-panel output at bar ``t``.

  Args:
    factor: Factor instance under test.
    panel: Full historical panel passed to ``factor.compute``.
    full_outputs: Normalized full-panel outputs ``{name: (n_objs, bars)}``.
    warmup_bars: Leading warmup bar count excluded from probe selection.
    n_probes: Number of truncation probe timestamps (default 2).
    rtol: Relative comparison tolerance.
    atol: Absolute comparison tolerance.
    fail_fast: If ``True``, raises ``ValueError`` immediately when lookahead is detected.

  Returns:
    Dictionary summarizing probe status (``status`` in ``'PASS'``, ``'FAIL'``,
    ``'SKIPPED'``), checked probe bars, and any detected violations.
  """
  bars = panel.bars
  n_objs = panel.n_objs
  start_probe = max(warmup_bars + 2, 5)
  end_probe = bars - 2

  if n_probes <= 0 or end_probe <= start_probe:
    return {
      "status": "SKIPPED",
      "reason": f"insufficient bars for causality probe (bars={bars}, warmup_bars={warmup_bars})",
      "probes": [],
      "violations": [],
    }

  if n_probes == 1:
    probe_bars = [(start_probe + end_probe) // 2]
  else:
    fracs = np.linspace(0.35, 0.80, n_probes)
    probe_bars = sorted(
      {int(round(start_probe + f * (end_probe - start_probe))) for f in fracs}
    )

  timeline = panel.timeline()
  default_name = factor.name or factor.__class__.__name__
  violations: list[dict[str, Any]] = []

  for t in probe_bars:
    sub_panel = panel.slice(0, t + 1)
    sub_raw = factor.compute(sub_panel)
    sub_outputs = normalize_factor_output(sub_raw, n_objs, t + 1, default_name)

    date_str = str(timeline[t])[:10] if len(timeline) > t else str(t)
    for fname, full_mat in full_outputs.items():
      if fname not in sub_outputs:
        violations.append({
          "factor": fname,
          "bar": t,
          "date": date_str,
          "reason": "factor missing on truncated panel",
        })
        continue

      full_cs = full_mat[:, t]
      sub_cs = sub_outputs[fname][:, -1]

      both_nan = np.isnan(full_cs) & np.isnan(sub_cs)
      one_nan = np.isnan(full_cs) ^ np.isnan(sub_cs)
      both_fin = ~np.isnan(full_cs) & ~np.isnan(sub_cs)

      diff_fin = np.zeros(n_objs, dtype=bool)
      max_diff = 0.0
      if np.any(both_fin):
        abs_diff = np.abs(full_cs[both_fin] - sub_cs[both_fin])
        tol = atol + rtol * np.abs(full_cs[both_fin])
        bad_fin = abs_diff > tol
        if np.any(bad_fin):
          diff_fin[both_fin] = bad_fin
          max_diff = float(np.max(abs_diff[bad_fin]))

      mismatch = one_nan | diff_fin
      n_mismatch = int(np.sum(mismatch))
      if n_mismatch > 0 and not np.all(both_nan):
        violations.append({
          "factor": fname,
          "bar": t,
          "date": date_str,
          "mismatched_objs": n_mismatch,
          "nan_mismatch": int(np.sum(one_nan)),
          "max_abs_diff": max_diff,
        })

  if violations:
    first = violations[0]
    msg = (
      f"[LOOKAHEAD BIAS DETECTED] Factor '{first['factor']}' failed causality probe at "
      f"bar {first['bar']} ({first['date']}): {first['mismatched_objs']} stock(s) changed "
      f"when future bars were truncated (max_abs_diff={first.get('max_abs_diff', 0.0):.6g}, "
      f"nan_mismatch={first.get('nan_mismatch', 0)}). "
      "Ensure computations do not aggregate across the entire time axis or index future bars."
    )
    if fail_fast:
      raise ValueError(msg)
    return {
      "status": "FAIL",
      "probes": probe_bars,
      "violations": violations,
      "message": msg,
    }

  return {
    "status": "PASS",
    "probes": probe_bars,
    "violations": [],
  }


def validate_factor_kwargs(factor_cls: type[Factor], kwargs: dict[str, Any] | None) -> None:
  """Fail-Fast check ensuring all ``-a key=value`` arguments are accepted by ``factor_cls``."""
  if not kwargs:
    return
  valid_keys = set(FACTOR_SYSTEM_ARGS)
  has_var_kw = False
  if factor_cls.__init__ is not object.__init__:
    sig = inspect.signature(factor_cls.__init__)
    for pname, param in sig.parameters.items():
      if pname == "self":
        continue
      if param.kind == inspect.Parameter.VAR_KEYWORD:
        has_var_kw = True
      else:
        valid_keys.add(pname)
  if has_var_kw:
    return

  orphan_keys = [k for k in kwargs if k not in valid_keys]
  if orphan_keys:
    custom_params = sorted(valid_keys - FACTOR_SYSTEM_ARGS)
    suggestions: list[str] = []
    for k in orphan_keys:
      m = difflib.get_close_matches(k, sorted(valid_keys), n=1, cutoff=0.6)
      if m:
        suggestions.append(f"'{k}' -> did you mean '{m[0]}'?")
    hint_parts = []
    if suggestions:
      hint_parts.append(" ".join(suggestions))
    if custom_params:
      hint_parts.append(f"Accepted parameters: {custom_params}.")
    else:
      hint_parts.append("Factor declares no custom __init__ parameters.")
    raise ValueError(
      f"Unrecognized argument(s) passed via -a/--arg: {orphan_keys}. "
      f"Factor ({factor_cls.__name__}) does not accept these arguments. "
      + " ".join(hint_parts)
    )


def load_factor(
  py_file: str,
  kwargs: dict[str, Any] | None = None,
  *,
  strict_args: bool = True,
) -> Factor:
  """Load and instantiate a ``Factor`` subclass from a Python source file.

  Performs static AST anti-pattern checks and Fail-Fast parameter validation
  before instantiating the factor class.

  Args:
    py_file: Path to the Python file defining a ``Factor`` subclass.
    kwargs: Optional keyword arguments parsed from CLI ``-a key=value``.
    strict_args: When ``True``, unrecognized arguments raise ``ValueError``.

  Returns:
    Instantiated ``Factor`` subclass with ``_source_module`` and ``_author``
    metadata attached when available.
  """
  path = os.path.abspath(py_file)
  if not os.path.isfile(path):
    raise FileNotFoundError(f"Factor file not found: '{py_file}'")

  with open(path, "r", encoding="utf-8") as f:
    source = f.read()
  check_factor_source(source, filename=path, fail_fast=True)

  dir_path = os.path.dirname(path)
  if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

  module_name = os.path.splitext(os.path.basename(path))[0]
  spec = importlib.util.spec_from_file_location(module_name, path)
  if not spec or not spec.loader:
    raise RuntimeError(f"Failed to load module spec from '{py_file}'")

  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)

  # Avoid importing ExpressionFactor as a discovered subclass when user imports it
  from mbt.factor.expr import ExpressionFactor

  candidates: list[type[Factor]] = []
  for obj in vars(module).values():
    if (
      isinstance(obj, type)
      and issubclass(obj, Factor)
      and obj not in (Factor, ExpressionFactor)
      and obj.__module__ == module.__name__
    ):
      candidates.append(obj)

  if not candidates:
    # Fallback: allow subclasses imported/defined without matching __module__
    for obj in vars(module).values():
      if (
        isinstance(obj, type)
        and issubclass(obj, Factor)
        and obj not in (Factor, ExpressionFactor)
      ):
        candidates.append(obj)

  if not candidates:
    raise ValueError(
      f"Could not find a Factor subclass in '{py_file}'. "
      "Define a class inheriting from `mbt.factor.Factor` implementing `compute(self, panel)`."
    )

  factor_cls = candidates[0]
  if strict_args:
    validate_factor_kwargs(factor_cls, kwargs)

  inst = instantiate_class(factor_cls, kwargs)
  setattr(inst, "_source_module", module)
  author = getattr(module, "M_AUTHOR", "")
  if author:
    setattr(inst, "_author", str(author))
  return inst
