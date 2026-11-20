# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Factor expression compiler, PanelExecContext bridge, and dialect standardization.

Uses ``py-alpha-lib`` (``alpha.lang.to_python`` + ``alpha.ExecContext``) to
compile WorldQuant BRAIN (canonical), WorldQuant 101, and GTJA 191 alpha
expressions directly onto :class:`mbt.select.Panel` buffers with Fail-Fast
static validation and automatic dependency inference.
"""

from __future__ import annotations

import ast
import dataclasses
import difflib
import json
import os
import re
from typing import Any, Callable

import alpha as al
import numpy as np

from mbt.factor.factor import Factor
from mbt.select.panel import Panel

_FIN_FIELD_RE = re.compile(r"^F(\d{3})$")
_ADV_RE = re.compile(r"^ADV(\d*)$")
_ADV_AMT_RE = re.compile(r"^(?:ADV_AMT|ADTV)(\d+)$")
_INLINE_NAME_RE = re.compile(r"^([a-zA-Z_]\w*)\s*=(?!=)\s*(.+)$")
_PER_SHARE_FIN_UPPER = {"F001", "F002", "F003", "F004", "F005", "F006"}
_FW_PRICE_UPPER = {"OPEN", "HIGH", "LOW", "CLOSE", "VWAP"}
_FORBIDDEN_OPS = {"FRET", "BACKSET"}
_LAG_CHECK_OPS = {"TS_DELAY", "DELAY", "REF", "TS_DELTA", "DELTA"}
_ROLLING_WINDOW_OPS = {
  "TS_SUM", "SUM", "SUMAC", "TS_MEAN", "MA", "MEAN", "EMA", "SMA",
  "TS_STD_DEV", "STDDEV", "STD", "TS_VARIANCE", "VAR",
  "TS_CORRELATION", "CORR", "CORRELATION", "TS_COVARIANCE", "COV", "COVARIANCE",
  "TS_MAX", "TSMAX", "TS_MIN", "TSMIN", "TS_ARGMAX", "TS_ARGMIN", "HIGHDAY", "LOWDAY",
  "TS_RANK", "TSRANK", "TS_QUANTILE", "TS_DELAY", "DELAY", "REF", "TS_DELTA", "DELTA",
  "TS_DECAY_LINEAR", "DECAY_LINEAR", "DECAYLINEAR", "WMA",
  "TS_PRODUCT", "PRODUCT", "PROD", "TS_REGBETA", "REGBETA", "TS_REGRESI", "REGRESI",
  "SLOPE", "INTERCEPT", "TS_COUNT", "COUNT", "TS_SUMIF", "SUMIF",
  "TS_ZSCORE", "TS_SKEWNESS", "TS_KURTOSIS", "TS_COUNT_NANS", "TS_ENTROPY",
  "TS_MIN_MAX_DIFF", "TS_WEIGHTED_DELAY", "TS_MOMENT",
}

KNOWN_FIELDS: set[str] = {
  # Forward-adjusted market columns
  "OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "VOLUME", "VOL", "AMOUNT",
  # Unadjusted raw price columns (for valuation & market cap)
  "RAW_OPEN", "RAW_HIGH", "RAW_LOW", "RAW_CLOSE", "RAW_VWAP", "PRICE",
  # Share capital & market cap
  "TOTAL_SHARES", "TRADABLE_SHARES", "CAP", "FLOAT_CAP",
  # Corporate actions & trading state
  "DIVIDEND", "TRANSFER_SHARES", "RIGHT_SHARES", "RIGHT_PRICE", "LIMITED",
  # Derived price-volume series
  "RETURNS", "RET", "DTM", "DBM", "TR", "HD", "LD", "SEQUENCE",
  # Benchmark series (GTJA191 typo preserved + clean alias)
  "BANCHMARKINDEXCLOSE", "BANCHMARKINDEXOPEN", "BENCHMARK_CLOSE", "BENCHMARK_OPEN",
  # Industry classification
  "INDUSTRY", "INDCLASS.INDUSTRY", "INDCLASS.SUBINDUSTRY", "INDCLASS.SECTOR",
}


class PanelExecContext(al.ExecContext):
  """Zero-copy :class:`alpha.ExecContext` adapter backed by :class:`mbt.select.Panel`.

  Improvements over the base ``ExecContext``:
    1. Binds directly to ``Panel`` buffers without DataFrame copying or column renaming.
    2. Exposes ``raw_close``/``raw_open``/``raw_high``/``raw_low``/``raw_vwap``,
       ``cap`` (``raw_close * total_shares``), ``float_cap``, ``amount`` (distinct
       from ``volume``), Shenwan industry codes, and any preloaded financial fields
       (``f001``..``fNNN``).
    3. Fixes ``SCALE`` to normalize strictly per cross-section (each bar) rather
       than across the entire time series, eliminating lookahead bias.
    4. Binds ``BANCHMARKINDEXCLOSE``/``OPEN`` (and ``BENCHMARK_CLOSE``/``OPEN``)
       to the actual benchmark curve when supplied, falling back to equal-weight
       market averages instead of stock index 0.
    5. Adds canonical BRAIN aliases (``GROUP_NEUTRALIZE``, ``REF``, ``IF``,
       ``SQRT``, ``POWER``, ``CLIP``) and blocks lookahead operators (``FRET``).
  """

  def __init__(
    self,
    panel: Panel,
    *,
    industry_codes: np.ndarray | None = None,
    benchmark_close: np.ndarray | None = None,
    benchmark_open: np.ndarray | None = None,
  ):
    self.panel = panel
    self.securities = panel.n_objs
    self.trades = panel.bars
    self.algo_ctx = al.Context(
      groups=max(1, self.securities),
      flags=al.FLAG_SKIP_NAN,
      start=0,
      end=self.trades,
    )
    self._cols_lower: dict[str, str] = {c.upper(): c for c in panel.columns}
    self._cache: dict[str, np.ndarray] = {}
    self._industry_codes = industry_codes
    self._benchmark_close = benchmark_close
    self._benchmark_open = benchmark_open

    # Core OHLCV buffers
    self.OPEN = self._buf("open")
    self.HIGH = self._buf("high")
    self.LOW = self._buf("low")
    self.CLOSE = self._buf("close")
    self.VOLUME = self._buf("volume") if "VOLUME" in self._cols_lower else np.ones_like(self.CLOSE)
    if "AMOUNT" in self._cols_lower:
      self.AMOUNT = self._buf("amount")
    else:
      self.AMOUNT = self.VOLUME * self.CLOSE

    if "VWAP" in self._cols_lower:
      self.VWAP = self._buf("vwap")
    elif "RAW_VWAP" in self._cols_lower:
      self.VWAP = self._buf("raw_vwap")
    else:
      self.VWAP = (self.HIGH + self.LOW + self.CLOSE) / 3.0

    # Raw unadjusted prices
    self.RAW_OPEN = self._buf("raw_open") if "RAW_OPEN" in self._cols_lower else self.OPEN
    self.RAW_HIGH = self._buf("raw_high") if "RAW_HIGH" in self._cols_lower else self.HIGH
    self.RAW_LOW = self._buf("raw_low") if "RAW_LOW" in self._cols_lower else self.LOW
    self.RAW_CLOSE = self._buf("raw_close") if "RAW_CLOSE" in self._cols_lower else (
      self._buf("price") if "PRICE" in self._cols_lower else self.CLOSE
    )
    self.PRICE = self.RAW_CLOSE
    self.RAW_VWAP = self._buf("raw_vwap") if "RAW_VWAP" in self._cols_lower else self.VWAP

    # Industry classification arrays
    if industry_codes is not None:
      ind_1d = np.asarray(industry_codes, dtype=np.float64)
      if ind_1d.shape == (self.securities,):
        ind_flat = np.repeat(ind_1d, self.trades)
      elif ind_1d.size == self.securities * self.trades:
        ind_flat = ind_1d.reshape(-1)
      else:
        raise ValueError(
          f"industry_codes shape mismatch: expected ({self.securities},), got {ind_1d.shape}"
        )
      self.INDCLASS_INDUSTRY = ind_flat
      self.INDCLASS_SUBINDUSTRY = ind_flat
      self.INDCLASS_SECTOR = ind_flat
      self.INDUSTRY = ind_flat

  def _buf(self, col_name: str) -> np.ndarray:
    arr = self.panel.buffer(col_name)
    if arr.dtype != np.float64:
      return arr.astype(np.float64, copy=False)
    return arr

  def _ensure_derived(self, name: str) -> np.ndarray:
    if name in self._cache:
      return self._cache[name]

    if name in ("RETURNS", "RET"):
      ref_c = al.REF(self.CLOSE, 1)
      with np.errstate(divide="ignore", invalid="ignore"):
        val = np.where(ref_c > 0.0, self.CLOSE / ref_c - 1.0, np.nan)
      self._cache["RETURNS"] = val
      self._cache["RET"] = val
      return val

    if name == "DTM":
      val = self._calc_DTM()
      self._cache[name] = val
      return val

    if name == "DBM":
      val = self._calc_DBM()
      self._cache[name] = val
      return val

    if name == "TR":
      val = self._calc_TR()
      self._cache[name] = val
      return val

    if name == "HD":
      val = self.HIGH - al.REF(self.HIGH, 1)
      self._cache[name] = val
      return val

    if name == "LD":
      val = self.LOW - al.REF(self.LOW, 1)
      self._cache[name] = val
      return val

    if name == "SEQUENCE":
      val = np.tile(np.arange(1, self.trades + 1, dtype=np.float64), self.securities)
      self._cache[name] = val
      return val

    if name == "CAP":
      if "TOTAL_SHARES" not in self._cols_lower:
        raise ValueError(
          "Field 'CAP' requires column 'total_shares' in Panel (not found)."
        )
      val = self.RAW_CLOSE * self._buf("total_shares")
      self._cache[name] = val
      return val

    if name == "FLOAT_CAP":
      if "TRADABLE_SHARES" not in self._cols_lower:
        raise ValueError(
          "Field 'FLOAT_CAP' requires column 'tradable_shares' in Panel (not found)."
        )
      val = self.RAW_CLOSE * self._buf("tradable_shares")
      self._cache[name] = val
      return val

    if name in ("BANCHMARKINDEXCLOSE", "BENCHMARK_CLOSE"):
      if self._benchmark_close is not None and len(self._benchmark_close) == self.trades:
        b = np.asarray(self._benchmark_close, dtype=np.float64)
      else:
        with np.errstate(all="ignore"):
          b = np.nanmean(self.CLOSE.reshape(self.securities, self.trades), axis=0)
      val = np.tile(b, self.securities)
      self._cache["BANCHMARKINDEXCLOSE"] = val
      self._cache["BENCHMARK_CLOSE"] = val
      return val

    if name in ("BANCHMARKINDEXOPEN", "BENCHMARK_OPEN"):
      if self._benchmark_open is not None and len(self._benchmark_open) == self.trades:
        b = np.asarray(self._benchmark_open, dtype=np.float64)
      else:
        with np.errstate(all="ignore"):
          b = np.nanmean(self.OPEN.reshape(self.securities, self.trades), axis=0)
      val = np.tile(b, self.securities)
      self._cache["BANCHMARKINDEXOPEN"] = val
      self._cache["BENCHMARK_OPEN"] = val
      return val

    raise KeyError(name)

  def __call__(self, name: str) -> np.ndarray:
    key = name.upper()
    if key == "VOL":
      return self.VOLUME
    if key == "AMOUNT":
      return self.AMOUNT
    if key in self._cache:
      return self._cache[key]

    m_amt = _ADV_AMT_RE.match(key)
    if m_amt:
      n = int(m_amt.group(1))
      if n <= 0:
        raise ValueError(f"Window for '{key}' must be positive, got {n}")
      val = al.MA(self.AMOUNT, n)
      self._cache[key] = val
      return val

    m_adv = _ADV_RE.match(key)
    if m_adv:
      n_str = m_adv.group(1)
      if not n_str:
        return self.VOLUME
      n = int(n_str)
      if n <= 0:
        raise ValueError(f"Window for '{key}' must be positive, got {n}")
      val = al.MA(self.VOLUME, n)
      self._cache[key] = val
      return val

    if key.startswith("INDCLASS."):
      attr = key.replace(".", "_")
      if hasattr(self, attr):
        return getattr(self, attr)
      raise ValueError(
        f"Field '{name}' requires industry classification metadata, which was not provided."
      )

    if key in (
      "RETURNS", "RET", "DTM", "DBM", "TR", "HD", "LD", "SEQUENCE",
      "CAP", "FLOAT_CAP", "BANCHMARKINDEXCLOSE", "BENCHMARK_CLOSE",
      "BANCHMARKINDEXOPEN", "BENCHMARK_OPEN",
    ):
      return self._ensure_derived(key)

    if hasattr(self, key):
      val = getattr(self, key)
      if isinstance(val, np.ndarray):
        return val

    if key in self._cols_lower:
      val = self._buf(self._cols_lower[key])
      self._cache[key] = val
      return val

    raise ValueError(
      f"Unknown field '{name}' referenced in factor expression. "
      f"Available Panel columns: {sorted(self.panel.columns)}"
    )

  # ── Fixed & Extended Operators ──────────────────────────────────────

  def SCALE(self, a: np.ndarray, k: float = 1.0) -> np.ndarray:
    """Cross-sectional L1 scaling per bar so ``sum(|x_t|) == k`` (strictly causal)."""
    arr_2d = np.asarray(a, dtype=np.float64).reshape(self.securities, self.trades)
    denom = np.nansum(np.abs(arr_2d), axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
      out = np.where(denom > 0.0, arr_2d * float(k) / denom, np.nan)
    return out.reshape(-1)

  def TS_DELAY(self, a: np.ndarray, p: int) -> np.ndarray:
    p_int = int(p)
    if p_int < 0:
      raise ValueError(f"Negative lag ({p_int}) in TS_DELAY/DELAY/REF is forbidden (lookahead bias).")
    return al.REF(a, p_int)

  DELAY = TS_DELAY
  REF = TS_DELAY

  def TS_DELTA(self, a: np.ndarray, p: int) -> np.ndarray:
    p_int = int(p)
    if p_int < 0:
      raise ValueError(f"Negative lag ({p_int}) in TS_DELTA/DELTA is forbidden (lookahead bias).")
    return a - al.REF(a, p_int)

  DELTA = TS_DELTA

  def FRET(self, open: np.ndarray, close: np.ndarray, is_calc: np.ndarray, delay: int, periods: int) -> np.ndarray:
    raise ValueError("Operator 'FRET' is forbidden inside factor expressions (lookahead bias).")

  GROUP_NEUTRALIZE = al.ExecContext.INDNEUTRALIZE

  def IF(self, cond: np.ndarray, true_val: Any, false_val: Any) -> np.ndarray:
    return np.where(cond, true_val, false_val)

  def LOG(self, a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.where(arr > 0.0, np.log(np.where(arr > 0.0, arr, 1.0)), np.nan)

  def LOG1P(self, a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.where(arr > -1.0, np.log1p(np.where(arr > -1.0, arr, 0.0)), np.nan)

  def SQRT(self, a: np.ndarray) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    with np.errstate(invalid="ignore"):
      return np.where(arr >= 0.0, np.sqrt(np.where(arr >= 0.0, arr, 0.0)), np.nan)

  def POWER(self, a: np.ndarray, p: Any) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.power(a, p)

  def SIGNEDPOWER(self, a: np.ndarray, p: Any) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
      return np.sign(a) * np.power(np.abs(a), p)

  def CLIP(self, a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(a, float(lo), float(hi))

  def SCAN_ADD(self, operand: Any, cond: np.ndarray) -> np.ndarray:
    c = np.asarray(cond, dtype=bool)
    op = np.asarray(operand, dtype=np.float64)
    if op.ndim == 0:
      op = np.full(c.shape, float(op), dtype=np.float64)
    return al.SCAN_ADD(op, c)

  def SCAN_MUL(self, operand: Any, cond: np.ndarray) -> np.ndarray:
    c = np.asarray(cond, dtype=bool)
    op = np.asarray(operand, dtype=np.float64)
    if op.ndim == 0:
      op = np.full(c.shape, float(op), dtype=np.float64)
    return al.SCAN_MUL(op, c)


def _valid_operator_names() -> set[str]:
  ops: set[str] = set()
  for cls in PanelExecContext.__mro__:
    for k, v in cls.__dict__.items():
      if not k.startswith("_") and callable(v) and k not in _FORBIDDEN_OPS:
        ops.add(k)
  return ops


@dataclasses.dataclass(frozen=True)
class CompiledExpression:
  """Compiled and statically verified factor expression artifact."""

  name: str
  expression: str
  python_code: str
  fn: Callable[[PanelExecContext], np.ndarray]
  variables: tuple[str, ...]
  operators: tuple[str, ...]
  fin_fields: tuple[str, ...]
  needs_industry: bool
  needs_cap: bool
  suggested_warmup: int


def _extract_int_const(node: ast.AST) -> int | None:
  if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
    return int(round(node.value))
  if (
    isinstance(node, ast.UnaryOp)
    and isinstance(node.op, ast.USub)
    and isinstance(node.operand, ast.Constant)
    and isinstance(node.operand.value, (int, float))
  ):
    return -int(round(node.operand.value))
  return None


def _is_ctx_var_call(node: ast.AST) -> str | None:
  if (
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Name)
    and node.func.id == "ctx"
    and len(node.args) == 1
    and isinstance(node.args[0], ast.Constant)
    and isinstance(node.args[0].value, str)
  ):
    return node.args[0].value.upper()
  return None


def _validate_variable_name(var_upper: str) -> None:
  if var_upper in KNOWN_FIELDS:
    return
  if _FIN_FIELD_RE.match(var_upper):
    return
  if _ADV_RE.match(var_upper) or _ADV_AMT_RE.match(var_upper):
    return
  matches = difflib.get_close_matches(var_upper, sorted(KNOWN_FIELDS), n=3, cutoff=0.5)
  hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
  raise ValueError(
    f"Unknown variable '{var_upper}' in factor expression.{hint} "
    "Use `python -m mbt.factor.cli --list-ops` to inspect supported fields and operators."
  )


def compile_expression(
  expression: str,
  name: str = "expr_factor",
  *,
  extra_fields: set[str] | None = None,
) -> CompiledExpression:
  """Transpile, statically validate, and compile an alpha expression string.

  Args:
    expression: Alpha formula string (WorldQuant BRAIN canonical, WQ101, or GTJA191).
    name: Logical factor name.
    extra_fields: Optional additional uppercase column names allowed on the panel.

  Returns:
    :class:`CompiledExpression` with inferred dependencies and compiled callable.

  Raises:
    ValueError: On syntax errors, unknown operators/fields, lookahead primitives,
      negative lags, or forward-adjusted valuation anti-patterns.
  """
  expr = expression.strip()
  if not expr:
    raise ValueError(f"Factor expression '{name}' cannot be empty.")

  if "**" in expr:
    raise ValueError(
      f"Invalid power operator '**' in factor expression '{name}': '{expr}'. "
      "Use '^' for exponentiation (alpha.lang syntax), e.g. 'close ^ 2' or 'signedpower(x, 2)'."
    )

  safe_fn_name = re.sub(r"\W+", "_", name).strip("_") or "expr_factor"
  if safe_fn_name[0].isdigit():
    safe_fn_name = "f_" + safe_fn_name

  try:
    py_code = al.to_python(
      safe_fn_name,
      expr,
      as_function=True,
      optimize=True,
      name_convertor=str.upper,
    )
  except Exception as e:
    raise ValueError(
      f"Failed to parse factor expression '{name}': {e}\n  Expression: {expr}"
    ) from e

  tree = ast.parse(py_code, filename=f"<expr:{name}>")
  valid_ops = _valid_operator_names()
  allowed_extra = {c.upper() for c in extra_fields} if extra_fields else set()

  var_aliases: dict[str, str] = {}
  variables: set[str] = set()
  operators: set[str] = set()
  suggested_warmup = 0

  # Pass 1: track local variable bindings (_CLOSE = ctx('CLOSE'))
  for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and len(node.targets) == 1:
      target = node.targets[0]
      vname = _is_ctx_var_call(node.value)
      if isinstance(target, ast.Name) and vname is not None:
        var_aliases[target.id] = vname

  def _resolve_var(n: ast.AST) -> str | None:
    direct = _is_ctx_var_call(n)
    if direct is not None:
      return direct
    if isinstance(n, ast.Name) and n.id in var_aliases:
      return var_aliases[n.id]
    return None

  # Pass 2: validate calls, variables, lags, and price-basis safety
  for node in ast.walk(tree):
    vname = _is_ctx_var_call(node)
    if vname is not None:
      variables.add(vname)
      if vname not in allowed_extra:
        _validate_variable_name(vname)
      m_adv = _ADV_RE.match(vname)
      if m_adv and m_adv.group(1):
        suggested_warmup = max(suggested_warmup, int(m_adv.group(1)))
      m_amt = _ADV_AMT_RE.match(vname)
      if m_amt:
        suggested_warmup = max(suggested_warmup, int(m_amt.group(1)))
      continue

    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
      if isinstance(node.func.value, ast.Name) and node.func.value.id == "ctx":
        op = node.func.attr
        if op in _FORBIDDEN_OPS:
          raise ValueError(
            f"[LOOKAHEAD DETECTED] Forbidden lookahead operator '{op}' in factor expression '{name}'."
          )
        if op not in valid_ops:
          matches = difflib.get_close_matches(op, sorted(valid_ops), n=3, cutoff=0.5)
          hint = f" Did you mean: {', '.join(matches)}?" if matches else ""
          raise ValueError(
            f"Unknown operator '{op}' in factor expression '{name}'.{hint} "
            "Use `python -m mbt.factor.cli --list-ops` to inspect supported operators."
          )
        operators.add(op)

        if op in _LAG_CHECK_OPS and len(node.args) >= 2:
          lag_val = _extract_int_const(node.args[1])
          if lag_val is not None and lag_val < 0:
            raise ValueError(
              f"[LOOKAHEAD DETECTED] Negative lag ({lag_val}) passed to '{op}' in expression '{name}'."
            )
        if op in _ROLLING_WINDOW_OPS:
          for arg in node.args[1:]:
            w = _extract_int_const(arg)
            if w is not None:
              if w < 0:
                raise ValueError(
                  f"[LOOKAHEAD DETECTED] Negative window ({w}) passed to '{op}' in expression '{name}'."
                )
              suggested_warmup = max(suggested_warmup, w)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
      lv = _resolve_var(node.left)
      rv = _resolve_var(node.right)
      if (lv in _PER_SHARE_FIN_UPPER and rv in _FW_PRICE_UPPER) or (
        rv in _PER_SHARE_FIN_UPPER and lv in _FW_PRICE_UPPER
      ):
        raise ValueError(
          f"[ANTI-PATTERN DETECTED] Factor expression '{name}' divides per-share financial "
          f"field ('{lv if lv in _PER_SHARE_FIN_UPPER else rv}') with forward-adjusted price "
          f"('{rv if rv in _FW_PRICE_UPPER else lv}'). "
          "Fix: Use 'raw_close' instead of 'close' for valuation ratios (e.g. 'f003 / raw_close')."
        )

  fin_fields = tuple(sorted(v.lower() for v in variables if _FIN_FIELD_RE.match(v)))
  needs_industry = any(v == "INDUSTRY" or v.startswith("INDCLASS.") for v in variables)
  needs_cap = any(v in ("CAP", "FLOAT_CAP", "TOTAL_SHARES", "TRADABLE_SHARES") for v in variables)

  ns: dict[str, Any] = {"np": np}
  exec(compile(tree, filename=f"<expr:{name}>", mode="exec"), ns)
  fn = ns[safe_fn_name]

  return CompiledExpression(
    name=name,
    expression=expr,
    python_code=py_code,
    fn=fn,
    variables=tuple(sorted(variables)),
    operators=tuple(sorted(operators)),
    fin_fields=fin_fields,
    needs_industry=needs_industry,
    needs_cap=needs_cap,
    suggested_warmup=suggested_warmup,
  )


class ExpressionFactor(Factor):
  """Vectorized :class:`Factor` backed by one or more ``alpha.lang`` expressions.

  Automatically infers required ``fin_fields``, ``needs_industry``, and suggested
  ``warmup_bars`` from the compiled expression AST.
  """

  def __init__(
    self,
    expressions: str | dict[str, str],
    *,
    name: str = "expr_factor",
    direction: int | dict[str, int] = 1,
    warmup_bars: int | None = None,
    fin_only_year: bool = False,
    fin_lookback_days: int | None = None,
    extra_fields: set[str] | None = None,
  ):
    if isinstance(expressions, str):
      self._is_single = True
      expr_map = {name: expressions}
    elif isinstance(expressions, dict):
      if not expressions:
        raise ValueError("ExpressionFactor requires at least one expression.")
      self._is_single = False
      expr_map = dict(expressions)
    else:
      raise TypeError(f"expressions must be str or dict[str, str], got {type(expressions).__name__}")

    self.name = name if self._is_single else "expression_batch"
    self.direction = direction
    self.fin_only_year = fin_only_year
    self.fin_lookback_days = fin_lookback_days

    self.compiled: dict[str, CompiledExpression] = {}
    all_fin: list[str] = []
    needs_ind = False
    max_warmup = 0
    for fname, expr_str in expr_map.items():
      ce = compile_expression(expr_str, name=fname, extra_fields=extra_fields)
      self.compiled[fname] = ce
      for f in ce.fin_fields:
        if f not in all_fin:
          all_fin.append(f)
      needs_ind = needs_ind or ce.needs_industry
      max_warmup = max(max_warmup, ce.suggested_warmup)

    self.fin_fields = all_fin if all_fin else None
    self.needs_industry = needs_ind
    self.warmup_bars = int(warmup_bars) if warmup_bars is not None else max_warmup
    self._industry_codes: np.ndarray | None = None
    self._benchmark_close: np.ndarray | None = None
    self._benchmark_open: np.ndarray | None = None

  def bind_context_metadata(
    self,
    *,
    industry_codes: np.ndarray | None = None,
    benchmark_close: np.ndarray | None = None,
    benchmark_open: np.ndarray | None = None,
  ) -> None:
    """Attach panel-aligned industry codes and benchmark curves before ``compute()``."""
    self._industry_codes = industry_codes
    self._benchmark_close = benchmark_close
    self._benchmark_open = benchmark_open

  def compute(self, panel: Panel) -> np.ndarray | dict[str, np.ndarray]:
    b_close = (
      self._benchmark_close[: panel.bars]
      if self._benchmark_close is not None and len(self._benchmark_close) >= panel.bars
      else None
    )
    b_open = (
      self._benchmark_open[: panel.bars]
      if self._benchmark_open is not None and len(self._benchmark_open) >= panel.bars
      else None
    )
    ctx = PanelExecContext(
      panel,
      industry_codes=self._industry_codes,
      benchmark_close=b_close,
      benchmark_open=b_open,
    )
    results: dict[str, np.ndarray] = {}
    with ctx, np.errstate(divide="ignore", invalid="ignore", over="ignore"):
      for fname, ce in self.compiled.items():
        raw = ce.fn(ctx)
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim == 0:
          arr = np.full((panel.n_objs, panel.bars), float(arr), dtype=np.float64)
        elif arr.ndim == 1 and arr.size == panel.n_objs * panel.bars:
          arr = arr.reshape(panel.n_objs, panel.bars)
        results[fname] = arr

    if self._is_single:
      return next(iter(results.values()))
    return results

  def finalize(self) -> dict[str, Any]:
    return {
      "expressions": {k: v.expression for k, v in self.compiled.items()},
      "inferred_fin_fields": list(self.fin_fields) if self.fin_fields else [],
      "needs_industry": self.needs_industry,
      "suggested_warmup_bars": self.warmup_bars,
    }


def parse_inline_expressions(exprs: list[str]) -> dict[str, str]:
  """Parse CLI ``-E/--expr`` arguments into ``{name: expression}``.

  Each item may be either ``name=expression`` (when the left-hand side is a
  simple identifier) or a bare ``expression`` (auto-named ``expr_01``, ...).
  """
  if not exprs:
    raise ValueError("No expressions provided.")
  out: dict[str, str] = {}
  auto_idx = 1
  for raw in exprs:
    text = raw.strip()
    if not text:
      continue
    m = _INLINE_NAME_RE.match(text)
    if m:
      fname, body = m.group(1), m.group(2).strip()
    else:
      fname = f"expr_{auto_idx:02d}" if len(exprs) > 1 else "expr_01"
      auto_idx += 1
      body = text
    if fname in out:
      raise ValueError(f"Duplicate expression factor name '{fname}'.")
    out[fname] = body
  if not out:
    raise ValueError("All provided expressions are empty.")
  return out


def load_expression_file(path: str) -> tuple[dict[str, str], dict[str, int]]:
  """Load a batch of factor expressions from a ``.txt`` or ``.json`` file.

  Supported formats:
    * ``.txt``: One expression per non-empty, non-comment (``#``) line.
      Lines may use ``name: expr``, ``name = expr``, or bare ``expr``.
    * ``.json``: Either ``{"name": "expr", ...}`` or a list of objects
      ``[{"name": "...", "expr": "...", "direction": 1}, ...]``.

  Returns:
    Tuple of ``(expressions_dict, directions_dict)``.
  """
  abs_path = os.path.abspath(path)
  if not os.path.isfile(abs_path):
    raise FileNotFoundError(f"Expression file not found: '{path}'")

  ext = os.path.splitext(abs_path)[1].lower()
  expressions: dict[str, str] = {}
  directions: dict[str, int] = {}

  if ext == ".json":
    with open(abs_path, "r", encoding="utf-8") as f:
      payload = json.load(f)
    if isinstance(payload, dict):
      for k, v in payload.items():
        if isinstance(v, str):
          expressions[str(k)] = v.strip()
          directions[str(k)] = 1
        elif isinstance(v, dict) and "expr" in v:
          expressions[str(k)] = str(v["expr"]).strip()
          directions[str(k)] = int(v.get("direction", 1))
        else:
          raise ValueError(
            f"Invalid JSON entry for '{k}' in '{path}': expected string or {{'expr': ...}}."
          )
    elif isinstance(payload, list):
      for i, item in enumerate(payload, 1):
        if isinstance(item, str):
          fname = f"expr_{i:02d}"
          expressions[fname] = item.strip()
          directions[fname] = 1
        elif isinstance(item, dict) and "expr" in item:
          fname = str(item.get("name") or f"expr_{i:02d}")
          expressions[fname] = str(item["expr"]).strip()
          directions[fname] = int(item.get("direction", 1))
        else:
          raise ValueError(
            f"Invalid JSON list entry #{i} in '{path}': expected string or {{'expr': ...}}."
          )
    else:
      raise ValueError(f"Unsupported JSON root type in '{path}': expected object or list.")
  else:
    with open(abs_path, "r", encoding="utf-8") as f:
      lines = f.readlines()
    auto_idx = 1
    for lineno, raw_line in enumerate(lines, 1):
      line = raw_line.strip()
      if not line or line.startswith("#"):
        continue
      m_eq = _INLINE_NAME_RE.match(line)
      # Support `name: expr` only when ':' is NOT part of a ternary `? ... : ...`
      colon_match = re.match(r"^([a-zA-Z_]\w*)\s*:\s*(.+)$", line)
      if colon_match and "?" not in colon_match.group(1):
        fname, body = colon_match.group(1), colon_match.group(2).strip()
      elif m_eq:
        fname, body = m_eq.group(1), m_eq.group(2).strip()
      else:
        fname = f"expr_{auto_idx:02d}"
        auto_idx += 1
        body = line
      if fname in expressions:
        raise ValueError(f"Duplicate factor name '{fname}' at {path}:{lineno}.")
      expressions[fname] = body
      directions[fname] = 1

  if not expressions:
    raise ValueError(f"No factor expressions found in '{path}'.")
  return expressions, directions


def format_operator_catalog() -> str:
  """Return a concise Markdown catalog of canonical operators, aliases, and fields."""
  lines = [
    "# mbt.factor Expression Specification (Canonical WorldQuant BRAIN + WQ101/GTJA191 Compatible)",
    "",
    "## Data Fields",
    "- **Forward-adjusted prices (for price-volume/momentum)**: `open`, `high`, `low`, `close`, `vwap`, `returns` (`ret`)",
    "- **Unadjusted raw prices (REQUIRED for valuation/market cap)**: `raw_open`, `raw_high`, `raw_low`, `raw_close` (`price`), `raw_vwap`",
    "- **Volume & Amount**: `volume` (`vol`), `amount`, `adv{N}` = `MA(volume, N)`, `adv_amt{N}` / `adtv{N}` = `MA(amount, N)`",
    "- **Market Cap & Shares**: `total_shares`, `tradable_shares`, `cap` (`raw_close * total_shares`), `float_cap` (`raw_close * tradable_shares`)",
    "- **Industry**: `indclass.industry` (Shenwan L1 industry code), `industry`",
    "- **Financials (auto-loaded with statutory disclosure lag)**: `f001`..`fNNN` (e.g. `f003` BVPS, `f205` ROE)",
    "- **Benchmark & Derived**: `benchmark_close` (`banchmarkindexclose`), `benchmark_open`, `dtm`, `dbm`, `tr`, `hd`, `ld`, `sequence`",
    "",
    "## Canonical Operators & Dialect Aliases",
    "| Category | Canonical (Recommended) | Compatible Aliases (WQ101 / GTJA191) |",
    "|---|---|---|",
    "| Rolling Mean / Sum | `ts_mean(x, d)`, `ts_sum(x, d)`, `ts_product(x, d)` | `ma`, `mean`, `sum`, `sumac`, `product`, `prod`, `ema`, `sma(x,n[,m])`, `dma` |",
    "| Rolling Dispersion | `ts_std_dev(x, d)`, `ts_variance(x, d)`, `ts_zscore(x, d)` | `stddev`, `std`, `var`, `ts_skewness`, `ts_kurtosis`, `ts_moment`, `ts_entropy` |",
    "| Rolling Delay / Diff | `ts_delay(x, d)`, `ts_delta(x, d)`, `ts_decay_linear(x, d)` | `delay`, `ref`, `delta`, `decay_linear`, `decaylinear`, `wma` |",
    "| Rolling Extremes / Rank | `ts_max(x, d)`, `ts_min(x, d)`, `ts_argmax(x, d)`, `ts_argmin(x, d)`, `ts_rank(x, d)`, `ts_quantile(x, d, q)` | `tsmax`, `tsmin`, `highday`, `lowday`, `tsrank`, `ts_min_max_diff` |",
    "| Rolling Bivariate / Reg | `ts_correlation(x, y, d)`, `ts_covariance(x, y, d)`, `ts_regbeta(x, y, d)`, `ts_regresi(x, y, d)` | `correlation`, `corr`, `covariance`, `cov`, `regbeta`, `regresi`, `slope`, `intercept` |",
    "| Rolling Counting / Scan | `ts_count(cond, d)`, `ts_sumif(x, d, cond)`, `barslast(cond)`, `barssince(cond)` | `count`, `sumif`, `sumbars`, `cross`, `longcross`, `SELF` recursion |",
    "| Cross-Sectional | `rank(x)`, `zscore(x)`, `scale(x[, k])` | `bins(x, n)` |",
    "| Industry Grouped | `group_neutralize(x, g)`, `group_rank(x, g)`, `group_zscore(x, g)` | `indneutralize(x, indclass.industry)` |",
    "| Math & Control | `cond ? a : b`, `if(cond, a, b)`, `abs`, `log`, `log1p`, `sqrt`, `sign`, `signedpower`, `power`, `min`, `max`, `clip` | `pasteurize` (`purify`), `tail` (`truncate`), `filter`, `^` (power), `&&`, `||` |",
    "",
    "## Guardrails (Fail-Fast)",
    "- Use `^` instead of `**` for power.",
    "- Never divide per-share financials (`f001`-`f006`) by forward-adjusted `close`; always use `raw_close`.",
    "- Lookahead operators (`fret`, `backset`) and negative lags (`ts_delay(x, -1)`) are rejected.",
  ]
  return "\n".join(lines)
