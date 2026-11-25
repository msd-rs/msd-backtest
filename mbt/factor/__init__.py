# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""mbt.factor: Agent-First alpha factor evaluation subsystem."""

from .evaluator import evaluate_factor, evaluate_panel
from .expr import (
  CompiledExpression,
  ExpressionFactor,
  PanelExecContext,
  compile_expression,
  format_operator_catalog,
  load_expression_file,
  parse_inline_expressions,
)
from .factor import (
  Factor,
  check_factor_causality,
  check_factor_source,
  load_factor,
  normalize_factor_output,
)
from .metrics import (
  ICSummary,
  QuantileAnalysis,
  QuantileMetrics,
  compute_factor_correlation_matrix,
  compute_ic_series,
  compute_quantile_analyses,
  compute_quantile_analysis,
  compute_rank_ic_series,
  compute_turnover_and_autocorr,
  summarize_ic,
)
from .preprocess import (
  compute_log_mktcap,
  neutralize,
  preprocess_factor,
  standardize,
  winsorize,
)
from .report import FactorReport, MultiFactorReport, export_report
from .returns import ForwardReturnSet, build_tradability_masks, compute_forward_returns

__all__ = [
  "CompiledExpression",
  "ExpressionFactor",
  "Factor",
  "FactorReport",
  "ForwardReturnSet",
  "ICSummary",
  "MultiFactorReport",
  "PanelExecContext",
  "QuantileAnalysis",
  "QuantileMetrics",
  "build_tradability_masks",
  "check_factor_causality",
  "check_factor_source",
  "compile_expression",
  "compute_factor_correlation_matrix",
  "compute_forward_returns",
  "compute_ic_series",
  "compute_log_mktcap",
  "compute_quantile_analyses",
  "compute_quantile_analysis",
  "compute_rank_ic_series",
  "compute_turnover_and_autocorr",
  "evaluate_factor",
  "evaluate_panel",
  "export_report",
  "format_operator_catalog",
  "load_expression_file",
  "load_factor",
  "neutralize",
  "normalize_factor_output",
  "parse_inline_expressions",
  "preprocess_factor",
  "standardize",
  "summarize_ic",
  "winsorize",
]
