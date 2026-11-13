# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Strategy validator and AST anti-pattern linting."""
from __future__ import annotations

import ast
import logging

logger = logging.getLogger(__name__)


class PriceCrossDimensionVisitor(ast.NodeVisitor):
  """Detects cross-dimensional arithmetic between forward-adjusted and raw position prices.

  Specifically checks for patterns where forward-adjusted price indicators (`ctx.close`, `ctx.open`,
  `ctx.high`, `ctx.low`) are mixed in arithmetic operations (subtraction, division) with position
  execution or acquisition prices (`pos.price`, `pos.cost_price`, `positions[...].price`).
  """

  def __init__(self, filename: str = "<strategy>"):
    self.filename = filename
    self.warnings: list[str] = []

  def visit_FunctionDef(self, node: ast.FunctionDef):
    self._check_function_scope(node)
    self.generic_visit(node)

  def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
    self._check_function_scope(node)
    self.generic_visit(node)

  def _check_function_scope(self, func_node: ast.FunctionDef | ast.AsyncFunctionDef):
    # Track variables assigned from forward-adjusted prices (ctx.close, ctx.open, etc.)
    fw_vars: set[str] = set()
    # Track variables assigned from position prices (pos.price, pos.cost_price, etc.)
    pos_price_vars: set[str] = set()

    for stmt in ast.walk(func_node):
      if isinstance(stmt, ast.Assign):
        val = stmt.value
        if self._is_forward_adjusted_expr(val):
          for target in stmt.targets:
            if isinstance(target, ast.Name):
              fw_vars.add(target.id)
        elif self._is_pos_price_expr(val):
          for target in stmt.targets:
            if isinstance(target, ast.Name):
              pos_price_vars.add(target.id)

    for stmt in ast.walk(func_node):
      if isinstance(stmt, ast.BinOp) and isinstance(stmt.op, (ast.Sub, ast.Div)):
        left_is_fw = self._is_forward_adjusted_expr(stmt.left) or (
          isinstance(stmt.left, ast.Name) and stmt.left.id in fw_vars
        )
        right_is_pos = self._is_pos_price_expr(stmt.right) or (
          isinstance(stmt.right, ast.Name) and stmt.right.id in pos_price_vars
        )

        left_is_pos = self._is_pos_price_expr(stmt.left) or (
          isinstance(stmt.left, ast.Name) and stmt.left.id in pos_price_vars
        )
        right_is_fw = self._is_forward_adjusted_expr(stmt.right) or (
          isinstance(stmt.right, ast.Name) and stmt.right.id in fw_vars
        )

        if (left_is_fw and right_is_pos) or (left_is_pos and right_is_fw):
          lineno = getattr(stmt, "lineno", "?")
          msg = (
            f"{self.filename}:{lineno}: [ANTI-PATTERN DETECTED] Cross-dimensional price arithmetic detected! "
            f"Mixing forward-adjusted price ('ctx.close/open') with raw position price ('pos.price/cost_price'). "
            f"Forward-adjusted prices and raw execution prices have mismatched baselines. "
            f"Upon stock splits or bonus shares (e.g. 10送10), this causes catastrophic false drawdowns (-50% false crash). "
            f"Fix: Use 'ctx.holding_ror(obj)' or 'pos.holding_ror' for holding returns/stop-loss, "
            f"or 'pos.adj_cost_price' if a forward-adjusted cost baseline is required."
          )
          if msg not in self.warnings:
            self.warnings.append(msg)
            logger.warning(msg)

  def _is_forward_adjusted_expr(self, node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
      return self._is_forward_adjusted_expr(node.value)
    if isinstance(node, ast.Attribute):
      if (
        node.attr in ("close", "open", "high", "low")
        and isinstance(node.value, ast.Name)
        and node.value.id == "ctx"
      ):
        return True
    return False

  def _is_pos_price_expr(self, node: ast.AST) -> bool:
    if isinstance(node, ast.Subscript):
      return self._is_pos_price_expr(node.value)
    if isinstance(node, ast.Attribute):
      if node.attr in ("price", "cost_price", "raw_cost_price", "raw_price", "raw_mark_price"):
        val = node.value
        if isinstance(val, ast.Name) and any(kw in val.id.lower() for kw in ("pos", "cost")):
          return True
        if isinstance(val, ast.Subscript):
          sub_val = val.value
          if isinstance(sub_val, ast.Attribute) and sub_val.attr == "positions":
            return True
        if isinstance(val, ast.Call):
          func = val.func
          if isinstance(func, ast.Attribute) and func.attr == "position":
            return True
    return False


def check_strategy_source(
  source: str,
  filename: str = "<strategy>",
  fail_fast: bool = False,
) -> list[str]:
  """Scan strategy source code for anti-patterns and cross-dimensional price misuse.

  Args:
    source: Python strategy source code.
    filename: Path or identifier for logging.
    fail_fast: If True and an anti-pattern is detected, raises ValueError instead of only warning.

  Returns:
    List of warning strings.
  """
  try:
    tree = ast.parse(source, filename=filename)
  except Exception as e:
    logger.debug(f"Could not parse strategy AST for validation: {e}")
    return []

  visitor = PriceCrossDimensionVisitor(filename=filename)
  visitor.visit(tree)
  if fail_fast and visitor.warnings:
    raise ValueError(visitor.warnings[0])
  return visitor.warnings
