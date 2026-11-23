# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Strategy, selector, and factor file inspection utilities shared across mbt."""

from __future__ import annotations

import ast
import dataclasses
import importlib.util
import inspect
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from mbt.select.selector import Selector, StaticSelector
from mbt.strategy import PortfolioStrategy, Strategy

RUN_MODE_SELECT = "select"
RUN_MODE_TIMING = "timing"
RUN_MODE_PORTFOLIO = "portfolio"
RUN_MODE_FACTOR = "factor"
VALID_RUN_MODES = (
  RUN_MODE_SELECT,
  RUN_MODE_TIMING,
  RUN_MODE_PORTFOLIO,
  RUN_MODE_FACTOR,
)


@dataclasses.dataclass
class ParamSpec:
  """Metadata for a single strategy/selector/factor ``__init__`` parameter."""

  name: str
  default_str: str
  type_label: str

  def to_dict(self) -> dict[str, str]:
    """Serialize parameter metadata to a JSON-friendly dictionary."""
    return {
      "name": self.name,
      "default_str": self.default_str,
      "type_label": self.type_label,
    }


@dataclasses.dataclass
class StrategyMetadata:
  """Detected capabilities, default mode, and parameters of a strategy or factor file."""

  path: str
  has_selector: bool
  has_strategy: bool
  has_portfolio_strategy: bool
  default_mode: str
  selector_name: str | None = None
  strategy_name: str | None = None
  selector_params: list[ParamSpec] = dataclasses.field(default_factory=list)
  strategy_params: list[ParamSpec] = dataclasses.field(default_factory=list)
  docstring: str = ""
  has_factor: bool = False
  factor_name: str | None = None
  factor_params: list[ParamSpec] = dataclasses.field(default_factory=list)
  is_expression_file: bool = False
  expression_names: list[str] = dataclasses.field(default_factory=list)
  author: str = ""

  def available_modes(self) -> list[str]:
    """Return supported run modes in logical order for this file."""
    modes: list[str] = []
    if self.has_portfolio_strategy:
      modes.append(RUN_MODE_PORTFOLIO)
    if self.has_strategy:
      if not self.has_portfolio_strategy:
        modes.append(RUN_MODE_TIMING)
      modes.append(RUN_MODE_PORTFOLIO)
    if self.has_factor:
      modes.append(RUN_MODE_FACTOR)
    if self.has_selector:
      modes.append(RUN_MODE_SELECT)
    seen: list[str] = []
    for m in modes:
      if m not in seen:
        seen.append(m)
    return seen or [RUN_MODE_TIMING]

  def params_for_mode(self, mode: str) -> list[ParamSpec]:
    """Return merged custom ``__init__`` parameter specs relevant to a run mode."""
    if mode == RUN_MODE_SELECT:
      return list(self.selector_params)
    if mode == RUN_MODE_FACTOR:
      return list(self.factor_params)
    merged: list[ParamSpec] = []
    seen: set[str] = set()
    for p in self.strategy_params + self.selector_params:
      if p.name not in seen:
        seen.add(p.name)
        merged.append(p)
    return merged

  def to_dict(self) -> dict[str, Any]:
    """Serialize full metadata to a JSON-friendly dictionary."""
    return {
      "path": self.path,
      "has_selector": self.has_selector,
      "has_strategy": self.has_strategy,
      "has_portfolio_strategy": self.has_portfolio_strategy,
      "has_factor": self.has_factor,
      "default_mode": self.default_mode,
      "available_modes": self.available_modes(),
      "selector_name": self.selector_name,
      "strategy_name": self.strategy_name,
      "factor_name": self.factor_name,
      "selector_params": [p.to_dict() for p in self.selector_params],
      "strategy_params": [p.to_dict() for p in self.strategy_params],
      "factor_params": [p.to_dict() for p in self.factor_params],
      "is_expression_file": self.is_expression_file,
      "expression_names": list(self.expression_names),
      "docstring": self.docstring,
      "author": self.author,
    }


def _format_default(val: Any) -> str:
  if val is inspect.Parameter.empty or val is None:
    return ""
  if isinstance(val, bool):
    return "true" if val else "false"
  return str(val)


def _format_annotation(ann: Any) -> str:
  if ann is inspect.Parameter.empty:
    return ""
  if hasattr(ann, "__name__"):
    return ann.__name__
  return str(ann).replace("typing.", "")


def _extract_init_params_runtime(cls: type) -> list[ParamSpec]:
  if cls.__init__ is object.__init__:
    return []
  sig = inspect.signature(cls.__init__)
  params: list[ParamSpec] = []
  for name, param in sig.parameters.items():
    if name == "self":
      continue
    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
      continue
    params.append(
      ParamSpec(
        name=name,
        default_str=_format_default(param.default),
        type_label=_format_annotation(param.annotation),
      )
    )
  return params


def _ast_base_name(node: ast.expr) -> str:
  if isinstance(node, ast.Name):
    return node.id
  if isinstance(node, ast.Attribute):
    return node.attr
  return ""


def _ast_default_str(node: ast.expr | None) -> str:
  if node is None:
    return ""
  try:
    val = ast.literal_eval(node)
    return _format_default(val)
  except Exception:
    return ast.unparse(node)


def _extract_init_params_ast(cls_node: ast.ClassDef) -> list[ParamSpec]:
  init_fn: ast.FunctionDef | None = None
  for item in cls_node.body:
    if isinstance(item, ast.FunctionDef) and item.name == "__init__":
      init_fn = item
      break
  if init_fn is None:
    return []

  args = init_fn.args
  pos_args = list(args.posonlyargs) + list(args.args)
  # Align defaults from the right
  n_pad = len(pos_args) - len(args.defaults)
  padded_defaults: list[ast.expr | None] = [None] * max(0, n_pad) + list(args.defaults)

  params: list[ParamSpec] = []
  for arg_node, default_node in zip(pos_args, padded_defaults):
    if arg_node.arg == "self":
      continue
    type_label = ast.unparse(arg_node.annotation) if arg_node.annotation else ""
    params.append(
      ParamSpec(
        name=arg_node.arg,
        default_str=_ast_default_str(default_node),
        type_label=type_label,
      )
    )

  for arg_node, default_node in zip(args.kwonlyargs, args.kw_defaults):
    type_label = ast.unparse(arg_node.annotation) if arg_node.annotation else ""
    params.append(
      ParamSpec(
        name=arg_node.arg,
        default_str=_ast_default_str(default_node),
        type_label=type_label,
      )
    )
  return params


def _class_has_portfolio_true(cls_node: ast.ClassDef) -> bool:
  for item in cls_node.body:
    if isinstance(item, ast.Assign):
      for target in item.targets:
        if isinstance(target, ast.Name) and target.id == "portfolio":
          try:
            return bool(ast.literal_eval(item.value))
          except Exception:
            pass
    elif isinstance(item, ast.AnnAssign):
      if isinstance(item.target, ast.Name) and item.target.id == "portfolio" and item.value:
        try:
          return bool(ast.literal_eval(item.value))
        except Exception:
          pass
  return False


def _extract_ast_author(tree: ast.Module) -> str:
  for item in tree.body:
    if isinstance(item, ast.Assign):
      for target in item.targets:
        if isinstance(target, ast.Name) and target.id == "M_AUTHOR":
          try:
            val = ast.literal_eval(item.value)
            return str(val).strip()
          except Exception:
            pass
    elif isinstance(item, ast.AnnAssign):
      if isinstance(item.target, ast.Name) and item.target.id == "M_AUTHOR" and item.value:
        try:
          val = ast.literal_eval(item.value)
          return str(val).strip()
        except Exception:
          pass
  return ""


def _resolve_default_mode(
  *,
  has_portfolio: bool,
  has_strategy: bool,
  has_factor: bool,
) -> str:
  if has_portfolio:
    return RUN_MODE_PORTFOLIO
  if has_strategy:
    return RUN_MODE_TIMING
  if has_factor:
    return RUN_MODE_FACTOR
  return RUN_MODE_SELECT


def _inspect_via_ast(abs_path: str, source: str) -> StrategyMetadata | None:
  tree = ast.parse(source, filename=abs_path)
  classes = {
    node.name: node for node in tree.body if isinstance(node, ast.ClassDef)
  }

  def _classify(name: str, seen: set[str] | None = None) -> str | None:
    if seen is None:
      seen = set()
    if name in seen or name not in classes:
      return None
    seen.add(name)
    cls_node = classes[name]
    for base in cls_node.bases:
      bname = _ast_base_name(base)
      if bname == "Selector":
        return "selector"
      if bname == "PortfolioStrategy":
        return "portfolio"
      if bname == "Strategy":
        return "portfolio" if _class_has_portfolio_true(cls_node) else "timing"
      if bname in ("Factor", "ExpressionFactor"):
        return "factor"
      inherited = _classify(bname, seen)
      if inherited is not None:
        if inherited == "timing" and _class_has_portfolio_true(cls_node):
          return "portfolio"
        return inherited
    return None

  selector_node: ast.ClassDef | None = None
  strategy_node: ast.ClassDef | None = None
  factor_node: ast.ClassDef | None = None
  has_portfolio = False

  for cname, cnode in classes.items():
    kind = _classify(cname)
    if kind == "selector":
      selector_node = cnode
    elif kind == "portfolio":
      strategy_node = cnode
      has_portfolio = True
    elif kind == "timing":
      strategy_node = cnode
    elif kind == "factor":
      if factor_node is None:
        factor_node = cnode

  if selector_node is None and strategy_node is None and factor_node is None:
    return None

  has_selector = selector_node is not None
  has_strategy = strategy_node is not None
  has_factor = factor_node is not None
  default_mode = _resolve_default_mode(
    has_portfolio=has_portfolio,
    has_strategy=has_strategy,
    has_factor=has_factor,
  )

  mod_doc = ast.get_docstring(tree) or ""
  target_node = strategy_node or factor_node or selector_node
  assert target_node is not None
  cls_doc = ast.get_docstring(target_node) or ""
  doc = mod_doc or cls_doc
  first_line = doc.splitlines()[0].strip() if doc else ""

  return StrategyMetadata(
    path=abs_path,
    has_selector=has_selector,
    has_strategy=has_strategy,
    has_portfolio_strategy=has_portfolio,
    has_factor=has_factor,
    default_mode=default_mode,
    selector_name=selector_node.name if selector_node else None,
    strategy_name=strategy_node.name if strategy_node else None,
    factor_name=factor_node.name if factor_node else None,
    selector_params=_extract_init_params_ast(selector_node) if selector_node else [],
    strategy_params=_extract_init_params_ast(strategy_node) if strategy_node else [],
    factor_params=_extract_init_params_ast(factor_node) if factor_node else [],
    docstring=first_line,
    author=_extract_ast_author(tree),
  )


def _inspect_via_import(abs_path: str) -> StrategyMetadata:
  from mbt.factor.expr import ExpressionFactor
  from mbt.factor.factor import Factor

  dir_path = os.path.dirname(abs_path)
  if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

  mod_name = f"_mbt_inspect_{uuid.uuid4().hex}"
  spec = importlib.util.spec_from_file_location(mod_name, abs_path)
  if spec is None or spec.loader is None:
    raise ValueError(f"Failed to create module spec for '{abs_path}'.")

  module = importlib.util.module_from_spec(spec)
  try:
    spec.loader.exec_module(module)
  except Exception as exc:
    raise ValueError(f"Failed to import '{abs_path}': {exc}") from exc
  finally:
    sys.modules.pop(mod_name, None)

  selector_cls: type | None = None
  strategy_cls: type | None = None
  factor_cls: type | None = None
  for obj in vars(module).values():
    if not isinstance(obj, type):
      continue
    if obj.__module__ != module.__name__:
      continue
    if issubclass(obj, Selector) and obj not in (Selector, StaticSelector):
      selector_cls = obj
    if issubclass(obj, Strategy) and obj not in (Strategy, PortfolioStrategy):
      strategy_cls = obj
    if issubclass(obj, Factor) and obj not in (Factor, ExpressionFactor):
      if factor_cls is None:
        factor_cls = obj

  if selector_cls is None and strategy_cls is None and factor_cls is None:
    raise ValueError(f"No Strategy, Selector, or Factor subclass found in '{abs_path}'.")

  has_selector = selector_cls is not None
  has_strategy = strategy_cls is not None
  has_factor = factor_cls is not None
  has_portfolio = bool(
    strategy_cls is not None
    and (issubclass(strategy_cls, PortfolioStrategy) or getattr(strategy_cls, "portfolio", False))
  )

  default_mode = _resolve_default_mode(
    has_portfolio=has_portfolio,
    has_strategy=has_strategy,
    has_factor=has_factor,
  )

  primary_cls = strategy_cls or factor_cls or selector_cls
  doc = inspect.getdoc(module) or (inspect.getdoc(primary_cls) if primary_cls else "") or ""
  first_line = doc.splitlines()[0].strip() if doc else ""
  author = str(getattr(module, "M_AUTHOR", "") or "").strip()

  return StrategyMetadata(
    path=abs_path,
    has_selector=has_selector,
    has_strategy=has_strategy,
    has_portfolio_strategy=has_portfolio,
    has_factor=has_factor,
    default_mode=default_mode,
    selector_name=selector_cls.__name__ if selector_cls else None,
    strategy_name=strategy_cls.__name__ if strategy_cls else None,
    factor_name=factor_cls.__name__ if factor_cls else None,
    selector_params=_extract_init_params_runtime(selector_cls) if selector_cls else [],
    strategy_params=_extract_init_params_runtime(strategy_cls) if strategy_cls else [],
    factor_params=_extract_init_params_runtime(factor_cls) if factor_cls else [],
    docstring=first_line,
    author=author,
  )


def _inspect_expression_file(abs_path: str) -> StrategyMetadata:
  from mbt.factor.expr import load_expression_file

  exprs, _ = load_expression_file(abs_path)
  names = list(exprs.keys())
  preview = ", ".join(names[:5])
  if len(names) > 5:
    preview += f", ... (+{len(names) - 5})"
  return StrategyMetadata(
    path=abs_path,
    has_selector=False,
    has_strategy=False,
    has_portfolio_strategy=False,
    has_factor=True,
    default_mode=RUN_MODE_FACTOR,
    factor_name=Path(abs_path).stem,
    docstring=f"Expression batch ({len(names)}): {preview}",
    is_expression_file=True,
    expression_names=names,
  )


def inspect_strategy_file(file_path: str | Path) -> StrategyMetadata:
  """Inspect a strategy, selector, or factor file and extract capabilities and parameters.

  Supports Python modules (``.py``) defining ``Strategy``, ``PortfolioStrategy``,
  ``Selector``, or ``Factor`` subclasses, as well as factor expression batch files
  (``.txt``, ``.json``). Prefers side-effect-free AST inspection for ``.py`` files
  and falls back to runtime import for cross-file class hierarchies.

  Raises:
    FileNotFoundError: If ``file_path`` does not exist.
    ValueError: If the file has invalid syntax, unsupported extension, or no valid class/expressions.
  """
  abs_path = os.path.realpath(os.path.abspath(str(file_path)))
  if not os.path.isfile(abs_path):
    raise FileNotFoundError(f"Strategy or factor file not found: '{abs_path}'")

  ext = os.path.splitext(abs_path)[1].lower()
  if ext in (".txt", ".json"):
    return _inspect_expression_file(abs_path)
  if ext != ".py":
    raise ValueError(
      f"Expected a .py, .txt, or .json strategy/factor file, got '{abs_path}'."
    )

  source = Path(abs_path).read_text(encoding="utf-8")
  try:
    ast_meta = _inspect_via_ast(abs_path, source)
  except SyntaxError as exc:
    raise ValueError(f"Syntax error in '{abs_path}': {exc}") from exc

  if ast_meta is not None:
    return ast_meta
  return _inspect_via_import(abs_path)
