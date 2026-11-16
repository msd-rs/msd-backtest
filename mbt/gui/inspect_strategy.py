# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Strategy file inspection utilities for the mbt GUI."""

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

from mbt.gui.config import RUN_MODE_PORTFOLIO, RUN_MODE_SELECT, RUN_MODE_TIMING
from mbt.select.selector import Selector, StaticSelector
from mbt.strategy import PortfolioStrategy, Strategy


@dataclasses.dataclass
class ParamSpec:
  """Metadata for a single strategy/selector __init__ parameter."""

  name: str
  default_str: str
  type_label: str


@dataclasses.dataclass
class StrategyMetadata:
  """Detected capabilities, default mode, and parameters of a strategy file."""

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

  def available_modes(self) -> list[str]:
    """Return supported run modes in logical order for this file."""
    modes: list[str] = []
    if self.has_portfolio_strategy:
      modes.append(RUN_MODE_PORTFOLIO)
    if self.has_strategy:
      if not self.has_portfolio_strategy:
        modes.append(RUN_MODE_TIMING)
      modes.append(RUN_MODE_PORTFOLIO)
    if self.has_selector:
      modes.append(RUN_MODE_SELECT)
    seen: list[str] = []
    for m in modes:
      if m not in seen:
        seen.append(m)
    return seen or [RUN_MODE_TIMING]

  def params_for_mode(self, mode: str) -> list[ParamSpec]:
    """Return merged custom __init__ parameter specs relevant to a run mode."""
    if mode == RUN_MODE_SELECT:
      return list(self.selector_params)
    merged: list[ParamSpec] = []
    seen: set[str] = set()
    for p in self.strategy_params + self.selector_params:
      if p.name not in seen:
        seen.add(p.name)
        merged.append(p)
    return merged


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
      inherited = _classify(bname, seen)
      if inherited is not None:
        if inherited == "timing" and _class_has_portfolio_true(cls_node):
          return "portfolio"
        return inherited
    return None

  selector_node: ast.ClassDef | None = None
  strategy_node: ast.ClassDef | None = None
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

  if selector_node is None and strategy_node is None:
    return None

  has_selector = selector_node is not None
  has_strategy = strategy_node is not None
  if has_portfolio:
    default_mode = RUN_MODE_PORTFOLIO
  elif has_strategy:
    default_mode = RUN_MODE_TIMING
  else:
    default_mode = RUN_MODE_SELECT

  mod_doc = ast.get_docstring(tree) or ""
  cls_doc = ast.get_docstring(strategy_node or selector_node) or ""
  doc = mod_doc or cls_doc
  first_line = doc.splitlines()[0].strip() if doc else ""

  return StrategyMetadata(
    path=abs_path,
    has_selector=has_selector,
    has_strategy=has_strategy,
    has_portfolio_strategy=has_portfolio,
    default_mode=default_mode,
    selector_name=selector_node.name if selector_node else None,
    strategy_name=strategy_node.name if strategy_node else None,
    selector_params=_extract_init_params_ast(selector_node) if selector_node else [],
    strategy_params=_extract_init_params_ast(strategy_node) if strategy_node else [],
    docstring=first_line,
  )


def _inspect_via_import(abs_path: str) -> StrategyMetadata:
  dir_path = os.path.dirname(abs_path)
  if dir_path not in sys.path:
    sys.path.insert(0, dir_path)

  mod_name = f"_mbt_gui_inspect_{uuid.uuid4().hex}"
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
  for obj in vars(module).values():
    if not isinstance(obj, type):
      continue
    if obj.__module__ != module.__name__:
      continue
    if issubclass(obj, Selector) and obj not in (Selector, StaticSelector):
      selector_cls = obj
    if issubclass(obj, Strategy) and obj not in (Strategy, PortfolioStrategy):
      strategy_cls = obj

  if selector_cls is None and strategy_cls is None:
    raise ValueError(f"No Strategy or Selector subclass found in '{abs_path}'.")

  has_selector = selector_cls is not None
  has_strategy = strategy_cls is not None
  has_portfolio = bool(
    strategy_cls is not None
    and (issubclass(strategy_cls, PortfolioStrategy) or getattr(strategy_cls, "portfolio", False))
  )

  if has_portfolio:
    default_mode = RUN_MODE_PORTFOLIO
  elif has_strategy:
    default_mode = RUN_MODE_TIMING
  else:
    default_mode = RUN_MODE_SELECT

  doc = inspect.getdoc(module) or inspect.getdoc(strategy_cls or selector_cls) or ""
  first_line = doc.splitlines()[0].strip() if doc else ""

  return StrategyMetadata(
    path=abs_path,
    has_selector=has_selector,
    has_strategy=has_strategy,
    has_portfolio_strategy=has_portfolio,
    default_mode=default_mode,
    selector_name=selector_cls.__name__ if selector_cls else None,
    strategy_name=strategy_cls.__name__ if strategy_cls else None,
    selector_params=_extract_init_params_runtime(selector_cls) if selector_cls else [],
    strategy_params=_extract_init_params_runtime(strategy_cls) if strategy_cls else [],
    docstring=first_line,
  )


def inspect_strategy_file(file_path: str | Path) -> StrategyMetadata:
  """Inspect a strategy Python file and extract its classes, default mode, and parameters.

  Prefers side-effect-free AST inspection so importing works even when top-level
  imports reference offline MSD services; falls back to module import for
  cross-file base class hierarchies.

  Raises:
    FileNotFoundError: If ``file_path`` does not exist.
    ValueError: If the file has invalid Python syntax or no Strategy/Selector class.
  """
  abs_path = os.path.abspath(str(file_path))
  if not os.path.isfile(abs_path):
    raise FileNotFoundError(f"Strategy file not found: '{abs_path}'")
  if not abs_path.endswith(".py"):
    raise ValueError(f"Expected a Python (.py) strategy file, got '{abs_path}'.")

  source = Path(abs_path).read_text(encoding="utf-8")
  try:
    ast_meta = _inspect_via_ast(abs_path, source)
  except SyntaxError as exc:
    raise ValueError(f"Syntax error in '{abs_path}': {exc}") from exc

  if ast_meta is not None:
    return ast_meta
  return _inspect_via_import(abs_path)
