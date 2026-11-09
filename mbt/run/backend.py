from mbt.strategy import Strategy, PortfolioStrategy
from typing import Any, IO
import inspect


def parse_kv_args(args: list[str] | dict[str, Any] | None) -> dict[str, Any]:
  """Parse a list of 'key=val' strings into a typed dictionary."""
  if args is None:
    return {}
  if isinstance(args, dict):
    return dict(args)

  def try_val(s: str) -> Any:
    low = s.lower()
    if low in ("true", "t", "yes", "y", "on"):
      return True
    if low in ("false", "f", "no", "n", "off"):
      return False
    try:
      return int(s)
    except ValueError:
      pass
    try:
      return float(s)
    except ValueError:
      pass
    return s

  kwargs: dict[str, Any] = {}
  for item in args:
    if isinstance(item, str) and "=" in item:
      k, v = item.split("=", 1)
      kwargs[k.strip()] = try_val(v.strip())
  return kwargs


def instantiate_class(cls: type, args: list | dict[str, Any] | None = None):
  """Instantiate `cls` with either `key=val` kwargs (filtered by signature) or positional args."""
  if not args:
    return cls()

  if isinstance(args, dict) or (
    isinstance(args, list)
    and all(isinstance(x, str) and "=" in x for x in args)
  ):
    kwargs = parse_kv_args(args)
    sig = inspect.signature(cls.__init__)
    has_var_kw = any(
      p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
      return cls(**kwargs)
    accepted = {
      k: v for k, v in kwargs.items() if k in sig.parameters and k != "self"
    }
    return cls(**accepted)

  return cls(*args)


class RunRequest:
  def __init__(
    self,
    strategy: str | IO | Strategy,
    objs: list[str],
    /,
    start: str,
    end: str,
    args: list | dict[str, Any] | None = None,
    fin_fields: list[str] | str | None = None,
    fin_only_year: bool = False,
  ):
    self.strategy = RunRequest.load_strategy(strategy, args=args)
    self.objs = objs
    self.start = start
    self.end = end
    self.fin_fields = fin_fields
    self.fin_only_year = fin_only_year

  @staticmethod
  def load_strategy(
    strategy: str | IO | Strategy, args: list | dict[str, Any] | None = None
  ) -> Strategy:
    import sys
    import os
    import importlib.util

    if isinstance(strategy, Strategy):
      return strategy

    if hasattr(strategy, "read"):
      # input is file-like object, read it and import it and return Strategy class instance in it
      if hasattr(strategy, "name") and strategy.name.endswith(".py"):
        path = os.path.abspath(strategy.name)
        dir_path = os.path.dirname(path)
        if dir_path not in sys.path:
          sys.path.insert(0, dir_path)

        module_name = os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
          module = importlib.util.module_from_spec(spec)
          spec.loader.exec_module(module)
          for obj in vars(module).values():
            if (
              isinstance(obj, type)
              and issubclass(obj, Strategy)
              and obj not in (Strategy, PortfolioStrategy)
            ):
              return instantiate_class(obj, args)

      content = strategy.read()
      if isinstance(content, bytes):
        content = content.decode("utf-8")
      return RunRequest.load_strategy(content, args=args)

    if isinstance(strategy, str):
      # input is source code, import it and return Strategy class instance in it
      loc = {}
      # Use the same dictionary for globals and locals to avoid NameError in class methods
      # also include Strategy class in the loc to avoid NameError
      loc["Strategy"] = Strategy
      loc["PortfolioStrategy"] = PortfolioStrategy
      exec(strategy, loc)
      for obj in loc.values():
        if (
          isinstance(obj, type)
          and issubclass(obj, Strategy)
          and obj not in (Strategy, PortfolioStrategy)
        ):
          return instantiate_class(obj, args)
      raise ValueError("No Strategy subclass found in source code")

    raise ValueError("Invalid strategy type")

  @staticmethod
  def from_json(
    data: str | IO[str] | IO[bytes] | bytes,
    /,
    objs: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    args: list | None = None,
    fin_fields: list[str] | str | None = None,
    fin_only_year: bool | None = None,
  ) -> "RunRequest":
    import json

    if hasattr(data, "read"):
      content = data.read()
    else:
      content = data

    config = json.loads(content) #type: ignore


    return RunRequest(
      config.get("strategy"),
      objs if objs is not None else config.get("objs"),
      start=start if start is not None else config.get("start"),
      end=end if end is not None else config.get("end"),
      args=args if args is not None else config.get("args"),
      fin_fields=fin_fields if fin_fields is not None else config.get("fin_fields"),
      fin_only_year=fin_only_year if fin_only_year is not None else config.get("fin_only_year", False),
    )
