from mbt import Strategy
from typing import IO


class RunRequest:
  def __init__(
    self,
    strategy: str | IO | Strategy,
    symbols: list[str],
    /,
    start: str,
    end: str,
    args: list | None = None,
  ):
    self.strategy = RunRequest.load_strategy(strategy, args=args)
    self.symbols = symbols
    self.start = start
    self.end = end

  @staticmethod
  def load_strategy(
    strategy: str | IO | Strategy, args: list | None = None
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
            if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
              return obj(*args) if args is not None else obj()

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
      exec(strategy, loc)
      for obj in loc.values():
        if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy:
          if args is None:
            return obj()
          else:
            return obj(*args)
      raise ValueError("No Strategy subclass found in source code")

    raise ValueError("Invalid strategy type")

  @staticmethod
  def from_json(
    data: str | IO,
    /,
    symbols: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    args: list | None = None,
  ) -> "RunRequest":
    import json

    if hasattr(data, "read"):
      content = data.read()
    else:
      content = data

    if isinstance(content, bytes):
      content = content.decode("utf-8")

    config = json.loads(content)

    return RunRequest(
      config.get("strategy"),
      symbols if symbols is not None else config.get("symbols"),
      start=start if start is not None else config.get("start"),
      end=end if end is not None else config.get("end"),
      args=args if args is not None else config.get("args"),
    )
