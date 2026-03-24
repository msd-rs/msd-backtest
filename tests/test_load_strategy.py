import io
from mbt.run.backend import RunRequest
from mbt import Strategy


def test_load_strategy_from_str():
  code = """
from mbt import Strategy
class MyStrategy(Strategy):
    def execute(self, ctx):
        pass
"""
  strategy = RunRequest.load_strategy(code)
  assert isinstance(strategy, Strategy)
  assert strategy.__class__.__name__ == "MyStrategy"


def test_load_strategy_from_io():
  code = """
from mbt import Strategy
class MyStrategyIO(Strategy):
    def execute(self, ctx):
        pass
"""
  f = io.StringIO(code)
  strategy = RunRequest.load_strategy(f)
  assert isinstance(strategy, Strategy)
  assert strategy.__class__.__name__ == "MyStrategyIO"


def test_load_strategy_from_instance():
  class ManualStrategy(Strategy):
    def execute(self, ctx):
      pass

  inst = ManualStrategy()
  strategy = RunRequest.load_strategy(inst)
  assert strategy is inst


def test_load_strategy_no_subclass():
  code = "class NotAStrategy: pass"
  try:
    RunRequest.load_strategy(code)
    assert False, "Should have raised ValueError"
  except ValueError as e:
    assert str(e) == "No Strategy subclass found in source code"


def test_load_strategy_with_args():
  code = """
from mbt import Strategy
class ArgsStrategy(Strategy):
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def execute(self, ctx):
        pass
"""
  strategy = RunRequest.load_strategy(code, args=[1, 2])
  assert isinstance(strategy, Strategy)
  assert strategy.a == 1
  assert strategy.b == 2

def test_from_json():
    import json
    code = "from mbt import Strategy\nclass MyStrategy(Strategy):\n    def __init__(self, *args): pass\n    def execute(self, ctx): pass"
    config = {
        "strategy": code,
        "symbols": ["AAPL"],
        "start": "2020-01-01",
        "end": "2020-12-31",
        "args": [10]
    }
    json_str = json.dumps(config)
    
    # Test from string
    req = RunRequest.from_json(json_str)
    assert req.symbols == ["AAPL"]
    assert req.start == "2020-01-01"
    
    # Test with overrides
    req2 = RunRequest.from_json(json_str, symbols=["MSFT"], start="2021-01-01")
    assert req2.symbols == ["MSFT"]
    assert req2.start == "2021-01-01"
    assert req2.end == "2020-12-31"

def test_from_json_io():
    import json
    code = "from mbt import Strategy\nclass MyStrategy(Strategy): pass"
    config = {
        "strategy": code,
        "symbols": ["AAPL"],
        "start": "2020-01-01",
        "end": "2020-12-31"
    }
    json_str = json.dumps(config)
    f = io.StringIO(json_str)
    
    req = RunRequest.from_json(f)
    assert req.symbols == ["AAPL"]
