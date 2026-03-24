import os
import sys
import pytest
import io
from mbt.run.backend import RunRequest
from mbt import Strategy

def test_load_strategy_global_vars():
  code = """
from mbt import Strategy
BASE_VAL = 100
class GlobalVarStrategy(Strategy):
  def execute(self, ctx):
    return BASE_VAL
"""
  strategy = RunRequest.load_strategy(code)
  assert strategy.execute(None) == 100

def test_load_strategy_local_import(tmp_path):
  # Create a local library
  lib_dir = tmp_path / "mystrat"
  lib_dir.mkdir()
  
  utils_py = lib_dir / "utils.py"
  utils_py.write_text("VAL = 99")
  
  strat_py = lib_dir / "my_strategy.py"
  strat_py.write_text("""
from mbt import Strategy
import utils
class ImportStrategy(Strategy):
  def execute(self, ctx):
    return utils.VAL
""")
  
  # Load from file
  with open(strat_py, "r") as f:
    # RunRequest.load_strategy uses f.name to find the path
    strategy = RunRequest.load_strategy(f)
  
  assert strategy.execute(None) == 99
