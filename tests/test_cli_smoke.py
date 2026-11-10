import typing
import pytest
from mbt.run.cli import parse_args, run_single_period
from mbt.data_provider import DataProvider


def test_cli_type_hints():
  hints = typing.get_type_hints(run_single_period)
  assert hints["return"] == tuple[list[typing.Any], DataProvider, typing.Any] or "DataProvider" in str(hints["return"])


def test_cli_help(capsys):
  import sys
  orig_argv = sys.argv
  sys.argv = ["cli.py", "-h"]
  with pytest.raises(SystemExit) as exc_info:
    parse_args()
  assert exc_info.value.code == 0
  sys.argv = orig_argv
