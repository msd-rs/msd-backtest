from pathlib import Path
import json
import pytest
from mbt import Fee
from mbt.account import FeeItem, DEFAULT_FEE_CONFIG
from mbt.run.cli import parse_args


def test_fee_default():
  fee = Fee.default()
  assert len(fee.config) >= 1
  item = fee.get(0)
  assert item.commission_buy == 0.0015
  assert item.min_commission_buy == 5.0
  assert item.tax_buy == 0.001
  assert item.transaction_fee_buy == 0.0001
  assert item.commission_sell == 0.0015
  assert item.min_commission_sell == 5.0
  assert item.tax_sell == 0.0
  assert item.transaction_fee_sell == 0.0


def test_fee_load_empty_or_none():
  fee_none = Fee.load(None)
  fee_empty = Fee.load("")
  assert fee_none.get(0).commission_buy == 0.0015
  assert fee_empty.get(0).commission_buy == 0.0015


def test_fee_load_non_existent_file():
  # When file does not exist, should gracefully fall back to default fee instead of raising error
  fee = Fee.load("path/that/does/not/exist/fee.json")
  assert fee.get(0).commission_buy == 0.0015
  assert fee.get(0).min_commission_buy == 5.0


def test_fee_load_json_string():
  custom_json = json.dumps([
    {
      "ts": 0,
      "commission_buy": 0.002,
      "min_commission_buy": 10.0,
      "tax_buy": 0.0,
      "transaction_fee_buy": 0.0,
      "commission_sell": 0.002,
      "min_commission_sell": 10.0,
      "tax_sell": 0.001,
      "transaction_fee_sell": 0.0,
    }
  ])
  fee = Fee.load(custom_json)
  assert fee.get(0).commission_buy == 0.002
  assert fee.get(0).min_commission_buy == 10.0
  assert fee.get(0).tax_sell == 0.001


def test_fee_load_custom_file(tmp_path):
  custom_fee_file = tmp_path / "custom_fee.json"
  custom_fee_file.write_text(
    json.dumps([
      {
        "ts": 0,
        "commission_buy": 0.003,
        "min_commission_buy": 1.0,
        "tax_buy": 0.0,
        "transaction_fee_buy": 0.0,
        "commission_sell": 0.003,
        "min_commission_sell": 1.0,
        "tax_sell": 0.0,
        "transaction_fee_sell": 0.0,
      }
    ]),
    encoding="utf-8",
  )
  fee = Fee.load(str(custom_fee_file))
  assert fee.get(0).commission_buy == 0.003
  assert fee.get(0).min_commission_buy == 1.0


def test_fee_default_fallback_when_no_files_exist(monkeypatch):
  # Simulate candidate.is_file() returning False for all candidate files
  monkeypatch.setattr(Path, "is_file", lambda self: False)
  fee = Fee.default()
  assert fee.get(0).commission_buy == 0.0015
  assert fee.get(0).min_commission_buy == 5.0


def test_cli_parse_args_without_fee_arg(tmp_path, monkeypatch):
  import sys

  strat_file = tmp_path / "dummy_strat.py"
  strat_file.write_text(
    """
import mbt

class DummyStrategy(mbt.Strategy):
  def execute(self, ctx):
    pass
""",
    encoding="utf-8",
  )

  # Clear MSD_BT_FEE env var
  monkeypatch.delenv("MSD_BT_FEE", raising=False)
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "mbt.run.cli",
      str(strat_file),
      "-s",
      "SH600000",
      "-B",
      "SH000300",
    ],
  )
  _, _, _, fee = parse_args()
  assert fee is not None
  assert fee.get(0).commission_buy == 0.0015
  assert fee.get(0).min_commission_buy == 5.0


def test_cli_parse_args_with_missing_fee_file(tmp_path, monkeypatch):
  import sys

  strat_file = tmp_path / "dummy_strat.py"
  strat_file.write_text(
    """
import mbt

class DummyStrategy(mbt.Strategy):
  def execute(self, ctx):
    pass
""",
    encoding="utf-8",
  )

  monkeypatch.setenv("MSD_BT_FEE", "/non/existent/fee.json")
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "mbt.run.cli",
      str(strat_file),
      "-s",
      "SH600000",
      "-B",
      "SH000300",
    ],
  )
  _, _, _, fee = parse_args()
  assert fee is not None
  # Gracefully fell back to default fee
  assert fee.get(0).commission_buy == 0.0015
  assert fee.get(0).min_commission_buy == 5.0


def test_fee_apply_overrides():
  fee = Fee.default()
  fee.apply_overrides({
    "commission": 0.0002,
    "min_commission": 0.0,
    "tax": 0.0005,
    "transaction_fee": 0.00002,
  })
  item = fee.get(0)
  assert item.commission_buy == 0.0002
  assert item.commission_sell == 0.0002
  assert item.min_commission_buy == 0.0
  assert item.min_commission_sell == 0.0
  assert item.transaction_fee_buy == 0.00002
  assert item.transaction_fee_sell == 0.00002
  assert (item.tax_buy == 0.0005 or item.tax_sell == 0.0005)


def test_cli_parse_args_with_fee_overrides(tmp_path, monkeypatch):
  import sys

  strat_file = tmp_path / "dummy_strat.py"
  strat_file.write_text("import mbt\nclass S(mbt.Strategy):\n  pass\n", encoding="utf-8")
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "mbt.run.cli",
      str(strat_file),
      "-s",
      "SH600000",
      "-a",
      "commission=0.0003",
      "min_commission=1.0",
    ],
  )
  _, _, _, fee = parse_args()
  assert fee.get(0).commission_buy == 0.0003
  assert fee.get(0).commission_sell == 0.0003
  assert fee.get(0).min_commission_buy == 1.0
  assert fee.get(0).min_commission_sell == 1.0
