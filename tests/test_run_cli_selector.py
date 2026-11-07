import sys
from pathlib import Path
from mbt import StaticSelector
from mbt.select import SelectorDataProvider
from mbt.run.cli import build_selector, parse_args


def test_build_selector_with_stock_codes(tmp_path: Path):
  objs_file = tmp_path / "objs.txt"
  objs_file.write_text("# comment\nSZ000001\nSZ000002\n", encoding="utf-8")

  selector, init_objs = build_selector(["SH600000", str(objs_file)])
  assert isinstance(selector, StaticSelector)
  assert init_objs == ["SH600000", "SZ000001", "SZ000002"]

  selected = selector.execute(SelectorDataProvider(), init_objs)
  assert selected == ["SH600000", "SZ000001", "SZ000002"]


def test_combined_selector_and_strategy_in_single_file(tmp_path: Path, monkeypatch):
  combo_file = tmp_path / "combo_strategy.py"
  combo_file.write_text(
    """
import mbt

class Selector(mbt.Selector):
  def __init__(self, prefix: str = "SH"):
    super().__init__()
    self.prefix = prefix

  def step00(self, stocks: list[str]) -> list[str]:
    \"\"\"Keep stocks matching prefix\"\"\"
    base = stocks if stocks else ["SH600000", "SZ000001", "SH601318"]
    return [s for s in base if s.startswith(self.prefix)]

class Strategy(mbt.Strategy):
  def __init__(self, ma_period: int = 20, threshold: float = 0.05):
    super().__init__()
    self.ma_period = ma_period
    self.threshold = threshold

  def execute(self, ctx: mbt.Context):
    pass
""",
    encoding="utf-8",
  )

  # 1. Run combo_strategy.py WITHOUT -s: automatically uses embedded Selector,
  #    and dispatches -a key=val arguments to both Selector and Strategy!
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "mbt.run.cli",
      str(combo_file),
      "-a",
      "prefix=SZ",
      "ma_period=30",
      "threshold=0.1",
      "-B",
      "SH000300",
      "-b",
      "2024-01-01",
      "-e",
      "2024-12-31",
    ],
  )
  req, _, _, _ = parse_args()
  assert req.objs == ["SH000300", "SZ000001"]
  assert req.strategy.ma_period == 30
  assert req.strategy.threshold == 0.1

  # 2. Override embedded Selector by passing explicit stock codes via -s
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "mbt.run.cli",
      str(combo_file),
      "-s",
      "SH600000",
      "SZ000001",
      "-a",
      "ma_period=15",
      "-B",
      "SH000300",
    ],
  )
  req2, _, _, _ = parse_args()
  assert req2.objs == ["SH000300", "SH600000", "SZ000001"]
  assert req2.strategy.ma_period == 15
