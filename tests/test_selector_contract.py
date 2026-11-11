import pytest
import mbt
from mbt.select import SelectorDataProvider, MsdSelectorDataProvider


def test_top_level_export():
  assert hasattr(mbt, "MsdSelectorDataProvider")
  assert mbt.MsdSelectorDataProvider is MsdSelectorDataProvider
  assert "MsdSelectorDataProvider" in mbt.__all__


def test_selector_data_provider_abstract_not_implemented():
  dp = SelectorDataProvider()

  with pytest.raises(NotImplementedError) as exc_kline:
    dp.load_kline(["SH600000"])
  assert "is an abstract interface method" in str(exc_kline.value)
  assert "MsdSelectorDataProvider" in str(exc_kline.value)

  with pytest.raises(NotImplementedError) as exc_fin:
    dp.load_financial(["SH600000"], ["f001"])
  assert "is an abstract interface method" in str(exc_fin.value)
  assert "MsdSelectorDataProvider" in str(exc_fin.value)

  with pytest.raises(NotImplementedError) as exc_snap:
    dp.load_snapshot(["SH600000"])
  assert "is an abstract interface method" in str(exc_snap.value)
  assert "MsdSelectorDataProvider" in str(exc_snap.value)
