import numpy as np
import pytest
from mbt.shared.stocks import get_limit_ratio, ALL_STOCKS

def test_get_limit_ratio_order():
  symbols = ["SH600000", "SZ300750", "SZ300477"]
  ratios = get_limit_ratio(symbols)
  
  assert len(ratios) == 3
  # SH600000 is standard stock -> 0.1
  # SZ300750 is ChiNext stock -> 0.2
  # SZ300477 is ST stock -> 0.05
  np.testing.assert_array_equal(ratios, np.array([0.1, 0.2, 0.05]))

def test_get_limit_ratio_reverse_order():
  symbols = ["SZ300477", "SZ300750", "SH600000"]
  ratios = get_limit_ratio(symbols)
  
  np.testing.assert_array_equal(ratios, np.array([0.05, 0.2, 0.1]))

def test_get_limit_ratio_empty():
  ratios = get_limit_ratio([])
  assert len(ratios) == 0
  assert ratios.dtype == np.float64
