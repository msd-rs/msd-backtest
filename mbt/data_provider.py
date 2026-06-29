from typing import Protocol
import numpy as np
from numpy.typing import DTypeLike

MAX_N = 1_000_000_000


class NextHook(Protocol):
  def __call__(self, i: int, groups: int, bars: int):
    """
    Args:
      i: current index
      groups: number of groups, eg. number of symbols
      bars: number of bars per group, eg. number of days
    """
    pass


class DataProvider:
  def __init__(
    self,
    data: dict[str, np.ndarray],
    /,
    symbols: list[str] = ["ANY"],
    ts_name: str = "ts",
    next_hook: NextHook | None = None,
  ):
    """
    Args:
      data: data to be used in the strategy, keys are data names, values are numpy arrays, each array must have the same length as dates and can be partitioned into groups of length bars
      symbols: symbols of the account, each symbol has bars bars of data
      ts_name: name of the time series data, default is 'ts', must be in data
      next_hook: hook to be called for each bar, used to do some configuration before the next bar
    """
    self.i = 0
    self.symbols = symbols
    self.data = data
    self.ts_name = ts_name
    self.dates = data[ts_name]
    self.next_hook = next_hook
    self.groups = len(self.symbols)
    self.bars = len(self.dates) // self.groups

  @staticmethod
  def verify_data(symbols: list[str], data: dict[str, np.ndarray], ts_name: str):
    """verify the data
    1. ts_name must be in data
    2. all data must have the same length as dates
    3. data length must be integer divisible by len(symbols)
    """
    if ts_name not in data:
      raise ValueError(f"data '{ts_name}' not found")
    dates = data[ts_name]
    if len(dates) % len(symbols) != 0:
      raise ValueError("data length must be integer divisible by len(symbols)")
    for kind in data:
      if len(data[kind]) != len(dates):
        raise ValueError(f"data '{kind}' has different length from dates")

  def all(
    self,
    kind: str,
    /,
    create_if_not_exist: DTypeLike | None = None,
    symbol: str | None = None,
  ) -> np.ndarray:
    """
    return the all data of the kind from the first to the last
    Args:
      kind: kind of data
      create_if_not_exist: if not None, create the data if it does not exist with the given dtype
      symbol: symbol of the data, when provided, return the data of the symbol and tile it `groups` times
    """
    if kind not in self.data:
      if create_if_not_exist is not None:
        self.data[kind] = np.zeros(len(self.dates), dtype=create_if_not_exist)
      else:
        raise ValueError(f"data '{kind}' not found")
    if symbol is not None and self.groups > 1:
      i = self.symbols.index(symbol)
      return np.tile(self.data[kind][i * self.bars : (i + 1) * self.bars], self.groups)
    else:
      return self.data[kind]

  def slice(self, kind: str, index: str | int) -> np.ndarray:
    """
    return the slice of the data of the kind
    Args:
      kind: kind of data
      index: index of the slice, can be symbol name or index
    Returns:
      np.ndarray: slice of the data
    Raise:
      ValueError: if the kind is not in the data or index is out of range
    """
    if kind not in self.data:
      raise ValueError(f"data '{kind}' not found")
    if isinstance(index, str):
      s, e = self.symbol_indices(index)
      return self.data[kind][s : e]
    else:
      return self.data[kind][index * self.bars : (index + 1) * self.bars]

  def symbol_indices(self, symbol: str) -> tuple[int, int]:
    """
    return the indices of the symbol
    Args:
      symbol: symbol of the data
    Returns:
      tuple[int, int]: indices of the symbol
    Raise:
      ValueError: if the symbol is not in the data
    """
    if symbol not in self.symbols:
      raise ValueError(f"symbol '{symbol}' not found")
    i = self.symbols.index(symbol)
    return i * self.bars, (i + 1) * self.bars

  def __len__(self):
    return self.bars

  def __iter__(self):
    return self

  def __next__(self):
    self.i += 1
    if self.i >= self.bars:
      raise StopIteration
    if self.next_hook is not None:
      self.next_hook(self.i, self.groups, self.bars)
    return self.i

  def reset(self):
    self.i = 0
