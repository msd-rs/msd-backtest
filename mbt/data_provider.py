# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

from typing import Protocol
import numpy as np
from numpy.typing import DTypeLike

MAX_N = 1_000_000_000


_UNMATERIALIZED = object()


class _LazySlicedDict(dict):
  """Dict view that slices (groups, bars) arrays lazily on first field access."""

  def __init__(
    self,
    src: dict[str, np.ndarray],
    groups: int,
    bars: int,
    start_bar: int,
    skip_keys: set[str],
  ):
    keys = [k for k in src if k not in skip_keys]
    super().__init__({k: _UNMATERIALIZED for k in keys})
    self._src = src
    self._groups = groups
    self._bars = bars
    self._start_bar = start_bar

  def _materialize(self, key):
    val = dict.__getitem__(self, key)
    if val is _UNMATERIALIZED:
      raw = self._src[key]
      if len(raw) == self._groups * self._bars:
        val = np.ascontiguousarray(
          raw.reshape(self._groups, self._bars)[:, self._start_bar :]
        ).reshape(-1)
      else:
        val = raw
      dict.__setitem__(self, key, val)
    return val

  def __getitem__(self, key):
    return self._materialize(key)

  def get(self, key, default=None):
    if key not in self:
      return default
    return self._materialize(key)

  def items(self):
    return [(k, self._materialize(k)) for k in self.keys()]

  def values(self):
    return [self._materialize(k) for k in self.keys()]

  def copy(self):
    return {k: self._materialize(k) for k in self.keys()}


class NextHook(Protocol):
  def __call__(self, i: int, groups: int, bars: int):
    """
    Args:
      i: current index
      groups: number of groups, eg. number of objs
      bars: number of bars per group, eg. number of days
    """
    pass


class DataProvider:
  def __init__(
    self,
    data: dict[str, np.ndarray],
    /,
    objs: list[str] = ["ANY"],
    ts_name: str = "ts",
    next_hook: NextHook | None = None,
  ):
    """
    Args:
      data: data to be used in the strategy, keys are data names, values are numpy arrays, each array must have the same length as dates and can be partitioned into groups of length bars
      objs: objs of the account, each obj has bars bars of data
      ts_name: name of the time series data, default is 'ts', must be in data
      next_hook: hook to be called for each bar, used to do some configuration before the next bar
    """
    self.i = 0
    self.objs = objs
    self.data = data
    self.ts_name = ts_name
    self.dates = data[ts_name]
    self.next_hook = next_hook
    self.groups = len(self.objs)
    self.bars = len(self.dates) // self.groups if self.groups > 0 else 0

  @staticmethod
  def verify_data(objs: list[str], data: dict[str, np.ndarray], ts_name: str):
    """verify the data
    1. ts_name must be in data
    2. all data must have the same length as dates
    3. data length must be integer divisible by len(objs)
    """
    if ts_name not in data:
      raise ValueError(f"data '{ts_name}' not found")
    dates = data[ts_name]
    if len(dates) % len(objs) != 0:
      raise ValueError("data length must be integer divisible by len(objs)")
    for kind in data:
      if len(data[kind]) != len(dates):
        raise ValueError(f"data '{kind}' has different length from dates")

  def all(
    self,
    kind: str,
    /,
    create_if_not_exist: DTypeLike | None = None,
    obj: str | None = None,
  ) -> np.ndarray:
    """
    return the all data of the kind from the first to the last
    Args:
      kind: kind of data
      create_if_not_exist: if not None, create the data if it does not exist with the given dtype
      obj: obj of the data, when provided, return the data of the obj and tile it `groups` times
    """
    if kind not in self.data:
      if create_if_not_exist is not None:
        self.data[kind] = np.zeros(len(self.dates), dtype=create_if_not_exist)
      else:
        raise ValueError(f"data '{kind}' not found")
    if obj is not None and self.groups > 1:
      i = self.objs.index(obj)
      return np.tile(self.data[kind][i * self.bars : (i + 1) * self.bars], self.groups)
    else:
      return self.data[kind]

  def slice(self, kind: str, index: str | int) -> np.ndarray:
    """
    return the slice of the data of the kind
    Args:
      kind: kind of data
      index: index of the slice, can be obj name or index
    Returns:
      np.ndarray: slice of the data
    Raise:
      ValueError: if the kind is not in the data or index is out of range
    """
    if kind not in self.data:
      raise ValueError(f"data '{kind}' not found")
    if isinstance(index, str):
      s, e = self.obj_indices(index)
      return self.data[kind][s : e]
    else:
      return self.data[kind][index: index + self.bars]

  def obj_indices(self, obj: str) -> tuple[int, int]:
    """
    return the indices of the obj
    Args:
      obj: obj of the data
    Returns:
      tuple[int, int]: indices of the obj
    Raise:
      ValueError: if the obj is not in the data
    """
    if obj not in self.objs:
      raise ValueError(f"obj '{obj}' not found")
    i = self.objs.index(obj)
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

  def to_panel(self):
    """Convert DataProvider's block-layout data dictionary into a mbt.select.Panel"""
    from mbt.select.panel import Panel

    if self.bars <= 0 or len(self.objs) == 0:
      raise ValueError("cannot build Panel from empty DataProvider")
    cols = {
      k: v for k, v in self.data.items() if len(v) == self.groups * self.bars
    }
    return Panel(cols, list(self.objs), self.bars)

  def slice_time_window(self, start_bar: int, warmup_bars: int = 0) -> "DataProvider":
    """Return a new DataProvider sliced from `start_bar` to the end without re-querying MSD."""
    if start_bar < 0 or start_bar >= self.bars:
      raise ValueError(f"start_bar {start_bar} out of range [0, {self.bars})")

    transient = {"assets", "ror", "ror_hold", "actions", "action_shares"}
    g = self.groups
    b = self.bars
    if start_bar == 0:
      new_data = {k: v for k, v in self.data.items() if k not in transient}
    else:
      new_data = _LazySlicedDict(self.data, g, b, start_bar, transient)

    dp_sub = DataProvider(
      new_data,
      objs=list(self.objs),
      ts_name=self.ts_name,
      next_hook=self.next_hook,
    )
    dp_sub.warmup_bars = max(0, int(warmup_bars))
    if hasattr(self, "fin_only_year"):
      dp_sub.fin_only_year = self.fin_only_year
    if hasattr(self, "financial_frame"):
      dp_sub.financial_frame = self.financial_frame
    return dp_sub
