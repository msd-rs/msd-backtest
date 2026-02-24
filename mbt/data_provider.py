import numpy as np

MAX_N = 1_000_000_000


class DataProvider:
  def __init__(self):
    self.i = 0

  def last(self, kind: str, default: float = 0.0) -> float:
    """
    return the last data of the kind
    Args:
      kind: kind of data
      default: default value if the kind is not found
    """
    return default

  def all(self, kind: str) -> np.ndarray:
    """
    return the all data of the kind from the first to the last
    Args:
      kind: kind of data
    """
    pass

  def __next__(self):
    self.i += 1
