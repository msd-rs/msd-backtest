from mbt.account import Operation
import numpy as np


class Context:
  """Context of a strategy"""

  def __init__(self):
    pass

  def order(self, op: Operation) -> Operation:
    """
    execute an operation and return the result

    Args:
      op: operation to be executed
    """
    pass

  def data(self, name: str) -> np.ndarray:
    """
    return the corresponding data

    Args:
      name: name of the data

    Returns:
      numpy array of the data
    """
    pass

  def symbols(self) -> list[str]:
    """
    return the symbols of the account

    Returns:
      list of symbols
    """
    pass


class Strategy:
  """Base class of a strategy"""

  def execute(self, ctx: Context):
    """
    execute the strategy, this method will be called for each bar.
    use ctx.order() to order, ctx.data() to get data

    Args:
      ctx: context of the strategy
    """
    pass
