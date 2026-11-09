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

  def data(self, name: str, /, obj: str | None = None) -> np.ndarray:
    """
    return the corresponding data

    Args:
      name: name of the data

    Returns:
      numpy array of the data
    """
    pass

  def __getattr__(self, name: str) -> np.ndarray:
    """
    return the corresponding data

    Args:
      name: name of the data

    Returns:
      numpy array of the data
    """
    return self.data(name)

  def objs(self) -> list[str]:
    """
    return the objs of the account

    Returns:
      list of objs
    """
    pass

  @property
  def bar(self) -> int:
    """current bar index (0-based) during execute()"""
    return 0

  @property
  def step(self) -> int:
    """alias for current bar index (0-based) during execute()"""
    return self.bar

  @property
  def bars(self) -> int:
    """total number of bars per stock"""
    return 0

  @property
  def n_objs(self) -> int:
    """total number of stocks/groups in the context"""
    return 0

  def panel(self, allow_future: bool = False, sliced: bool = False):
    """
    return a Panel view of the underlying data provider
    """
    pass

  def slice_panel(self, lastN: int = 100):
    """
    return a Panel sliced up to the current bar (no lookahead)
    """
    pass

  def buy(self, flags: np.ndarray, percent: float = 1.0):
    """
    buy a position

    Args:
      flags: boolean array of flags, True means buy
      percent: percent of cash to buy
    """
    pass

  def sell(self, flags: np.ndarray, percent: float = 1.0):
    """
    sell a position

    Args:
      flags: boolean array of flags, True means sell
      percent: percent of position to sell
    """
    pass

  def rebalance(self, weights: np.ndarray, tolerance: float | None = None):
    """
    rebalance portfolio holdings to target weights (used in PortfolioStrategy).

    Args:
      weights: 1D array of target weights per stock (shape: (groups,) or (groups * bars,)).
               Non-negative values; if sum > 1.0, weights are automatically normalized to sum to 1.0.
      tolerance: optional tolerance band (e.g. 0.01 for 1%). If position weight deviation
                 from target is within tolerance, no trade is executed.
    """
    pass


class Strategy:
  """Base class of a strategy"""

  portfolio: bool = False
  fin_fields: list[str] | None = None
  fin_only_year: bool = False

  def prepare(self, ctx: Context):
    """
    prepare static indicators or signals once before the bar-by-bar loop starts.
    During prepare(), alpha-lib context is set to the full bar range [0, bars),
    reducing indicator calculation complexity from O(N * T^2) to O(N * T).

    Args:
      ctx: context of the strategy
    """
    pass

  def execute(self, ctx: Context):
    """
    execute the strategy, this method will be called for each bar.
    use ctx.order() to order, ctx.data() to get data

    Args:
      ctx: context of the strategy
    """
    pass


class PortfolioStrategy(Strategy):
  """Base class for multi-stock portfolio rebalancing strategies sharing a single cash pool."""

  portfolio: bool = True
