from mbt.account import Operation, Position
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
  def date(self) -> str:
    """current bar date as YYYY-MM-DD string during execute()"""
    return ""

  @property
  def current_date(self) -> str:
    """alias for current bar date as YYYY-MM-DD string during execute()"""
    return self.date

  def index_of(self, obj: str) -> int:
    """return the index of an obj in ctx.objs(). Raises ValueError if not found."""
    return self.objs().index(obj)

  @property
  def step(self) -> int:
    """alias for current bar index (0-based) during execute()"""
    return self.bar

  @property
  def warmup_bars(self) -> int:
    """number of warmup bars prior to the backtest start date"""
    return 0

  @property
  def is_warmup(self) -> bool:
    """True if current bar is in the warmup period before the formal start date"""
    return self.bar < self.warmup_bars

  @property
  def bars(self) -> int:
    """total number of bars per stock"""
    return 0

  @property
  def n_objs(self) -> int:
    """total number of stocks/groups in the context"""
    return 0

  def panel(self, allow_future: bool = False, sliced: bool | None = None):
    """Return a Panel view of the underlying data provider.

    During strategy execute(), defaults to returning a strictly sliced Panel
    (up to the current bar) to prevent lookahead bias.
    To explicitly obtain the full-range panel with future bars, pass allow_future=True.
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

  def rebalance_objs(self, target_weights: dict[str, float], tolerance: float | None = None):
    """
    rebalance portfolio holdings using a mapping of {symbol: weight} (used in PortfolioStrategy).

    Args:
      target_weights: dict mapping stock symbol to target weight (e.g. {"SH600000": 0.05}).
                      Missing symbols receive a weight of 0.0.
      tolerance: optional tolerance band (e.g. 0.01 for 1%).
    """
    pass

  def position(self, obj: str | int | None = None) -> Position:
    """Return the Position object for obj or the single stock account.

    The returned Position instance exposes built-in computed properties:
      - `pos.ror`: rate of return as decimal (e.g. -0.05 for -5% loss).
      - `pos.unrealized_pnl`: net unrealized profit/loss amount.
      - `pos.holding_days`: number of bars/trading days this position has been held.
      - `pos.adj_cost_price`: equivalent forward-adjusted cost basis, safe to compare with `ctx.close`.
      - `pos.is_holding`: True if currently holding shares (shares > 0).
    """
    pass

  def holding_ror(self, obj: str | int | None = None) -> float:
    """Return holding rate of return as a decimal (e.g. -0.05 for -5%).

    Safe, cross-dimensional and corporate-action-aware holding return calculator.
    Eliminates forward-adjusted vs raw price mismatch and dividend/split distortions.
    """
    pass

  def holding_pnl(self, obj: str | int | None = None) -> float:
    """Return holding profit/loss amount for obj."""
    pass

  def cost_price(self, obj: str | int | None = None, adjusted: bool = False) -> float:
    """Return holding acquisition cost price.

    Args:
      obj: Symbol or group index.
      adjusted: If True, returns forward-adjusted equivalent cost (adj_cost_price),
        safe to compare with `ctx.close`. If False, returns raw execution cost price.
    """
    pass

  def holding_days(self, obj: str | int | None = None) -> int:
    """Return number of bars since position was first acquired."""
    pass


class Strategy:
  """Base class of a trading strategy.

  Attributes:
    portfolio: If True, indicates this strategy operates in shared-cash portfolio mode
      using `PortfolioRunner` and `PortfolioAccount`. Default False.
    exec_mode: Execution mode for order matching:
      - "close" (default): orders are matched against the same bar's close price.
      - "next_open": signals generated at bar T's close are matched against bar T+1's open price,
        faithfully simulating real-world next-day market opening execution.
    fin_fields: List of financial field names (e.g. ['f001', 'f205']) to query from
      `stock_financial`. If None or empty (default), financial queries are skipped
      to save memory and prevent accidental memory exhaustion.
    fin_only_year: If True, only annual financial reports (ts.month=12) are queried.
    fin_lookback_days: Explicit lookback window in calendar days for financial statement history.
      Defaults to 1200 days (~3.3 years) for annual reports and 500 days for quarterly reports.
    warmup_bars: Number of warmup bars loaded before the start date for indicator initialization.
      
      Warmup Bars Lifecycle:
      1. Declaration: Set `warmup_bars = N` on the Strategy class or via CLI `--warmup-bars N`.
      2. Data Extension: `load_data` queries historical data before the target `start` date,
         locating the start date bar index as `dp.warmup_bars`.
      3. Execution: During bars `0 <= bar < warmup_bars`, `execute(ctx)` runs normally to allow
         rolling indicators (EMA, MACD, etc.) to stabilize, but trading execution is suppressed
         (orders are not matched, holdings and cash remain unchanged).
         IMPORTANT: If your strategy maintains stateful counters, signal caches, or pending flags,
         check `if not ctx.is_warmup:` before mutating state, or precompute indicators in `prepare(ctx)`.
      4. Evaluation: `Report` slices out warmup bars, reporting metrics strictly starting
         from the user's target start date.
  """

  portfolio: bool = False
  exec_mode: str = "close"
  fin_fields: list[str] | None = None
  fin_only_year: bool = False
  fin_lookback_days: int | None = None
  warmup_bars: int = 0

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

  def finalize(self, ctx: Context) -> dict | None:
    """
    finalize and summarize strategy execution after the bar-by-bar loop completes.
    During finalize(), alpha-lib context is set to the full bar range [0, bars),
    and the account history (records, assets, ror) is fully populated.

    Args:
      ctx: context of the strategy

    Returns:
      Optional dict of custom metrics/diagnostics (e.g. exit reason counts)
      to be recorded in Report.diagnostics and included in summaries.
    """
    pass


class PortfolioStrategy(Strategy):
  """Base class for multi-stock portfolio rebalancing strategies sharing a single cash pool."""

  portfolio: bool = True
