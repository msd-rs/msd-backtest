

# Coding Specification

You should create a subclass named `Strategy` of `mbt.Strategy` and implement `prepare` (optional, recommended for static indicators) and `execute`. The skeleton of the code is shown below:

```python

import mbt
import numpy as np
import alpha as al

# Other libraries can be imported as needed

class Strategy(mbt.Strategy):

  # Optional: declare needed financial fields to avoid loading all columns (*)
  # fin_fields = ["f001", "f003"]

  def __init__(self):
    super().__init__()
    # TODO: add your custom parameters to __init__ method if needed.

  def prepare(self, ctx: mbt.Context):
    """
    prepare static indicators or signals once before the bar-by-bar loop starts.
    During prepare(), alpha-lib context is set to the full bar range [0, bars),
    reducing indicator calculation complexity from O(N * T^2) to O(N * T).
    """
    # Example: pre-compute indicators once across the entire series
    # self.ma20 = al.MA(ctx.close, 20)
    pass

  def execute(self, ctx: mbt.Context):
    """
    execute the strategy, this method will be called for each bar.

    Args:
      ctx: context of the strategy 
    """
    # TODO: implement strategy, building buy/sell signals then call ctx.buy/ctx.sell
    pass

  def finalize(self, ctx: mbt.Context) -> dict | None:
    """
    optional: called once after the backtest finishes.
    Use for exit reason counting, custom diagnostics, or performance summaries.
    Returning a dict merges it into report.diagnostics, Markdown summary, and JSON export.
    """
    # return {"exit_reasons": self.exit_reasons}
    pass
```


## Basic logic

`mbt` is optimized to run a strategy for multiple stocks's parallel, this will huge improve the backtest performance, it use
- `alpha-lib` as factor compute library, a parallel factor building algorithms
- `prepare(self, ctx)` lifecycle hook: called once before the bar-by-bar loop with full-range `alpha-lib` context (`start=0, end=bars`). Always pre-compute static technical indicators (such as `al.MA`) in `prepare()` so they run in `O(N * T)` instead of recomputing them inside `execute()` on every bar (`O(N * T^2)`). Use `execute(self, ctx)` for dynamic signals that depend on account state (such as `ctx.ror_hold`) or to submit precomputed signals.
- `finalize(self, ctx)` lifecycle hook: called once after the backtest completes with full-range `alpha-lib` context (`start=0, end=bars`) and fully-settled accounts. Use this to count exit/rebalance reasons, log summary diagnostics, or return a structured dict of custom metrics to appear under `## Strategy Diagnostics` in Markdown summaries and JSON exports.
- `mbt.Context` hold context of strategy running, including:
  - data (`ctx.data(name)` or `ctx.<name>`)
  - stock list (`ctx.objs()`)
  - current bar index (`ctx.bar` or `ctx.step`, 0-based), total bars per stock (`ctx.bars`), and stock count (`ctx.n_objs`)
  - 2D `Panel` view (`ctx.panel()`) for multi-period resampling (`panel.resample("1W")`) and 2D array slicing (`panel.compute(...)`, `panel.column(...)`)
- `fin_fields` parameter: MSD `stock_financial` contains hundreds of financial columns. If your strategy accesses financial data (`ctx.f001`, etc.), always define `fin_fields = ["f001", "f003", ...]` on your Strategy class. If not provided, `load_data` queries all fields `*`, which significantly increases memory usage and loading time.
- when you to implement the `execute`, use any loop carefully, because it may too slow, vector computing should use preferentially.

# Execution

After writing a strategy, you can run it to verify it works as expected.

## Test

```shell
# Run strategy with explicit stock codes (use -S to output a compact Markdown summary)
[uv run] python -m mbt.run.cli <STRATEGY_PATH> -s SH600000 -b 2025-01-01 -e 2025-12-31 [-S]

# Run strategy with a Selector .py file (or omit -s if <STRATEGY_PATH> defines both Selector and Strategy)
[uv run] python -m mbt.run.cli <STRATEGY_PATH> [-s <SELECTOR_PATH.py>] [-a key=value ...] -b 2025-01-01 -e 2025-12-31 -S

# Run in portfolio rebalancing mode (auto-enabled for mbt.PortfolioStrategy, or force via -p, custom cash via -c)
[uv run] python -m mbt.run.cli <STRATEGY_PATH> -s SH600000 SZ000001 [-p] [-c 1000000] -b 2025-01-01 -e 2025-12-31 -S
```

`uv run` can be omitted current workspace not using `uv` as package manager.


# API reference

## mbt.Context

- `ctx.data(name: str, /, obj: str | None = None) -> np.ndarray`
  the primary method to get data, `name` is the data name, `obj` is used only when comparison, keep `None` for no explicit requirements. `name` can be:
  - `open`, `high`, `low`, `close`, `volume`, `amount`, `vwap` on bar level (`open`, `high`, `low`, `close`, `vwap` are forward-adjusted to share a consistent scale for technical indicators)
  - `price`, `raw_open`, `raw_vwap`: unadjusted raw prices. **Important**: When computing market cap (`ctx.price * ctx.total_shares`) or comparing with per-share financial fields (such as PB = `ctx.price / ctx.f003` or PE), always use unadjusted `ctx.price` rather than forward-adjusted `ctx.close`.
  - `ror` for rate of return since first bar
  - `ror_hold` for rate of return of holding since last buy after clearance.
  - `limited` used to check whether the bar had reached the daily limit. `0` for no limit, `1` for limit up, `-1` for limit down
  - `total_shares`: total shares, total stock shares, use to calculate total market capitalization etc.
  - `tradable_shares`: tradable shares, tradable stock shares, use to calculate tradable market capitalization, turnover etc.
  - Financial fields are named like `f001` can be referred to [financial](./financial.md) (Note: financial records in MSD are indexed by fiscal period end date such as `03-31`, `06-30`, `09-30`, `12-31` and aligned backward to daily bars).
    > **Memory Optimization**: By default, `load_data` queries all financial columns (`*`) when `fin_fields` is not provided. To minimize memory consumption and boost load speed, always declare `fin_fields = ["f001", ...]` in your `Strategy` class.
- `ctx.bar -> int` (alias `ctx.step -> int`)
  current 0-based bar index during `execute(ctx)`. Also `ctx.bars -> int` (total bars per stock) and `ctx.n_objs -> int` (number of stocks in context).
- `ctx.date -> str` (alias `ctx.current_date -> str`)
  current bar trading date as `YYYY-MM-DD` string (e.g. `'2024-05-06'`), avoiding manual `ctx.data("ts")[ctx.bar]` indexing.
- `ctx.index_of(obj: str) -> int`
  return the index of stock `obj` in `ctx.objs()`. Raises `ValueError` if not found.
- `ctx.objs() -> list[str]`
  return the list of stock objs in the current backtest.
- `ctx.panel() -> mbt.select.Panel`
  return a 2D `Panel` view (`(n_objs, bars)`) of the strategy data. Useful in `prepare(ctx)` for multi-period resampling (e.g. `ctx.panel().resample("1W")`), `panel.compute(...)`, or 2D views (`panel.column("close")`).
- `ctx.buy(signals: np.ndarray, percent: float = 1.0)`
  `ctx.sell(signals: np.ndarray, percent: float = 1.0)`
  buy/sell a position, `signals` is a boolean array of signals, `True` means buy/sell, `percent` is the percent of cash/position to buy/sell.
  - Calling both `ctx.sell` and `ctx.buy` on the same bar is supported without overwriting: sells are executed first to release cash before buys.
- `ctx.rebalance(weights: np.ndarray, tolerance: float | None = None)`
  rebalance portfolio holdings to target weights (for `mbt.PortfolioStrategy` or `-p` mode). `weights` is a non-negative 1D array of target portfolio weights per stock (either shape `(n_objs,)` or full block length `(n_objs * bars,)`; if the sum exceeds `1.0`, it is normalized automatically). Sells are executed first to release cash before buys.
- `ctx.rebalance_objs(target_weights: dict[str, float], tolerance: float | None = None)`
  convenient mapping-based rebalance for portfolio mode (e.g. `ctx.rebalance_objs({"SH600000": 0.05, "SZ000001": 0.05})`). Stocks not in `target_weights` receive `0.0` target weight (liquidated).

## Financial Fields Workflow & Schema Introspection

MSD `stock_financial` contains over 160+ quarterly financial fields (`f000` to `f242`).

> [!IMPORTANT]
> **Single Source of Truth Rule**:
> Financial field code names (`f***`) must **never be guessed or hallucinated**. Always query the running MSD instance using `msd shell` before writing strategy factors.

Run the following commands to inspect fields:

```bash
# Query the full schema with column descriptions:
msd shell -s "${MSD_HOST:-http://127.0.0.1:50510}" ".schema stock_financial"

# Grep specific fundamental metrics:
msd shell -s "${MSD_HOST:-http://127.0.0.1:50510}" ".schema stock_financial" | grep "每股净资产"   # -> f003 (BVPS)
msd shell -s "${MSD_HOST:-http://127.0.0.1:50510}" ".schema stock_financial" | grep "净资产收益率" # -> f205 (ROE)
msd shell -s "${MSD_HOST:-http://127.0.0.1:50510}" ".schema stock_financial" | grep "增长率"       # -> revenue / profit growth
```

For full inspection guidance and common indicator mapping, see [financial.md](./financial.md).

## alpha-lib

When you need calculate any factor, you may to use alpha-lib first

refer to [algo.md](./algo.md) to see what algorithm function supported.