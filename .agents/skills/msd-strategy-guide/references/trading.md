

# Coding Specification

You should create a subclass named `Strategy` of `mbt.Strategy`.

- **Fast-Path (Recommended when signals do NOT depend on holding/account state)**: Implement `signals(self, ctx)` returning `(buy_signals, sell_signals)` over the full series. `Runner` bypasses the Python bar-by-bar loop and executes directly in the multi-threaded Rust `alpha.sim.Simulator` kernel (recommended for full-market runs such as `5,000 stocks × 2,000 bars`). See [double_avg_fast.py](https://github.com/msd-rs/msd-backtest/strategy/double_avg_fast.py) for a complete example.
- **Bar-by-Bar Path (When signals depend on position/account state)**: Implement `prepare(self, ctx)` (optional, recommended for static indicators) and `execute(self, ctx)`.

The skeleton of the code is shown below:

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
    prepare static indicators or signals once before execution starts.
    During prepare(), alpha-lib context is set to the full bar range [0, bars),
    reducing indicator calculation complexity from O(N * T^2) to O(N * T).
    """
    # Example: pre-compute indicators once across the entire series
    # self.ma20 = al.MA(ctx.close, 20)
    pass

  # --- Option A: Fast-Path (when logic does NOT use position / holding_ror / holding_days) ---
  # def signals(self, ctx: mbt.Context) -> tuple[np.ndarray, np.ndarray]:
  #   """Return (buy_signals, sell_signals) of length ctx.n_objs * ctx.bars (bool or float64 in [0, 1])."""
  #   fast_ma = al.MA(ctx.close, 5)
  #   slow_ma = al.MA(ctx.close, 20)
  #   return al.CROSS(fast_ma, slow_ma), al.RCROSS(fast_ma, slow_ma)

  # --- Option B: Bar-by-Bar (when logic depends on dynamic position / account state) ---
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
- `prepare(self, ctx)` lifecycle hook: called once before execution with full-range `alpha-lib` context (`start=0, end=bars`). Always pre-compute static technical indicators (such as `al.MA`) in `prepare()` so they run in `O(N * T)` instead of recomputing them inside `execute()` on every bar (`O(N * T^2)`).
- `signals(self, ctx) -> tuple[np.ndarray, np.ndarray] | None` fast-path hook (optional): called once right after `prepare()` under the full-range `alpha-lib` context (`start=0, end=bars`).
  - **When to use**: Whenever your entry/exit logic depends purely on market/financial features and does **not** query dynamic account or holding state (`ctx.position()`, `ctx.holding_ror()`, `ctx.holding_days()`, `ctx.ror_hold`), and you are running independent per-stock simulation (`Runner`, non-portfolio mode).
  - **Contract**: Return `(buy_signals, sell_signals)` where each element is a 1D array of length `ctx.n_objs * ctx.bars` (`bool` mask treated as `1.0`, or `float64` fraction in `[0.0, 1.0]`). All operators used to build `buy_signals`/`sell_signals` must be strictly causal (no lookahead).
  - **Benefit**: When non-`None` is returned, `Runner` skips the Python bar-by-bar `execute()` loop completely and runs the entire backtest in the multi-threaded Rust `alpha.sim.Simulator` kernel while preserving exact corporate-action (`dividend`, `transfer_shares`, `right_shares`, `right_price`), fee, lot-size, slippage, and limit semantics.
- `execute(self, ctx)` lifecycle hook: use `execute(self, ctx)` when signals depend on dynamic account/position state (such as `ctx.holding_ror`, `ctx.position`, `ctx.holding_days`) or in `PortfolioStrategy` (`-p`) rebalancing mode. Note that `Runner` still accelerates bar-by-bar `execute()` with the Rust `AccountBatch` engine underneath.
- `finalize(self, ctx)` lifecycle hook: called once after the backtest completes with full-range `alpha-lib` context (`start=0, end=bars`) and fully-settled accounts. Use this to count exit/rebalance reasons, log summary diagnostics, or return a structured dict of custom metrics to appear under `## Strategy Diagnostics` in Markdown summaries and JSON exports.
- `mbt.Context` hold context of strategy running, including:
  - data (`ctx.data(name)` or `ctx.<name>`)
  - stock list (`ctx.objs()`)
  - current bar index (`ctx.bar` or `ctx.step`, 0-based), total bars per stock (`ctx.bars`), and stock count (`ctx.n_objs`)
  - 2D `Panel` view (`ctx.panel()`) for multi-period resampling (`panel.resample("1W")`) and 2D array slicing (`panel.compute(...)`, `panel.column(...)`)
- `fin_fields` parameter: MSD `stock_financial` contains hundreds of financial columns. If your strategy accesses financial data (`ctx.f001`, etc.), always define `fin_fields = ["f001", "f003", ...]` on your Strategy class. If not provided, `load_data` queries all fields `*`, which significantly increases memory usage and loading time.
- when you to implement the `execute`, use any loop carefully, because it may too slow, vector computing should use preferentially.

## Position-Aware Strategies (Vectorized `execute()`)

When your strategy **must** inspect current holdings or position returns (e.g., take-profit, stop-loss, trailing stop, or grid add-on buying), you cannot use `signals(ctx)`, so you should implement `prepare()` + `execute()`. Follow the pattern in [holding_ror_grid.py](https://github.com/msd-rs/msd-backtest/strategy/holding_ror_grid.py) to keep `5,000 stocks × 2,000 bars` under ~1 second:

1. **Precompute static market conditions in `prepare(ctx)` as 2D `(n_objs, bars)` masks**, and preallocate per-stock state arrays of length `n_objs`:
   - Use `al.FLAG_STRICTLY_CYCLE` (with `warmup_bars`) so rolling indicators like `MA60` produce `NaN` until a full window is available.
   - Maintain per-stock thresholds/flags as 1D NumPy arrays (e.g., `self.next_add_ror`, `self.pending_exit`).
2. **Avoid `for g in range(ctx.n_objs): ctx.position(g)` in `execute(ctx)`**:
   - Calling scalar `ctx.position(g)` and `ctx.buy(g)` inside a per-stock loop across 5,000 stocks × 2,000 bars causes 10M Python `Position` allocations and FFI calls.
   - Instead, query the full cross-section in one call with **`ctx.holding_shares()`** and **`ctx.holding_rors()`**, compute boolean masks with NumPy, and submit orders once per bar with `ctx.sell(sell_mask, ...)` and `ctx.buy(buy_mask, ...)`.
3. **Never compute holding return manually from `ctx.close` vs `pos.price`**:
   - Always use `ctx.holding_rors()` (or `ctx.holding_ror(obj)`). The engine applies ex-dividend/ex-split adjustments *before* `execute()` runs and accounts for cash dividends and split cost dilution, preventing false `-50%` triggers on 10-for-10 stock splits.
4. **Respect warmup, `pending_exit`, and lot-size cash sizing**:
   - Return early `if ctx.is_warmup:` before mutating state arrays.
   - Track `pending_exit[tp_mask] = True` when submitting a sell and reset it when `shares <= 0.0` so `next_open` mode does not duplicate exits or add-ons before the open fill settles.
   - Remember A-shares trade in lots of `unit = 100`: ensure `initial_cash * buy_percent >= 100 * price` (CLI `-c` defaults to `10,000,000`), otherwise orders are rejected as `ACTION_BUY_NO_CASH`.


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
  - `open`, `high`, `low`, `close`, `volume`, `amount`, `vwap` on bar level (`open`, `high`, `low`, `close`, `vwap` are geometric backward-adjusted via `BW_SPLIT_FACTOR` to share a consistent scale for technical indicators; `bw_factor`/`fw_factor` and `bw_price`/`fw_price` are also available)
  - `price`, `raw_open`, `raw_vwap`: unadjusted raw prices. **Important**: When computing market cap (`ctx.price * ctx.total_shares`) or comparing with per-share financial fields (such as PB = `ctx.price / ctx.f003` or PE), always use unadjusted `ctx.price` rather than adjusted `ctx.close`.
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
  buy/sell a position, `signals` is a boolean array (shape `(n_objs,)` or `(n_objs * bars,)`) or stock index/symbol, `True` means buy/sell, `percent` is the percent of cash/position to buy/sell.
  - Calling both `ctx.sell` and `ctx.buy` on the same bar is supported without overwriting: sells are executed first to release cash before buys.
- `ctx.holding_shares() -> np.ndarray`
  `ctx.holding_rors() -> np.ndarray`
  **Vectorized position accessors** (preferred inside `execute()` over per-stock loops): return 1D `float64` arrays of shape `(n_objs,)` containing currently held shares and corporate-action-aware holding rates of return for all stocks at the current bar.
- `ctx.position(obj: str | int | None = None) -> Position`
  `ctx.holding_ror(obj: str | int | None = None) -> float`
  `ctx.holding_pnl(obj: str | int | None = None) -> float`
  `ctx.cost_price(obj: str | int | None = None, adjusted: bool = False) -> float`
  `ctx.holding_days(obj: str | int | None = None) -> int`
  Single-stock position & corporate-action-aware return/cost/holding-duration accessors (for portfolio strategies or targeted per-stock inspection). Always prefer `holding_ror` / `holding_rors` or `cost_price(..., adjusted=True)` over raw arithmetic with `ctx.close`.
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