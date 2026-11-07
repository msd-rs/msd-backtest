---
name: write-strategy
description: guide user to write a strategy for msd backtest
tags: [msd, strategy, backtest]
version: 1.0
author: elsejj
---


This guide will help you to write a strategy for msd backtest.

# Technical stack

Programming language: Python 3.12+


# Preparation

- Ensure `msd-backtest[cli]` be installed use project package manager.
- When your need to run in server mode, ensure `msd-backtest[server]` is installed. Normally, server mode is not required.
- Ensure `MSD_HOST` environment variable be set, can be find in `.env` file or test by shell command `env | grep MSD_HOST`. If not set, you can ask the user to set it first. When user can't provide it, you can ask and install it by [the guide](./references/msd.md).
- `MSD_BT_FEE` is optional. `./etc/fee.json` can be used to customize trade fees, defaulting to standard rates if omitted:

```json
[
  {
    "ts": 4102416000,
    "commission_buy": 0.0015,
    "min_commission_buy": 5.0,
    "tax_buy": 0.001,
    "transaction_fee_buy": 0.0001,
    "commission_sell": 0.0015,
    "min_commission_sell": 5.0,
    "tax_sell": 0.0,
    "transaction_fee_sell": 0.0
  }
]
```
more less ts item, can be more reality.


# Coding Specification

You should create a subclass named `Strategy` of `mbt.Strategy` and implement `prepare` (optional, recommended for static indicators) and `execute`. The skeleton of the code is shown below:

```python

import mbt
import numpy as np
import alpha as al

# Other libraries can be imported as needed

class Strategy(mbt.Strategy):

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
```

## Portfolio Rebalancing Strategy (`mbt.PortfolioStrategy`)

When the strategy needs to manage a **multi-stock portfolio sharing a single cash pool** and rebalance positions across stocks (rather than testing each stock independently), subclass `mbt.PortfolioStrategy` and use `ctx.rebalance(weights)` (or `ctx.buy` / `ctx.sell`):

```python
class Strategy(mbt.PortfolioStrategy):

  def __init__(self, period: int = 20, top_k: int = 5):
    super().__init__()
    self.period = period
    self.top_k = top_k

  def prepare(self, ctx: mbt.Context):
    # Pre-compute momentum factor across all stocks
    self.roc = al.ROC(ctx.close, self.period)

  def execute(self, ctx: mbt.Context):
    # Call ctx.rebalance(weights) on rebalance bars; sells are executed first to free cash for buys
    ...
```

## Combining Selector and Strategy in a Single File

A single `.py` file can define both a `Selector` (subclass of `mbt.Selector`, see `write-selector` skill) and a `Strategy` (subclass of `mbt.Strategy` or `mbt.PortfolioStrategy`):

```python
class Selector(mbt.Selector):
  def __init__(self, min_roe: float = 15.0):
    super().__init__()
    self.min_roe = min_roe

  def step00(self, stocks: list[str]) -> list[str]:
    """Filter candidate stocks"""
    ...

class Strategy(mbt.Strategy):
  def __init__(self, ma_period: int = 20):
    super().__init__()
    self.ma_period = ma_period

  def prepare(self, ctx: mbt.Context):
    self.ma = al.MA(ctx.close, self.ma_period)

  def execute(self, ctx: mbt.Context):
    ctx.buy(ctx.close > self.ma)
    ctx.sell(ctx.close < self.ma)
```
- When running `mbt.run.cli <STRATEGY_PATH>` without `-s`, it automatically runs the `Selector` in the file first to select the stock pool, then runs `Strategy` on the selected stocks.
- Arguments passed via `-a key=value` are automatically matched and dispatched to `Selector.__init__` and `Strategy.__init__` by parameter name.

## basic logic

`mbt` is optimized to run a strategy for multiple stocks's parallel, this will huge improve the backtest performance, it use
- `alpha-lib` as factor compute library, a parallel factor building algorithms
- `prepare(self, ctx)` lifecycle hook: called once before the bar-by-bar loop with full-range `alpha-lib` context (`start=0, end=bars`). Always pre-compute static technical indicators (such as `al.MA`, `al.MACD`, `al.RSI`) in `prepare()` so they run in `O(N * T)` instead of recomputing them inside `execute()` on every bar (`O(N * T^2)`). Use `execute(self, ctx)` for dynamic signals that depend on account state (such as `ctx.ror_hold`) or to submit precomputed signals.
- `mbt.Context` hold context of strategy running, including:
  - data (`ctx.data(name)` or `ctx.<name>`)
  - stock list (`ctx.objs()`)
  - current bar index (`ctx.bar` or `ctx.step`, 0-based), total bars per stock (`ctx.bars`), and stock count (`ctx.n_objs`)
  - 2D `Panel` view (`ctx.panel()`) for multi-period resampling (`panel.resample("1W")`) and 2D array slicing (`panel.compute(...)`, `panel.column(...)`)
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
  - `price`, `raw_open`, `raw_vwap`: unadjusted raw prices. **Important**: When computing market cap (`ctx.price * ctx.total_shares`) or comparing with per-share financial fields (such as PB = `ctx.price / ctx.f004` or PE), always use unadjusted `ctx.price` rather than forward-adjusted `ctx.close`.
  - `ror` for rate of return since first bar
  - `ror_hold` for rate of return of holding since last buy after clearance.
  - `limited` used to check whether the bar had reached the daily limit. `0` for no limit, `1` for limit up, `-1` for limit down
  - `total_shares`: total shares, total stock shares, use to calculate total market capitalization etc.
  - `tradable_shares`: tradable shares, tradable stock shares, use to calculate tradable market capitalization, turnover etc.
  - Financial fields are named like `f001` can be referred to [financial](./references/financial.md) (Note: financial records in MSD are indexed by fiscal period end date such as `03-31`, `06-30`, `09-30`, `12-31` and aligned backward to daily bars).
- `ctx.bar -> int` (alias `ctx.step -> int`)
  current 0-based bar index during `execute(ctx)`. Also `ctx.bars -> int` (total bars per stock) and `ctx.n_objs -> int` (number of stocks in context).
- `ctx.objs() -> list[str]`
  return the list of stock objs in the current backtest.
- `ctx.panel() -> mbt.select.Panel`
  return a 2D `Panel` view (`(n_objs, bars)`) of the strategy data. Useful in `prepare(ctx)` for multi-period resampling (e.g. `ctx.panel().resample("1W")`), `panel.compute(...)`, or 2D views (`panel.column("close")`).
- `ctx.buy(signals: np.ndarray, percent: float = 1.0)`
  `ctx.sell(signals: np.ndarray, percent: float = 1.0)`
  buy/sell a position, `signals` is a boolean array of signals, `True` means buy/sell, `percent` is the percent of cash/position to buy/sell.
  - Calling both `ctx.sell` and `ctx.buy` on the same bar is supported without overwriting: sells are executed first to release cash before buys.
- `ctx.rebalance(weights: np.ndarray)`
  rebalance portfolio holdings to target weights (for `mbt.PortfolioStrategy` or `-p` mode). `weights` is a non-negative 1D array of target portfolio weights per stock (either shape `(n_objs,)` or full block length `(n_objs * bars,)`; if the sum exceeds `1.0`, it is normalized automatically). Sells are executed first to release cash before buys.

## alpha-lib

When you need calculate any factor, you may to use alpha-lib first

refer to [algo.md](./references/algo.md) to see what algorithm function supported.