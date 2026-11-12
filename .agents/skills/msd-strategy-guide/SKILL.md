---
name: msd-strategy-guide
description: guide user to write a stock strategy use msd stack
---

# Technical stack

Programming language: Python 3.12+
Package manager: `uv` is preferred when it had installed, or use `pip` create a local virtual environment.
Database: msd, see [msd setup guide](./references/msd.md).
Algorithm Library: [py-alpha-lib](./references/algo.md), polars, numpy
Strategy Framework: msd-backtest


# Preparation

- Ensure `msd-backtest[cli]` be installed use project package manager.
- When your need to run in server mode, ensure `msd-backtest[server]` is installed. Normally, server mode is not required.
- Ensure `MSD_HOST` environment variable be set, can be find in `.env` file or test by shell command `env | grep MSD_HOST`. If not set, you can ask the user to set it first. When user can't provide it, you can ask and install it by [msd setup guide](./references/msd.md).
- `MSD_BT_FEE` is optional. `./etc/fee.json` can be used to customize trade fees, defaulting to standard rates if omitted:

# Write Strategy

The `msd-backtest` import name is `mbt`, it's well documented. use `python3 -m pydoc mbt` to read it.

There are 3 kinds of Strategy

- Selector: Used to build cross-sectional stock selection
- Trading: Use to build time series trading signals strategy, each stock has it's fund account, such strategies are often used to evaluate performance on a per-stock basis.
- Portfolio: Use to build  time series trading signals strategy, but share a portfolio account, such strategies are usually paired with a stock-selection strategy for portfolio rebalancing, and then the overall performance is evaluated.

So, first, please choose what will be build, then reading the relevant documentation to proceed with the build.

- [How to build a selector](./references/selector.md)
- [How to build a trading strategy](./references/trading.md)
- [How to build a portfolio trading strategy](./references/portfolio.md)


## Combining Selector and Strategy in a Single File

A single `.py` file can define both a `Selector` (subclass of `mbt.Selector`) and a `Strategy` (subclass of `mbt.Strategy` or `mbt.PortfolioStrategy`):

- When running `mbt.run.cli <STRATEGY_PATH>` without `-s`, it automatically runs the `Selector` in the file first to select the stock pool, then runs `Strategy` on the selected stocks.
- Arguments passed via `-a key=value` are automatically matched and dispatched to `Selector.__init__` and `Strategy.__init__` by parameter name.


## Official Strategy Templates (Cookbook)

Before writing a strategy from scratch, check the official verified templates in package `mbt.strategy_templates`:
- `mbt.strategy_templates.fundamental_timing`: Fundamental screening + trend timing (combined Selector + PortfolioStrategy, Davis Double Play / Fisher / Dreman style).
- `mbt.strategy_templates.multifactor_rotation`: Cross-sectional multi-factor periodic rotation (Value + Quality + Momentum, periodic rebalancing using `ctx.rebalance_objs`).
- `mbt.strategy_templates.cta_momentum`: CTA pure price-volume momentum and channel tracking (Donchian breakout + ATR trailing stop, zero-overhead in `execute()` using vectorization in `prepare()`).

You can inspect or import templates directly via `from mbt import strategy_templates`:
- Read source: `inspect.getsource(strategy_templates.fundamental_timing)` or `strategy_templates.get_template_source("fundamental_timing")`
- Get file path: `strategy_templates.fundamental_timing.__file__` or `strategy_templates.get_template_path("fundamental_timing")`

# Optimize Strategy

When you had written a strategy, user may ask you to need optimize it, such as adjust the parameters, choose time range, etc. You should read the [guide of optimize](./references/optimize.md) to do that.

If the user has not explicitly asked you to optimize, please do not do so on your own.
