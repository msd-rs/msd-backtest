---
name: write-strategy
description: guide user to write a strategy for msd backtest
tags: [msd, strategy, backtest]
version: 1.0
author: elsejj
---


This guide will help you to write a strategy for msd backtest.

# Technical stack

Programming language: Python 3.11+


# Preparation

- Ensure `msd-backtest[cli]` be installed use project package manager.
- Ensure `MSD_HOST` environment variable be set, can be find in `.env` file or test by shell command `env | grep MSD_HOST`. If not set, you can ask the user to set it first. When user can't provide it, you can ask and install it by [the guide](./references/msd.md).
- There should have a `./etc/fee.json` used calculate each trade fee, default can be 

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

You should create a subclass named `Strategy` of `mbt.Strategy` and implement the `execute` method. The skeleton of the code is shown below:

```python

import mbt
import numpy as np
import alpha as al

# Other libraries can be imported as needed

class Strategy(mbt.Strategy):

  def __init__(self):
    super().__init__()
    # TODO: add your custom parameters to __init__ method if needed.

  def execute(self, ctx: mbt.Context):
    """
    execute the strategy, this method will be called for each bar.

    Args:
      ctx: context of the strategy 
    """
    # TODO: implement strategy, building buy/sell signals then call ctx.buy/ctx.sell
    pass
```

## basic logic

`mbt` is optimized to run a strategy for multiple stocks's parallel, this will huge improve the backtest performance, it use
- `alpha-lib` as factor compute library, a parallel factor building algorithms
- `mbt.Context` hold context of strategy running, including:
  - data
  - stock list
  - step
  - see `mbt.RunnerContext`, which is an implementation of abstract `mbt.Context`, use attributes of `mbt.RunnerContext` is not suggested, use it when you really needed.
- when you to implement the `execute`, use any loop carefully, because it may too slow, vector computing should use preferentially.

# Execution

After writing a strategy, you can run it to verify it works as expected.

## Test

```shell
[uv run] python -m mbt.run.cli <STRATEGY_PATH> -s SH600000 -b 2025-01-01 -e 2025-12-31
```

`uv run` can be omitted current workspace not using `uv` as package manager.


# API reference

## mbt.Context

- `ctx.data(name: str, /, symbol: str | None = None) -> np.ndarray`
  the primary method to get data, `name` is the data name, `symbol` is used only when comparison, keep `None` for no explicit requirements. `name` can be:
  - `open`, `high`, `low`, `close`, `volume`, `amount`, `vwap` on bar level
  - `ror` for rate of return since first bar
  - `ror_hold` for rate of return of holding since last buy after clearance.
  - `limited` used to check whether the bar had reached the daily limit.  0 for no limit, 1 for limit up, -1 for limit down
  - `total_shares`: total shares, total stock shares, use to calculate total market capitalization etc.
  - `tradable_shares`: tradable shares, tradable stock shares, use to calculate tradable market capitalization, turnover etc.
  - Financial fields are named like `f001` can be referred to [financial](./references/financial.md)
- `ctx.buy(signals: np.ndarray, percent: float = 1.0)`
  `ctx.sell(signals: np.ndarray, percent: float = 1.0)`
  buy/sell a position, `signals` is a boolean array of signals, `True` means buy/sell, `percent` is the percent of cash/position to buy/sell.

## alpha-lib

When you need calculate any factor, you may to use alpha-lib first

refer to [algo.md](./references/algo.md) to see what algorithm function supported.