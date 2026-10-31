---
name: write-selector
description: guide to write a selector to select stocks from some initial set
tags: [msd, strategy, backtest]
version: 1.0
author: elsejj
---


This guide will help you to write a selector to select stocks.

# Technical stack

Programming language: Python 3.11+


# Preparation

- Ensure `msd-backtest[cli]` be installed use project package manager.
- Ensure `MSD_HOST` environment variable be set, can be find in `.env` file or test by shell command `env | grep MSD_HOST`. If not set, you can ask the user to set it first. When user can't provide it, you can ask and install it by [the guide](./references/msd.md).


# Coding Specification

You should create a subclass named `Selector` of `mbt.Selector`. 

```python

import mbt
import numpy as np
import alpha as al
import polars as pd
from mbt.select import A_STOCKS_EXCLUDE_ST


# Other libraries can be imported as needed

class MySelector(mbt.Selector):

  def __init__(self):
    super().__init__()
    # TODO: add your custom parameters to __init__ method if needed.

  def step00(self, stocks: list[str]) -> list[str]:
    # TODO: write select logic here
```

- You will build a polars DataFrame named 'factors' to select stocks as class member. It have `symbol` column, other columns in `factor` is according to the user's requirements.
- You write several `stepXX` methods according to the user's requirements to gradually narrow down the result set, and upl.te the `factors`. The `stepXX` methods are executed in order, and each method will be called with the result of the previous method as input. You should carefully design each step, there are some rules:
  - Usually, it's a good idea to first by snapshot by `load_snapshot`, then other deep analyze that need more data.
  - Keep the number of stocks in each step as small as possible.
  - Add comments to each step explain what you do.
  - You should use caching to avoid fetching data multiple times.
- For `step00`, when incoming `stocks` is empty and user not specifies any initial symbol rules, you should use `A_STOCKS_EXCLUDE_ST` as the initial set of stocks. There are some other initial stocks can be use depending on the user's requirements.
  - `A_STOCKS`: all A-share stocks
  - `A_STOCKS_EXCLUDE_ST`: all A-share stocks except ST and *ST stocks
  - `FOUNDS`: all funds
  - `ALL_STOCKS`: is a polars DataFrame that hold all stocks and can be used to select stocks by some it's columns.
    - `symbol`: stock symbol, prefix with market 'SH' | 'SZ', then stock code
    - `name`: stock name
    - `kind`: stock kind
    - `status`: stock status
- You can use `Selector.dp` (which is instance of `mbt.select.SelectorDataProvider`) to get data, only use it public API. The `SelectorDataProvider.date` is the date to perform selection, default to last, but usually your not need to use it explicitly. All `SelectorDataProvider` method is well documented with Python docstring, View it to learn how to use it get data. 
- Use `alpha-lib` library do calculating factor.
- Financial fields are named like `f001` can be referred to [financial](./references/financial.md)


# Execution

After writing a strategy, you can run it to verify it works as expected.

## Test

```shell
[uv run] python -m mbt.select.cli <SELECTOR_NAME> # to run the selector
[uv run] python -m mbt.select.cli --help # to see all command args
```


`uv run` can be omitted current workspace not using `uv` as package manager.



## alpha-lib

refer to [algo.md](./references/algo.md) to see what algorithm function supported.
