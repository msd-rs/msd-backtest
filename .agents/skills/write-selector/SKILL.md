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
- Ensure `MSD_HOST` environment variable be set, can be find in `.env` file or test by shell command `env | grep MSD_HOST`. If not set, you can ask the user to set it first.


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
- You can use `Selector.dp` (which is instance of `SelectorDataProvider`) to get data, only use it public API. The `SelectorDataProvider.date` is the date to perform selection, default to last, but usually your not need to use it explicitly, 
- Use `alpha-lib` library do calculating factor.



# Execution

After writing a strategy, you can run it to verify it works as expected.

## Test

```shell
[uv run] python -m mbt.select.cli <SELECTOR_NAME> # to run the selector
[uv run] python -m mbt.select.cli --help # to see all command args
```


`uv run` can be omitted current workspace not using `uv` as package manager.


# API reference

## SelectorDataProvider.load_kline

`load_kline(self, stocks: list[str], lastN: int = 100) -> dict[str, pl.DataFrame]`
  load `lastN` kline bars for `stocks`

Returns:
  A dict of [symbol] -> pl.DataFrame of lastN kline bars
  DataFrame must have columns 
    ['ts', 'open', 'high', 'low', 'close', 'amount', 'volume', 'limited' 'total_shares', 'tradable_shares']
  All prices had been forward-adjusted (spl.t, bonus, etc. already reflected).
  limited is the status of reach price limit of each bar, 0 for normal, -1 for limit down, 1 for limit up

## SelectorDataProvider.load_snapshot
`load_snapshot(self, symbols: list[str], fin_fields: list[str] = [], fin_only_year: bool = True) -> pl.DataFrame`
  Take a snapshot for the given symbols at 'snapshot_date', this is more faster than 
  load_kline and load_financial separately.

  User usually use this function to get the latest data for symbols.

  Args:
    symbols: symbols to snapshot
    fin_fields: fields to snapshot from financial data
    fin_only_year: whether to only snapshot annual financial data

  Returns:
    A DataFrame of snapshot for the given symbols, with columns combined from 
    `load_kline` and `load_financial` except the financial data's 'ts` column will be renamed to 'f_ts'

## SelectorDataProvider.load_financial
`load_financial(self, stocks: list[str], fields: list[str], only_year: bool = True, lastN: int = 12) -> dict[str, pl.DataFrame]`
  load `lastN` financial data for `symbols`
  Financial data is provided on a quarterly basis, with dates of 03-30, 06-30, 09-30, and 12-31. 
  When the only_year parameter is set to true, only the annual data (12-31) is selected.

Returns:
  A dict of [symbol] -> pl.DataFrame of lastN financial data
  DataFrame columns ['ts', *fields] 
  `ts` is the date of the financial data

  Financial fields are named like `f001` can be referred to [financial](./references/financial.md)

## alpha-lib

refer to [algo.md](./references/algo.md) to see what algorithm function supported.
