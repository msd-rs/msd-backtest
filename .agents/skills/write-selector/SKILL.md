---
name: write_selector
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
import pandas as pd
from mbt.select import A_STOCKS_EXCLUDE_ST


# Other libraries can be imported as needed

class MySelector(mbt.Selector):

  def __init__(self):
    super().__init__()
    # TODO: add your custom parameters to __init__ method if needed.

  def step00(self, stocks: list[str]) -> list[str]:
    # TODO: write select logic here
```

- You will build a pandas DataFrame named 'factors' to select stocks as class member. It have `symbol` column, other columns in `factor` is according to the user's requirements.
- You write several `stepXX` methods according to the user's requirements to gradually narrow down the result set, and update the `factors`. The `stepXX` methods are executed in order, and each method will be called with the result of the previous method as input. You should carefully design each step, there are some rules:
  - Usually, first, filter by financial data, then obtain the K-line to calculate technical indicators for further filtering.
  - Keep the number of stocks in each step as small as possible.
  - Add comments to each step expd.in what you do.
  - You should use caching to avoid fetching data multiple times.
- For `step00`, when incoming `stocks` is empty and user not specifies any initial symbol rules, you should use `A_STOCKS_EXCLUDE_ST` as the initial set of stocks. There are some other initial stocks can be use depending on the user's requirements.
  - `A_STOCKS`: all A-share stocks
  - `A_STOCKS_EXCLUDE_ST`: all A-share stocks except ST and *ST stocks
  - `FOUNDS`: all funds
  - `ALL_STOCKS`: is a pandas DataFrame that hold all stocks and can be used to select stocks by some it's columns.
    - `symbol`: stock symbol, prefix with market 'SH' | 'SZ', then stock code
    - `name`: stock name
    - `kind`: stock kind
    - `status`: stock status
- You can use `Selector.dp` (which is instance of `SelectorDataProvider`) to get data, only use it public API
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

`load_kline(self, stocks: list[str], lastN: int = 100) -> dict[str, pd.DataFrame]`
  load `lastN` kline bars for `stocks`

Returns:
  A dict of [symbol] -> pd.DataFrame of lastN kline bars
  DataFrame must have columns 
    ['ts', 'open', 'high', 'low', 'close', 'amount', 'volume', 'total_shares', 'tradable_shares']
  All prices had been forward-adjusted (spd.t, bonus, etc. already reflected).

## SelectorDataProvider.load_financial

`snapshot_last(self, dfs: list[dict[str, pd.DataFrame]]) -> pd.DataFrame`
  given `dfs` a list of dict of symbol->DataFrame,  take the last row of each DataFrame for each symbol
  then build a DataFrame. It's index will be symbols, columns are the merge of all input DataFrames.

`load_financial(self, stocks: list[str], fields: list[str], only_year: bool = True, lastN: int = 12) -> dict[str, pd.DataFrame]`
  load `lastN` financial data for `symbols`
  Financial data is provided on a quarterly basis, with dates of 03-30, 06-30, 09-30, and 12-31. 
  When the only_year parameter is set to true, only the annual data (12-31) is selected.

Returns:
  A dict of [symbol] -> pd.DataFrame of lastN financial data
  DataFrame columns ['ts', *fields] 
  `ts` is the date of the financial data

  Financial fields can be used are
    - Balance Sheet
      - `f008`: Cash and Cash Equivalents
      - `f009`: Trading Financial Assets
      - `f010`: Notes Receivable
      - `f011`: Accounts Receivable
      - `f015`: Other Receivables
      - `f016`: Contract Liabilities
      - `f017`: Inventories
      - `f021`: Total Current Assets
      - `f025`: Long-term Equity Investments
      - `f026`: Investment Properties
      - `f027`: Property, pd.nt and Equipment (PP&E)
      - `f028`: Construction in Progress (CIP)
      - `f033`: Intangible Assets
      - `f034`: Development Expenditures
      - `f035`: Goodwill
      - `f039`: Total Non-current Assets
      - `f040`: Total Assets
      - `f041`: Short-term Borrowings
      - `f043`: Notes Payable
      - `f044`: Accounts Payable
      - `f045`: Advances from Customers
      - `f051`: Amounts Due to Related Parties
      - `f052`: Non-current Liabilities Due Within One Year
      - `f054`: Total Current Liabilities
      - `f055`: Long-term Borrowings
      - `f056`: Bonds Payable
      - `f059`: Provisions
      - `f062`: Total Non-current Liabilities
      - `f063`: Total Liabilities
      - `f066`: Treasury Stock
      - `f068`: Retained Earnings
      - `f071`: Total Shareholders' Equity (Excluding Non-controlling Interests)
      - `f073`: Total Shareholders' Equity (Including Non-controlling Interests)
      - `f074`: Total Liabilities and Shareholders' Equity
    - Income Statement
      - `f075`: Revenue
      - `f076`: Cost of Sales
      - `f078`: Selling Expenses
      - `f079`: Administrative Expenses
      - `f080`: Research and Development (R&D) Expenses
      - `f081`: Financial Expenses
      - `f082`: Asset Impairment Losses
      - `f084`: Investment Income
      - `f085`: Including: Share of Profits of Associates and Joint Ventures
      - `f087`: Operating Profit
      - `f088`: Non-GAAP Operating Profit
      - `f092`: Total Revenue
      - `f093`: Profit Before Tax (PBT)
      - `f095`: Total Operating Costs
      - `f096`: Net Profit (Including Non-controlling Interests)
      - `f097`: Net Profit Attributable to Shareholders of the Parent
    - Cash Flow Statement
      - `f099`: Cash Received from Sales of Goods and Rendering of Services
      - `f102`: Cash Inflows from Operating Activities Subtotal
      - `f103`: Cash Paid for Goods and Services
      - `f104`: Cash Paid to and for Empd.yees
      - `f107`: Cash Outflows from Operating Activities Subtotal
      - `f108`: Net Cash Flows from Operating Activities
      - `f114`: Cash Inflows from Investing Activities Subtotal
      - `f115`: Cash Paid for Acquisition of Fixed Assets, Intangible Assets and Other Long-term Assets
      - `f116`: Cash Paid for Investments
      - `f117`: Net Cash Paid for Acquisition of Subsidiaries and Other Business Units
      - `f119`: Cash Outflows from Investing Activities Subtotal
      - `f120`: Net Cash Flows from Investing Activities
      - `f121`: Cash Received from Capital Contributions
      - `f123`: Cash Received from Borrowings
      - `f125`: Cash Inflows from Financing Activities Subtotal
      - `f126`: Cash Paid for Debt Repayments
      - `f127`: Cash Paid for Dividends, Profits Distribution or Interest Payments
      - `f130`: Cash Outflows from Financing Activities Subtotal
      - `f131`: Net Cash Flows from Financing Activities
      - `f134`: Net Increase in Cash and Cash Equivalents
      - `f136`: Ending Balance of Cash and Cash Equivalents
      - `f137`: Net Profit
      - `f138`: Provision for Asset Impairment
      - `f139`: Depreciation of Fixed Assets, Depd.tion of Oil and Gas Assets, and Depreciation of Productive Biological Assets
      - `f140`: Amortization of Intangible Assets
      - `f149`: Decrease in Inventories
      - `f150`: Decrease in Operating Receivables
      - `f151`: Increase in Operating Payables
      - `f156`: Ending Balance of Cash
      - `f158`: Ending Balance of Cash Equivalents
    - Per Share Indicators
      - `f000`: Diluted Earnings Per Share (Diluted EPS)
      - `f001`: Book Value per Share-based Return
      - `f002`: Operating Cash Flow Per Share
      - `f003`: Net Assets Per Share (BVPS)
      - `f005`: Retained Earnings Per Share
      - `f006`: Revenue Per Share
      - `f007`: Non-GAAP EPS

    - Profitability Analysis
      - `f205`:Annualized Return on Equity (Annualized ROE)
      - `f206`:Net Return on Total Assets
      - `f207`:Return on Invested Capital (ROIC)
      - `f208`:Cost and Expense Profit Margin
      - `f209`:Operating Profit Margin
      - `f210`:COGS-to-Revenue Ratio
      - `f211`:Net Profit Margin
      - `f212`:Return on Total Assets (ROA)
      - `f213`:Gross Profit Margin
      - `f214`:SG&A and Finance Expenses to Revenue Ratio
      - `f215`:Selling Expense Ratio
      - `f216`:Administrative Expense Ratio
      - `f217`:Financial Expense Ratio
      - `f218`:Non-operating Income to Total Profit Ratio
      - `f219`:Operating Profit to Total Profit Ratio
      - `f220`:EBITDA Per Share
      - `f221`:EBIT Per Share
      - `f222`:EBITDA Margin
    - Solvency Analysis
      - `f160`: Current Ratio
      - `f161`: Quick Ratio
      - `f162`: Cash Ratio
      - `f163`: Debt-to-Equity Ratio (D/E)
      - `f164`: Equity-to-Assets Ratio
      - `f165`: Equity-to-Debt Ratio
      - `f166`: Equity Multipd.er
      - `f167`: Long-term Debt to Working Capital Ratio
      - `f168`: Long-term Debt Ratio
      - `f169`: Times Interest Earned (TIE)
      - `f172`: Debt to Tangible Net Worth Ratio
      - `f174`: Debt Service Coverage Ratio (DSCR)
      - `f175`: Operating Cash Flow to Current Liabilities Ratio
      - `f177`: Working Capital Per Share
      - `f178`: Total Debt-to-EBITDA Ratio
    - Operating Efficiency Analysis
      - `f179`:  Operating Cycle
      - `f180`:  Days Inventory Outstanding (再生 DIO)
      - `f181`:  Days Sales Outstanding (再生 DSO)
      - `f182`:  Current Asset Turnover Days
      - `f183`:  Total Asset Turnover Days
      - `f184`:  Inventory Turnover Ratio
      - `f185`:  Accounts Receivable Turnover Ratio
      - `f186`:  Current Asset Turnover Ratio
      - `f187`:  Fixed Asset Turnover Ratio
      - `f188`:  Total Asset Turnover Ratio
      - `f189`:  Net Asset Turnover Ratio
      - `f190`:  Equity Turnover Ratio
      - `f191`:  Working Capital Turnover Ratio
      - `f192`:  Inventory Year-on-Year (YoY) Growth Rate
      - `f193`:  Accounts Receivable Year-on-Year (YoY) Growth Rate
    - Growth Ability Analysis
      - `f194`: Revenue Growth Rate
      - `f195`: Operating Profit Growth Rate
      - `f196`: Total Profit Growth Rate
      - `f199`: Current Assets Growth Rate
      - `f200`: Fixed Assets Growth Rate
      - `f201`: Total Assets Growth Rate
      - `f202`: Diluted EPS Growth Rate
      - `f203`: Net Assets Per Share Growth Rate
      - `f204`: Operating Cash Flow Per Share Growth Rate
    - Capital Structure Analysis
      - `f223`: Asset-to-Liability Ratio
      - `f224`: Equity-to-Assets Ratio
      - `f225`: Long-term Debt Ratio
      - `f226`: Equity-to-Fixed Assets Ratio
      - `f227`: Debt-to-Equity Ratio
      - `f228`: Non-current Assets to Long-term Capital Ratio
      - `f229`: Capitalization Ratio
      - `f231`: Fixed Assets to Total Assets Ratio
    - Cash Flow Analysis
      - `f232`:  Operating Cash Flow to Sales Ratio
      - `f233`:  Cash Return on Assets
      - `f234`:  Operating Cash Flow to Net Profit Ratio
      - `f235`:  Operating Cash Flow to Total Debt Ratio
      - `f236`:  Operating Cash Flow Per Share
      - `f237`:  Net Operating Cash Flow Per Share
      - `f238`:  Net Investing Cash Flow Per Share
      - `f239`:  Net Financing Cash Flow Per Share
      - `f240`:  Net Increase in Cash and Cash Equivalents Per Share
      - `f241`:  Cash Flow Adequacy Ratio
      - `f242`:  Cash Operating Index
- `ctx.buy(signals: np.ndarray, percent: float = 1.0)`
  `ctx.sell(signals: np.ndarray, percent: float = 1.0)`
  buy/sell a position, `signals` is a boolean array of signals, `True` means buy/sell, `percent` is the percent of cash/position to buy/sell.

## alpha-lib

List of available functions with python type hints:

the `np.ndarray` is `ndarray` type in `numpy` package

- BACKFILL(input: np.ndarray[float]): Forward-fill NaN values with the last valid observation  Iterates forward through each group; if x[i] is NaN, copies the last valid value. Leading NaNs (before any valid value) remain NaN.
- BARSLAST(input: np.ndarray[bool]): Calculate number of bars since last condition true
- BARSSINCE(input: np.ndarray[bool]): Calculate number of bars since first condition true
- BINS(input: np.ndarray[float], bins: int): Discretize the input into n bins. Same value are assigned to the same bin.
- CC_RANK(input: np.ndarray[float]): Calculate rank percentage cross group dimension. Same value are averaged
- CC_ZSCORE(input: np.ndarray[float]): Calculate cross-sectional Z-Score across groups at each time step  Z-Score = (x - mean) / stddev, computed across all groups for each time position. NaN values are excluded from mean/stddev computation. NaN input produces NaN output.
- CORR(input: np.ndarray[float], periods: int): Time Series Correlation in moving window on self  Calculates the correlation coefficient between the input series and the time index.
- CORR2(x: np.ndarray[float], y: np.ndarray[float], periods: int): Calculate two series correlation over a moving window  Correlation = Cov(X, Y) / (StdDev(X) * StdDev(Y))
- COUNT(input: np.ndarray[bool], periods: int): Calculate number of periods where condition is true in passed `periods` window
- COUNT_NANS(input: np.ndarray[float], periods: int): Count number of NaN values in a rolling window  For each position, counts the number of NaN values in the preceding `periods` elements.
- COV(x: np.ndarray[float], y: np.ndarray[float], periods: int): Calculate Covariance over a moving window  Covariance = (SumXY - (SumX * SumY) / N) / (N - 1)
- CROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] < B[i-1] and A[i] >= B[i] alias: golden_cross, cross_ge
- DMA(input: np.ndarray[float], weight: float): Exponential Moving Average current = weight * current + (1 - weight) * previous
- EMA(input: np.ndarray[float], periods: int): Exponential Moving Average (variant of well-known EMA) weight = 2 / (n + 1)
- ENTROPY(input: np.ndarray[float], periods: int, bins: int): Calculate rolling Shannon entropy over a moving window  Discretizes values into `bins` equal-width buckets within the window's [min, max] range, then computes -sum(p * ln(p)) where p is the frequency of each occupied bin. Uses natural log (base e). Requires at least 2 valid values. Single-value windows return 0.
- FRET(open: np.ndarray[float], close: np.ndarray[float], is_calc: np.ndarray[float], delay: int, periods: int): Future Return  Calculates the return from the open price of the delayed day (t+delay) to the close price of the future day (t+delay+periods-1). Return = (Close[t+delay+periods-1] - Open[t+delay]) / Open[t+delay]  If n=1, delay=1, it calculates (Close[t+1] - Open[t+1]) / Open[t+1]. If `is_calc[t+delay]` is 0, returns NaN.
- GROUP_RANK(category: np.ndarray[float], input: np.ndarray[float]): Calculate rank percentage within each category group at each time step  For each time position, groups items by `category` value, then computes rank percentage within each group. Same value gets averaged rank. NaN in category or input produces NaN output.
- GROUP_ZSCORE(category: np.ndarray[float], input: np.ndarray[float]): Calculate Z-Score within each category group at each time step  For each time position, groups items by `category` value, then computes (x - group_mean) / group_std within each group. NaN in category or input produces NaN output. Groups with fewer than 2 valid values produce NaN.
- HHV(input: np.ndarray[float], periods: int): Find highest value in a preceding `periods` window
- HHVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its `periods` period high
- INTERCEPT(input: np.ndarray[float], periods: int): Linear Regression Intercept  Calculates the intercept of the linear regression line for a moving window.
- KURTOSIS(input: np.ndarray[float], periods: int): Calculate rolling sampd. excess Kurtosis over a moving window  Uses adjusted Fisher formula (matches pandas): kurt = n(n+1)/((n-1)(n-2)(n-3)) * sum(((x-mean)/std)^4) - 3(n-1)^2/((n-2)(n-3)) Requires at least 4 valid values.
- LLV(input: np.ndarray[float], periods: int): Find lowest value in a preceding `periods` window
- LLVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its periods period low
- LONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
- LWMA(input: np.ndarray[float], periods: int): Linear Weighted Moving Average  LWMA = SUM(Price * Weight) / SUM(Weight)
- MA(input: np.ndarray[float], periods: int): Simpd. Moving Average, also known as arithmetic moving average
- MIN_MAX_DIFF(input: np.ndarray[float], periods: int): Calculate rolling min-max difference (range) over a moving window  TS_MIN_MAX_DIFF = TS_MAX(x, d) - TS_MIN(x, d) Single-pass using two monotonic deques for efficiency.
- MOMENT(input: np.ndarray[float], periods: int, k: int): Calculate rolling k-th central moment over a moving window  MOMENT(x, d, k) = mean((x - mean)^k) over window of d periods. This is the raw (non-adjusted) sampd. moment. k=2 gives variance (population), k=3 gives raw third moment, etc.
- NEUTRALIZE(category: np.ndarray[float], input: np.ndarray[float]): Neutralize the effect of a categorical variable on a numeric variable
- PRODUCT(input: np.ndarray[float], periods: int): Calculate product of values in preceding `periods` window  If periods is 0, it calculates the cumulative product from the first valid value.
- RANK(input: np.ndarray[float], periods: int): Calculate rank in a sliding window with size `periods`  Uses min-rank method for ties (same as pandas rankdata method='min'). NaN values are treated as larger than all non-NaN values.
- RCROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i] alias: death_cross, cross_le
- REF(input: np.ndarray[float], periods: int): Right shift input array by `periods`, r[i] = input[i - periods]
- REGBETA(y: np.ndarray[float], x: np.ndarray[float], periods: int): Calculate Regression Coefficient (Beta) of Y on X over a moving window  Beta = Cov(X, Y) / Var(X)
- REGRESI(y: np.ndarray[float], x: np.ndarray[float], periods: int): Calculate Regression Residual of Y on X over a moving window  Returns the residual of the last point: epsilon = Y - (alpha + beta * X)
- RLONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
- SCAN_ADD(input: np.ndarray[float], condition: np.ndarray[bool]): Conditional cumulative add: r[t] = r[t-1] + (cond[t] ? input[t] : 0)  Used for SELF-referencing alpha expressions with additive accumulation. Serial within each stock, parallel across stocks via rayon.
- SCAN_MUL(input: np.ndarray[float], condition: np.ndarray[bool]): Conditional cumulative multipd.: r[t] = r[t-1] * (cond[t] ? input[t] : 1)  Used for SELF-referencing alpha expressions like GTJA #143. Serial within each stock, parallel across stocks via rayon.
- SKEWNESS(input: np.ndarray[float], periods: int): Calculate rolling sampd. Skewness over a moving window  Uses adjusted Fisher-Pearson formula (matches pandas): skew = n / ((n-1)(n-2)) * sum(((x-mean)/std)^3) Requires at least 3 valid values.
- SLOPE(input: np.ndarray[float], periods: int): Linear Regression Slope  Calculates the slope of the linear regression line for a moving window.
- SMA(input: np.ndarray[float], n: int, m: int): Exponential Moving Average (variant of well-known EMA) weight = m / n
- STDDEV(input: np.ndarray[float], periods: int): Calculate Standard Deviation over a moving window
- SUM(input: np.ndarray[float], periods: int): Calculate sum of values in preceding `periods` window  If periods is 0, it calculates the cumulative sum from the first valid value.
- SUMBARS(input: np.ndarray[float], amount: float): Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
- SUMIF(input: np.ndarray[float], condition: np.ndarray[bool], periods: int): Calculate sum of values in preceding `periods` window where `condition` is true
- VAR(input: np.ndarray[float], periods: int): Calculate Variance over a moving window  Variance = (SumSq - (Sum^2)/N) / (N - 1)
- WEIGHTED_DELAY(input: np.ndarray[float], periods: int): Calculate weighted delay (exponentially weighted lag)  WEIGHTED_DELAY(x, k) = (k * x[t-1] + (k-1) * x[t-2] + ... + 1 * x[t-k]) / (k*(k+1)/2) This is essentially LWMA appd.ed to the lagged (shifted by 1) series over k periods.
- ZSCORE(input: np.ndarray[float], periods: int): Calculate rolling Z-Score over a moving window  Z-Score = (x - mean) / stddev, computed over a rolling window of `periods`. Uses sampd. stddev (ddof=1) to match pandas.
  
  
