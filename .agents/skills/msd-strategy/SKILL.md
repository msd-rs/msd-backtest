---
name: msd_strategy
description: guide user to write a strategy for msd backtest
tags: [msd, strategy, backtest]
version: 1.0
author: elsejj
---


This guide will help you to write a strategy for msd backtest.

# Technical stack

Programming language: Python 3.11+
Libraries: numpy, py-alpha-lib

# Coding Specification

You should create a subclass named `Strategy` of `mbt.Strategy` and implement the `execute` method. The skeleton of the code is shown below:

```python

import mbt
import numpy as np
import alpha as al

# Other libraries can be imported as needed

class Strategy(mbt.Strategy):

  def execute(self, ctx: mbt.Context):
    """
    execute the strategy, this method will be called for each bar.

    Args:
      ctx: context of the strategy 
    """
    # TODO: implement strategy, building buy/sell signals then call ctx.buy/ctx.sell
    pass
```

# API reference

## mbt.Context

- `ctx.data(name: str, /, symbol: str | None = None) -> np.ndarray`
  the primary method to get data, `name` is the data name, `symbol` is used only when comparison, keep `None` for no explicit requirements. `name` can be:
  - `open`, `high`, `low`, `close`, `volume`, `amount`, `vwap` on bar level
  - `ror` for rate of return since first bar
  - `ror_hold` for rate of return of holding since last buy after clearance.
  - `limited` used to check whether the bar had limit up/down, 0 for no limit, 1 for limit up, -1 for limit down
  - Financial basic fields (基本财务简表, updated quarterly):
    - `eps_basic`: basic earnings per share, 基本每股收益, 单位 元
    - `eps_diluted`: diluted earnings per share, 摊薄每股收益, 单位 元
    - `eps_deduct`: adjusted earnings per share, 扣非每股收益, 单位 元
    - `bps`: book value per share, 每股净资产, 单位 元
    - `net_profit`: net profit, 净利润, 单位 元
    - `np_parent_growth`: net profit growth, 净利润增长率, 单位 无
    - `net_profit_deduct`: adjusted net profit, 扣非净利润, 单位 元
    - `total_revenue`: total revenue, 营业总收入, 单位 元
    - `revenue_growth`: revenue growth, 营业总收入增长率, 单位 无
    - `ocfps`: operating cash flow per share, 每股经营现金流量, 单位 元
    - `undist_profit_ps`: undistributed profit per share, 每股未分配利润, 单位 元
    - `capital_reserve_ps`: capital reserve per share, 每股资本公积金, 单位 元
    - `gross_margin`: gross margin, 销售毛利率, 单位 无
    - `sales_cost_rate`: sales cost rate, 销售成本率, 单位 无
    - `net_margin`: net margin, 销售净利率, 单位 无
    - `op_margin`: operating margin, 营业利润率, 单位 无
    - `roe_diluted`: return on equity, 摊薄净资产收益率, 单位 无
    - `roe`: return on equity, 净资产收益率, 单位 无
    - `debt_asset_ratio`: debt asset ratio, 资产负债率, 单位 无
  - 
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
- KURTOSIS(input: np.ndarray[float], periods: int): Calculate rolling sample excess Kurtosis over a moving window  Uses adjusted Fisher formula (matches pandas): kurt = n(n+1)/((n-1)(n-2)(n-3)) * sum(((x-mean)/std)^4) - 3(n-1)^2/((n-2)(n-3)) Requires at least 4 valid values.
- LLV(input: np.ndarray[float], periods: int): Find lowest value in a preceding `periods` window
- LLVBARS(input: np.ndarray[float], periods: int): The number of periods that have passed since the array reached its periods period low
- LONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A < B, Current A >= B
- LWMA(input: np.ndarray[float], periods: int): Linear Weighted Moving Average  LWMA = SUM(Price * Weight) / SUM(Weight)
- MA(input: np.ndarray[float], periods: int): Simple Moving Average, also known as arithmetic moving average
- MIN_MAX_DIFF(input: np.ndarray[float], periods: int): Calculate rolling min-max difference (range) over a moving window  TS_MIN_MAX_DIFF = TS_MAX(x, d) - TS_MIN(x, d) Single-pass using two monotonic deques for efficiency.
- MOMENT(input: np.ndarray[float], periods: int, k: int): Calculate rolling k-th central moment over a moving window  MOMENT(x, d, k) = mean((x - mean)^k) over window of d periods. This is the raw (non-adjusted) sample moment. k=2 gives variance (population), k=3 gives raw third moment, etc.
- NEUTRALIZE(category: np.ndarray[float], input: np.ndarray[float]): Neutralize the effect of a categorical variable on a numeric variable
- PRODUCT(input: np.ndarray[float], periods: int): Calculate product of values in preceding `periods` window  If periods is 0, it calculates the cumulative product from the first valid value.
- RANK(input: np.ndarray[float], periods: int): Calculate rank in a sliding window with size `periods`  Uses min-rank method for ties (same as pandas rankdata method='min'). NaN values are treated as larger than all non-NaN values.
- RCROSS(a: np.ndarray[float], b: np.ndarray[float]): For 2 arrays A and B, return true if A[i-1] > B[i-1] and A[i] <= B[i] alias: death_cross, cross_le
- REF(input: np.ndarray[float], periods: int): Right shift input array by `periods`, r[i] = input[i - periods]
- REGBETA(y: np.ndarray[float], x: np.ndarray[float], periods: int): Calculate Regression Coefficient (Beta) of Y on X over a moving window  Beta = Cov(X, Y) / Var(X)
- REGRESI(y: np.ndarray[float], x: np.ndarray[float], periods: int): Calculate Regression Residual of Y on X over a moving window  Returns the residual of the last point: epsilon = Y - (alpha + beta * X)
- RLONGCROSS(a: np.ndarray[float], b: np.ndarray[float], n: int): For 2 arrays A and B, return true if previous N periods A > B, Current A <= B
- SCAN_ADD(input: np.ndarray[float], condition: np.ndarray[bool]): Conditional cumulative add: r[t] = r[t-1] + (cond[t] ? input[t] : 0)  Used for SELF-referencing alpha expressions with additive accumulation. Serial within each stock, parallel across stocks via rayon.
- SCAN_MUL(input: np.ndarray[float], condition: np.ndarray[bool]): Conditional cumulative multiply: r[t] = r[t-1] * (cond[t] ? input[t] : 1)  Used for SELF-referencing alpha expressions like GTJA #143. Serial within each stock, parallel across stocks via rayon.
- SKEWNESS(input: np.ndarray[float], periods: int): Calculate rolling sample Skewness over a moving window  Uses adjusted Fisher-Pearson formula (matches pandas): skew = n / ((n-1)(n-2)) * sum(((x-mean)/std)^3) Requires at least 3 valid values.
- SLOPE(input: np.ndarray[float], periods: int): Linear Regression Slope  Calculates the slope of the linear regression line for a moving window.
- SMA(input: np.ndarray[float], n: int, m: int): Exponential Moving Average (variant of well-known EMA) weight = m / n
- STDDEV(input: np.ndarray[float], periods: int): Calculate Standard Deviation over a moving window
- SUM(input: np.ndarray[float], periods: int): Calculate sum of values in preceding `periods` window  If periods is 0, it calculates the cumulative sum from the first valid value.
- SUMBARS(input: np.ndarray[float], amount: float): Calculate number of periods (bars) backwards until the sum of values is greater than or equal to `amount`
- SUMIF(input: np.ndarray[float], condition: np.ndarray[bool], periods: int): Calculate sum of values in preceding `periods` window where `condition` is true
- VAR(input: np.ndarray[float], periods: int): Calculate Variance over a moving window  Variance = (SumSq - (Sum^2)/N) / (N - 1)
- WEIGHTED_DELAY(input: np.ndarray[float], periods: int): Calculate weighted delay (exponentially weighted lag)  WEIGHTED_DELAY(x, k) = (k * x[t-1] + (k-1) * x[t-2] + ... + 1 * x[t-k]) / (k*(k+1)/2) This is essentially LWMA applied to the lagged (shifted by 1) series over k periods.
- ZSCORE(input: np.ndarray[float], periods: int): Calculate rolling Z-Score over a moving window  Z-Score = (x - mean) / stddev, computed over a rolling window of `periods`. Uses sample stddev (ddof=1) to match pandas.
  
  
