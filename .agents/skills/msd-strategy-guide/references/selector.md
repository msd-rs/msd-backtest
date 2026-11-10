
# Coding Specification

You should create a subclass named `Selector` of `mbt.Selector`. 

```python

import mbt
import numpy as np
import alpha as al
import polars as pl
from mbt.select import A_STOCKS_EXCLUDE_ST, Panel


# Other libraries can be imported as needed

class MySelector(mbt.Selector):

  def __init__(self):
    super().__init__()
    # TODO: add your custom parameters to __init__ method if needed.

  def step00(self, stocks: list[str]) -> list[str]:
    # TODO: write select logic here
```

- You will build a polars DataFrame named `factors` to select stocks as class member. It has `obj` (or `obj`) column, other columns in `factors` are according to the user's requirements.
- You write several `stepXX` methods according to the user's requirements to gradually narrow down the result set, and update the `factors`. The `stepXX` methods are executed in order, and each method will be called with the result of the previous method as input. You should carefully design each step, there are some rules:
  - Usually, it's a good idea to first filter by snapshot via `load_snapshot`, then do deeper analysis that needs historical series via `load_kline`.
  - Keep the number of stocks in each step as small as possible.
  - Add docstring to each `stepXX` method to explain what it does (it will be logged automatically during execution).
  - You should use caching to avoid fetching data multiple times.
- For `step00`, when incoming `stocks` is empty and user not specifies any initial obj rules, you should use `A_STOCKS_EXCLUDE_ST` as the initial set of stocks. There are some other initial stocks can be use depending on the user's requirements.
  - `A_STOCKS`: all A-share stocks
  - `A_STOCKS_EXCLUDE_ST`: all A-share stocks except ST and *ST stocks
  - `FOUNDS`: all funds
  - `ALL_STOCKS`: is a polars DataFrame that hold all stocks and can be used to select stocks by some it's columns.
    - `obj`: stock obj, prefix with market 'SH' | 'SZ', then stock code
    - `name`: stock name
    - `kind`: stock kind
    - `status`: stock status
- You can use `Selector.dp` (which is instance of `mbt.select.SelectorDataProvider`) to get data, only use it public API. The `SelectorDataProvider.date` is the date to perform selection, default to last, but usually your not need to use it explicitly. All `SelectorDataProvider` method is well documented with Python docstring, View it to learn how to use it get data. 
- Use `alpha-lib` library do calculating factor.
- Financial fields are named like `f001` can be referred to [financial](./references/financial.md)


## Using `Panel` (`mbt.select.Panel`)

`self.dp.load_kline(objs, lastN=100, fin_fields=[], fin_only_year=True, baseline=None)` returns a `Panel` object.

`Panel` wraps the 1D contiguous block buffer (`n_objs * bars` rows) into a `(n_objs, bars)` 2D view and **always carries `objs` metadata** (row `i` corresponds to `panel.objs[i]`).
- **Never** map results to your input `objs` list positionally — `load_kline` may reorder or deduplicate objs; always use `panel.objs`.
- **Recommended**: pass `baseline="SH000001"` to `load_kline` so a full-calendar index defines the shared time axis (the baseline block itself is automatically removed from the returned `Panel`).
- **Legacy compatibility**: `data, objs, bars = self.dp.load_kline(...)` triple unpacking is still supported.

### Price & Adjustment Semantics (复权与估值规范)

> [!WARNING] 复权口径与财务估值规则
> - **前复权价 (`close`, `open`, `high`, `low`)**: 仅用于连续时序与技术指标（如均线 `al.MA`、动量收益率、MACD）。
> - **未复权真实价格 (`raw_close`, `raw_open`, `raw_high`, `raw_low`, `price`)**: **必须**用于财务估值指标计算（如 PB、PE、真实市值）。
>   - 严禁使用前复权 `close` 计算 PB/PE/市值！例如某高送转股票真实股价为 50 元，每股净资产 `f004` 为 10 元，真实 PB 为 5.0；但前复权价可能缩放到 5 元，若错误计算 `close / f004` 将得到 0.5（严重失真的虚假低估）。
>   - **正确做法**：在 `load_kline` (Panel) 中使用 `panel.column("raw_close")` 或 `panel.last("raw_close")`；在 `load_snapshot` 中使用 `raw_close`（或 `close`，snapshot 中两者相同）。

- `load_kline` 提供完整的原始未复权价格：
  - `raw_open`, `raw_high`, `raw_low`, `raw_close`（以及兼容别名 `price`）
  - `raw_vwap`（成交金额 / 成交量）
- `load_snapshot` 同样提供：
  - `close` / `raw_close`, `open` / `raw_open`, `high` / `raw_high`, `low` / `raw_low`, `raw_vwap`
- **财务数据口径**:
  - Financial records in MSD are indexed by report period end date (`03-31`, `06-30`, `09-30`, `12-31`). When `fin_only_year=False` is used on dates between `01-01` and `04-30`, companies that have not yet published their annual report will naturally return their latest available Q3 (`09-30`) report. Do not confuse cumulative Q3 income/profit figures with full-year figures.

### Attributes & Metadata
- `panel.frame`: underlying `pl.DataFrame` of length `n_objs * bars`.
- `panel.objs`: `list[str]`, stock objs corresponding to the row order.
- `panel.bars`: `int`, number of K-line bars per obj. Note: when `self.dp.date` is near the start of database history, `panel.bars` may be smaller than the requested `lastN`; use `min(window, panel.bars)` when setting rolling windows or minimum valid bar thresholds.
- `panel.n_objs` / `len(panel)`: number of objs.
- `panel.pick(obj) -> int`: maps a stock obj to its row index in `(n_objs, bars)` (raises `KeyError` if not found).

### 2D & 1D Views
- `panel.buffer(name) -> np.ndarray`: 1D contiguous array `(n_objs * bars,)` in block layout, directly consumed by `alpha-lib`.
- `panel.column(name) -> np.ndarray`: 2D view `(n_objs, bars)` sharing memory with `buffer(name)`.
- `panel.last(name) -> np.ndarray`: value of the last bar for every obj, shape `(n_objs,)`.
- `panel.offset(name, k) -> np.ndarray`: value of the `k`-th bar (supports negative index like `-2`) for every obj, shape `(n_objs,)`.
- `panel.timeline() -> np.ndarray`: shared date axis of shape `(bars,)`.

### Vectorized `alpha-lib` Computation
Avoid per-obj Python loops; compute over the entire panel at once:
- `panel.compute(fn, *args, **kwargs) -> np.ndarray`: enters `panel.ctx()`, automatically resolves column name strings (e.g. `"close"`, `"dividend"`) into `panel.buffer(col)`, executes `fn`, and reshapes the output to `(n_objs, bars)`:
  ```python
  ma15 = panel.compute(al.MA, "close", 15)          # shape: (n_objs, bars)
  div_ttm = panel.compute(al.SUM, "dividend", 250)  # shape: (n_objs, bars)
  ```
- `with panel.ctx(flags=None):` / `panel.set_ctx(flags=None)`: binds `al.set_ctx(groups=panel.n_objs, flags=al.FLAG_SKIP_NAN)` to the current scope. Because `alpha-lib` group state is process-global and `load_kline` overwrites it, always use `panel.compute(...)` or `with panel.ctx():` when calling `alpha` functions directly:
  ```python
  with panel.ctx():
    close = panel.column("close")
    ma15 = al.MA(close, 15)
    cross = al.RCROSS(close, ma15)  # shape: (n_objs, bars)
  ```

### Valuation Factors & Market Cap (估值与市值计算)
计算估值与市值时，**务必使用未复权价 `raw_close`**，切勿使用 `close`（前复权价）：
```python
# 1. PB (市净率): 最新未复权收盘价 / 每股净资产 (f004)
pb = panel.last("raw_close") / panel.last("f004")

# 2. Total Market Cap (总市值): 最新未复权收盘价 * 总股本 (total_shares)
market_cap = panel.last("raw_close") * panel.last("total_shares")

# 3. Dividend Yield TTM (股息率): 近一年未复权每股分红 / 最新未复权收盘价
div_ttm = panel.compute(al.SUM, "dividend", 250)[:, -1]
div_yield = div_ttm / panel.last("raw_close")
```

### Multi-Period Resampling (`panel.resample`)
Convert daily K-line panels into weekly, monthly, quarterly, or yearly panels directly:
- `monthly_panel = panel.resample("1M")` (supports `"1W"`/`"W"`, `"1M"`/`"M"`, `"1Q"`/`"Q"`, `"1Y"`/`"Y"`, or Polars `dt.truncate` syntax like `"1w"`, `"1mo"`).
- Returns a standard `Panel` object with shape `(n_objs, new_bars)` and the exact same column schema.
- **OHLCV Aggregation**:
  - `open`: first valid open in period
  - `close`: last valid close in period
  - `high`: maximum high in period
  - `low`: minimum low in period
  - `volume`, `amount`, `dividend`, `transfer_shares`: summed over the period
  - other columns (such as `total_shares`): carried forward as the last valid value in period
  - `ts`: labeled as the last trading day of each period
- **Example Usage** (e.g. Monthly 3 consecutive Yang bars):
  ```python
  # Resample to monthly Panel and check 3 consecutive Yang bars
  m_panel = panel.resample("1M")
  m_open = m_panel.column("open")[:, -3:]
  m_close = m_panel.column("close")[:, -3:]
  is_yang_3m = (m_close > m_open).all(axis=1)
  ```

### Cross-Section Assembly & Debugging
- `panel.cross_section(**columns) -> pl.DataFrame`: packs multiple `(n_objs,)` 1D arrays into an `n_objs`-row `pl.DataFrame` with `"obj"` (`panel.objs`) as the first column. Ideal for building `self.factors` and filtering stocks:
  ```python
  panel = self.dp.load_kline(stocks, lastN=100, baseline="SH000001")
  ma15 = panel.compute(al.MA, "close", 15)
  with panel.ctx():
    broke = al.EXIST(al.RCROSS(panel.column("close"), ma15), 3)

  df = panel.cross_section(
    close=panel.last("close"),
    ma15=ma15[:, -1],
    broke=broke[:, -1],
  ).filter(pl.col("broke") > 0)

  self.factors = df
  return df["obj"].to_list()
  ```
- `panel.to_frame(key="enum" | "str") -> pl.DataFrame`: expands the panel into a long table with a `obj` column (`pl.Enum` by default to save memory, or `"str"`). Use **only** for debugging/exporting; keep calculations on `column()` / `last()` / `compute()` / `cross_section()`.


# Execution

After writing a strategy, you can run it to verify it works as expected.

## Test

```shell
[uv run] python -m mbt.select.cli <SELECTOR_NAME> # to run the selector
[uv run] python -m mbt.select.cli --help # to see all command args
```


`uv run` can be omitted current workspace not using `uv` as package manager.


## alpha-lib

refer to [algo.md](./algo.md) to see what algorithm function supported.
