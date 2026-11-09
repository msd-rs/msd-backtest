# Portfolio Rebalancing Strategy (`mbt.PortfolioStrategy`)

When the strategy needs to manage a **multi-stock portfolio sharing a single cash pool** and rebalance positions across stocks (rather than testing each stock independently), subclass `mbt.PortfolioStrategy` and use `ctx.rebalance(weights)` (or `ctx.buy` / `ctx.sell`):

```python
class Strategy(mbt.PortfolioStrategy):

  def __init__(self, period: int = 20, top_k: int = 5):
    super().__init__()
    self.period = period
    self.top_k = top_k

  def prepare(self, ctx: mbt.Context):
    # Pre-compute momentum factor across all stocks
    self.ma = al.MA(ctx.close, self.period)

  def execute(self, ctx: mbt.Context):
    # Call ctx.rebalance(weights) on rebalance bars; sells are executed first to free cash for buys
    ...
```

---

## 1. Execution Flow & Rebalance Mechanism

During portfolio backtesting (`PortfolioRunner`):
1. **Target Weights**: The strategy calls `ctx.rebalance(weights)` where `weights` is an array of target percentages for each stock. If sum of weights exceeds 1.0, it is normalized to 1.0.
2. **Two-Pass Execution**:
   - **Pass 1 (Sell / 调出)**: Stocks whose current holding value exceeds the target value (or target value is 0) are sold first. Selling proceeds (minus transaction fees) immediately return to the shared `cash` pool.
   - **Pass 2 (Buy / 调入)**: Available cash is allocated across stocks needing to increase positions. If total cash is insufficient to meet all target buys, allocations are scaled proportionally (`min(1.0, cash / deficit)`).
3. **Execution Constraints**:
   - **Slippage & Fees**: Buy price is adjusted up by `+slippage`, sell price down by `-slippage`. Fees are calculated dynamically based on date rules.
   - **Trading Unit**: Orders are rounded down to multiples of `unit` (default `100` shares).
   - **Price Limits**: Stocks hitting limit-down (`limited == -1`) cannot be sold; stocks hitting limit-up (`limited == 1`) cannot be bought.

---

## 2. Trade & Rebalance History (`PortfolioRecord`)

`PortfolioAccount` tracks every individual stock trade during rebalancing and signal execution.

### Accessing History

- `account.records`: A list of `PortfolioRecord` dataclass instances.
- `account.trades_df()` (or `account.records_df()`): Returns a **Polars DataFrame** containing all trade records.

### Record Schema

Each `PortfolioRecord` contains the following fields:

| Field | Type | Description |
| :--- | :--- | :--- |
| `bar` | `int` | Current K-line bar index ($0, 1, \dots$). |
| `ts` | `int` | Unix timestamp of the transaction date. |
| `group` | `int` | Group index of the stock in the data provider. |
| `obj` | `str` | Stock symbol (e.g. `SH600000`, `SZ000001`). |
| `action` | `str` | `"BUY"` (调入/买入) or `"SELL"` (调出/卖出). |
| `shares` | `float` | Executed quantity in shares. |
| `deal_price` | `float` | Benchmark execution price before slippage. |
| `exec_price` | `float` | Actual executed price after slippage. |
| `amount` | `float` | Executed transaction turnover value (`shares * exec_price`). |
| `fee` | `float` | Commissions and taxes deducted for this transaction. |
| `cash` | `float` | Remaining shared cash balance after this transaction. |
| `pos_shares` | `float` | Remaining held shares for this stock after this transaction. |

`PortfolioRecord` supports both attribute access (`rec.obj`) and dictionary-style access (`rec["obj"]`).

---

## 3. Analysis Examples

### Inspecting Trades as a DataFrame

```python
acc = PortfolioAccount(1000000.0, dp)
runner = PortfolioRunner(acc)
runner.run(strategy)

# Convert trade history to Polars DataFrame
trades = acc.trades_df()
print(trades)
```

Example output:
```text
shape: (4, 12)
┌─────┬────────────┬───────┬──────────┬────────┬────────┬────────────┬────────────┬──────────┬──────┬──────────┬────────────┐
│ bar ┆ ts         ┆ group ┆ obj      ┆ action ┆ shares ┆ deal_price ┆ exec_price ┆ amount   ┆ fee  ┆ cash     ┆ pos_shares │
│ --- ┆ ---        ┆ ---   ┆ ---      ┆ ---    ┆ ---    ┆ ---        ┆ ---        ┆ ---      ┆ ---  ┆ ---      ┆ ---        │
│ i64 ┆ i64        ┆ i64   ┆ str      ┆ str    ┆ f64    ┆ f64        ┆ f64        ┆ f64      ┆ f64  ┆ f64      ┆ f64        │
╞═════╪════════════╪═══════╪══════════╪════════╪════════╪════════════╪════════════╪══════════╪══════╪══════════╪════════════╡
│ 1   ┆ 1704153600 ┆ 0     ┆ SH600000 ┆ BUY    ┆ 500.0  ┆ 20.0       ┆ 20.0       ┆ 10000.0  ┆ 5.0  ┆ 899995.0 ┆ 500.0      │
│ 2   ┆ 1704240000 ┆ 0     ┆ SH600000 ┆ SELL   ┆ 500.0  ┆ 22.0       ┆ 22.0       ┆ 11000.0  ┆ 5.0  ┆ 910990.0 ┆ 0.0        │
│ 2   ┆ 1704240000 ┆ 1     ┆ SZ000001 ┆ BUY    ┆ 1000.0 ┆ 10.0       ┆ 10.0       ┆ 10000.0  ┆ 5.0  ┆ 900985.0 ┆ 1000.0     │
└─────┴────────────┴───────┴──────────┴────────┴────────┴────────────┴────────────┴──────────┴──────┴──────────┴────────────┘
```

### Filtering In/Out Flow and Turnover Analysis

```python
import polars as pl

# 1. Separate Inflow (调入) and Outflow (调出)
buys = trades.filter(pl.col("action") == "BUY")
sells = trades.filter(pl.col("action") == "SELL")

# 2. Total turnover and fees per stock
obj_summary = trades.group_by("obj").agg([
  pl.col("amount").sum().alias("total_turnover"),
  pl.col("fee").sum().alias("total_fees"),
  pl.len().alias("trade_count"),
])

# 3. Bar-level rebalance turnover (换手金额与手续费)
rebalance_summary = trades.group_by("bar").agg([
  pl.col("amount").sum().alias("rebalance_amount"),
  pl.col("fee").sum().alias("rebalance_fees"),
  pl.col("obj").n_unique().alias("stocks_adjusted"),
])
```


Apart from the different base class and calling `rebalance` at the appropriate time, everything else can be referenced from [How to build a trading strategy](./trading.md).
