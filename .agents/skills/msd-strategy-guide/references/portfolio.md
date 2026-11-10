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

  def finalize(self, ctx: mbt.Context) -> dict | None:
    # Post-backtest diagnostics (e.g. exit reasons: stop_loss, discount_closed, leverage_up)
    return {
      "exit_reasons": self.exit_reasons,
    }
```

---

## 1. Execution Flow & Rebalance Mechanism

During portfolio backtesting (`PortfolioRunner`):
1. **Target Weights & Allocation**:
   - The strategy can call `ctx.rebalance(weights)` with target weights across stocks.
   - Or call `ctx.buy` / `ctx.sell` with individual stock symbols or percentage signals. Cash is dynamically allocated based on available cash (`min(1.0, total_req) * cash`).
2. **Two-Pass Execution**:
   - **Pass 1 (Sell / 调出)**: Sells are executed first. Selling proceeds (minus transaction fees) immediately return to the shared `cash` pool.
   - **Pass 2 (Buy / 调入)**: Available cash is allocated across stocks needing to increase positions. If total cash requested exceeds available cash, budgets are scaled proportionally.
3. **Execution Constraints**:
   - **Slippage & Fees**: Buy price is adjusted up by `+slippage`, sell price down by `-slippage`. Fees are calculated dynamically based on date rules.
   - **Trading Unit**: Orders are rounded down to multiples of `unit` (default `100` shares).
   - **Price Limits**: Stocks hitting limit-down (`limited == -1`) cannot be sold; stocks hitting limit-up (`limited == 1`) cannot be bought.

---

## 2. Dynamic Universe & Natural Liquidation (自然平仓 / Runoff Mode)

In a two-layer architecture, a `Selector` periodically filters a broad universe into an active whitelist (e.g. Top 30 fundamental/momentum stocks), while daily strategy logic runs tactical trading within this dynamic pool.

### Core Primitives

- **Set Active Whitelist**:
  ```python
  ctx.set_universe(["SH600000", "SZ000001", ...])
  ```
- **Execute Selector Safely (No-Lookahead)**:
  ```python
  # Executes selector on historical slice up to current bar i (inclusive)
  new_whitelist = ctx.run_selector(self.selector, lastN=120)
  ctx.set_universe(new_whitelist)
  ```
- **Symbol-Based Trading (No Boolean Mask Required)**:
  ```python
  ctx.buy("SH600000", percent=0.2)             # Buy single stock with 20% budget
  ctx.buy(["SH600000", "SZ000001"], percent=0.2)
  ctx.sell("SH600000")                         # Sell 100% of held shares
  ctx.sell(["SH600000", "SZ000001"])
  ```
- **Automatic Whitelist Guard**:
  - `ctx.buy(...)` automatically shields any stock not in `ctx.universe` from being bought.
  - `ctx.sell(...)` does **not** filter by whitelist, allowing stocks dropped from the universe to be gracefully liquidated.
- **Account State Probes**:
  ```python
  ctx.account.held_objs                         # List of all currently held stocks: ['SH600000', ...]
  ctx.account.has_position("SH600000")         # True if shares > 0
  ctx.account.position("SH600000")             # Position object (shares, price, value)
  ctx.account.runoff_objs(ctx.universe)         # Stocks held but no longer in active universe (Runoff)
  ```

### Dual-Layer Strategy Template

```python
class DualLayerStrategy(mbt.PortfolioStrategy):

  def __init__(self, selector: mbt.Selector, reselect_period: int = 20):
    super().__init__()
    self.selector = selector
    self.reselect_period = reselect_period

  def execute(self, ctx: mbt.PortfolioContext):
    # 1. Macro layer: Periodically refresh active whitelist
    if ctx.bar % self.reselect_period == 0 or not ctx.universe:
      new_pool = ctx.run_selector(self.selector, lastN=120)
      ctx.set_universe(new_pool)

    # 2. Micro layer: Daily exit & risk control (monitors ALL held stocks, including Runoff stocks)
    for obj in ctx.account.held_objs:
      if self.should_sell(ctx, obj):
        ctx.sell(obj)

    # 3. Micro layer: Daily entry (automatically guarded by active whitelist)
    for obj in ctx.universe:
      if not ctx.account.has_position(obj):
        if self.should_buy(ctx, obj):
          ctx.buy(obj, percent=0.2)  # Hard cash budget + Soft dynamic slots
```

---

## 3. Trade & Rebalance History (`PortfolioRecord`)

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

## 4. Analysis & Summary Reporting

### Markdown Summary Report (`## Rebalance Records`)

When running with CLI `-S/--summary` or calling `build_markdown_summary(reports)` on a portfolio strategy, a dedicated **`## Rebalance Records`** section is automatically included, sorting all holding cycles by entry date:

```markdown
## Rebalance Records

| obj | entry | exit | hold profit |
| :--- | :--- | :--- | ---: |
| SH600000 | 2020-01-06 | 2020-02-10 | +12.34% (+12,345.00) |
| SZ000001 | 2020-02-10 | Holding | -2.50% (-2,500.00) |
```

- **Entry Time**: Position opening/portfolio inclusion date (if you continue adding to the position during the holding period, use the first purchase time).
- **Exit Time**: Date of full liquidation/removal; if still held at the end of the backtest, display as `Holding`.
- **Holding P&L**: Return rate and realized/unrealized P&L amount `+12.34% (+12,345.00)`, net of buy/sell commissions on both sides and slippage.
- **Sorting**: Strictly ascending by **Entry Time**.

### Inspecting Trades as a DataFrame

```python
acc = PortfolioAccount(1000000.0, dp)
runner = PortfolioRunner(acc)
runner.run(strategy)

# Convert trade history to Polars DataFrame
trades = acc.trades_df()
print(trades)
```

---

Apart from the different base class and calling `rebalance` at the appropriate time, everything else can be referenced from [How to build a trading strategy](./trading.md).
