# Financial Fields Schema (MSD stock_financial)

MSD database `stock_financial` contains over 160+ quarterly financial fields (`f000` to `f242`), covering Balance Sheets, Income Statements, Cash Flow Statements, and financial capability ratios (solvency, efficiency, profitability, growth).

> [!IMPORTANT]
> **Single Source of Truth (SSOT)**:
> Financial fields are numbered code-names. **Never guess `f***` field numbers from memory or hallucination.**
> Always run the introspection command below to verify the exact column name, Chinese description, and calculation formula from the running MSD instance.

---

## 1. Introspection Commands

Ensure your MSD server is running (default: `http://127.0.0.1:50510` or as specified by `MSD_HOST` / `MSD_SERVER_URL`).

### View Full Schema
```bash
# Using msd shell with server URL
msd shell -s "${MSD_HOST:-http://127.0.0.1:50510}" ".schema stock_financial"
```

### Search Specific Metrics with grep
```bash
MSD_URL="${MSD_HOST:-http://127.0.0.1:50510}"

# Search per-share indicators (每股净资产, 每股收益, etc.)
msd shell -s "$MSD_URL" ".schema stock_financial" | grep "每股"

# Search valuation / balance sheet equity (净资产, 股东权益)
msd shell -s "$MSD_URL" ".schema stock_financial" | grep "净资产"

# Search profitability (ROE, ROIC, 毛利率, 净利润)
msd shell -s "$MSD_URL" ".schema stock_financial" | grep "盈利能力"

# Search growth / YoY change (营业总收入增长率, 净利润增长率)
msd shell -s "$MSD_URL" ".schema stock_financial" | grep "增长率"

# Search cash flow metrics (经营现金流, 现金净流量)
msd shell -s "$MSD_URL" ".schema stock_financial" | grep "现金流量"

# Search solvency & leverage (资产负债率, 流动比率)
msd shell -s "$MSD_URL" ".schema stock_financial" | grep "资产负债率"
```

---

## 2. Common Verification Examples

When writing strategy factors, verify the field first:

- **Book Value Per Share (BVPS)**:
  `msd shell -s "$MSD_URL" ".schema stock_financial" | grep "每股净资产"`
  $\to$ **`f003`** (`每股指标 >> 每股净资产`) — *Notice: there is NO `f004`!*
- **Diluted EPS**:
  `msd shell -s "$MSD_URL" ".schema stock_financial" | grep "摊薄每股收益"`
  $\to$ **`f000`** (`每股指标 >> 摊薄每股收益`)
- **Annualized ROE (%)**:
  `msd shell -s "$MSD_URL" ".schema stock_financial" | grep "年化净资产收益率"`
  $\to$ **`f205`** (`盈利能力分析 >> 年化净资产收益率`)
- **Revenue YoY Growth (%)**:
  `msd shell -s "$MSD_URL" ".schema stock_financial" | grep "主营业务收入增长率"`
  $\to$ **`f194`** (`发展能力分析 >> 主营业务收入增长率`)
- **Net Profit YoY Growth (%)**:
  `msd shell -s "$MSD_URL" ".schema stock_financial" | grep "净利润增长率"`
  $\to$ **`f197`** (`发展能力分析 >> 净利润增长率`)
- **Debt-to-Asset Ratio (%)**:
  `msd shell -s "$MSD_URL" ".schema stock_financial" | grep "资产负债率"`
  $\to$ **`f223`** (`股本结构分析 >> 资产负债率`)

---

## 3. Best Practices in Strategies

1. **Memory Guard**: MSD `stock_financial` contains hundreds of columns. Always specify needed fields on your `Strategy` class:
   ```python
   class MyStrategy(mbt.Strategy):
     fin_fields = ["f000", "f003", "f205"]
   ```
2. **Annual vs. Quarterly**: Set `fin_only_year = True` if your strategy only requires full-year audited reports (`ts.month == 12`).
3. **No Lookahead**: The framework automatically aligns quarterly reports strictly to statutory disclosure deadlines (`04-30`, `08-31`, `10-31`). Do NOT manually lag report dates.
