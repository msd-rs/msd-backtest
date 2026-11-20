# How to Build and Evaluate Alpha Factors (`mbt.factor`)

`mbt.factor` provides a fast, vectorized, Agent-First factor research engine sharing the exact same `Panel`, price-adjustment semantics, statutory financial disclosure alignment, and `py-alpha-lib` operators as `mbt.select` and `mbt.run`.

Use `mbt.factor` to validate predictive power (`RankIC`, `RankICIR`, quantile monotonicity, long-only excess return, turnover/half-life, OOS holdout retention, industry/market-cap neutrality) **before** building a full trading or portfolio strategy.

---

## 1. Two Ways to Define Factors

### Option A: Inline / Batch Expressions (`alpha.lang` — Fastest Iteration)

Ideal for price-volume formulas, fundamental ratios, WorldQuant 101 / GTJA 191 / WorldQuant BRAIN formulas, and rapid batch screening.

- **Canonical syntax**: WorldQuant BRAIN naming (`ts_*` for rolling, unprefixed for cross-sectional, `group_*` for industry grouped).
- **Compatible aliases**: Full compatibility with WQ101 (`delay`, `delta`, `stddev`, `correlation`, `decay_linear`, `indneutralize`, `adv20`) and GTJA191 (`tsmax`, `tsmin`, `tsrank`, `decaylinear`, `std`, `corr`, `sma(x,n,m)`, `SELF` recursion).
- **Auto-inferred dependencies**: Financial fields (`f001`..`fNNN`), industry classification (`indclass.industry`), and rolling `warmup_bars` are automatically inferred from the expression AST.

Inspect all supported operators and fields in one command:
```bash
uv run python -m mbt.factor --list-ops
```

Evaluate single or multiple expressions directly from CLI (`-E` is repeatable):
```bash
uv run python -m mbt.factor \
  -E 'bp=f003 / raw_close' \
  -E 'wq7=((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1)' \
  -s A_STOCKS_EXCLUDE_ST -b 2022-01-01 -e 2025-12-31 \
  --winsorize mad --neutralize industry,mktcap --holdout 2025-01-01 \
  -S --overview factor_overview.png
```

Or evaluate a batch expression pool file (`.txt` with `name: expr` lines, or `.json` dict/list):
```bash
uv run python -m mbt.factor factors/pool.txt -s A_STOCKS_EXCLUDE_ST -b 2022-01-01 -e 2025-12-31 -S
```

### Option B: Python `Factor` Subclass (Full Flexibility)

Ideal for custom logic, multi-factor families, or complex numpy/alpha-lib pipelines. See [strategy/demo_factor.py](../../../../strategy/demo_factor.py).

```python
import alpha as al
import numpy as np
from mbt.factor import Factor
from mbt.select import Panel

M_AUTHOR = "<Your Model Name>"


class MyMultiFactor(Factor):
  """Reference multi-factor family."""

  fin_fields: list[str] | None = ["f003"]
  fin_only_year: bool = False
  warmup_bars: int = 60
  direction: int | dict[str, int] = 1  # 1: higher is better, -1: lower is better

  def __init__(self, mom_window: int = 60):
    self.mom_window = int(mom_window)

  def compute(self, panel: Panel) -> dict[str, np.ndarray]:
    # Price-volume momentum uses forward-adjusted `close`
    close = np.asarray(panel.column("close"), dtype=np.float64)
    ref_c = panel.compute(al.REF, "close", self.mom_window)
    with np.errstate(divide="ignore", invalid="ignore"):
      mom = np.where(ref_c > 0.0, (close - ref_c) / ref_c, np.nan)

      # Valuation multiples MUST use unadjusted `raw_close`
      raw_close = np.asarray(panel.column("raw_close"), dtype=np.float64)
      bvps = np.asarray(panel.column("f003"), dtype=np.float64)
      bp = np.where((raw_close > 0.0) & (bvps > 0.0), bvps / raw_close, np.nan)

    return {"mom": mom, "value_bp": bp}
```

Run with CLI (supports `-a key=value` with Fail-Fast validation):
```bash
uv run python -m mbt.factor strategy/demo_factor.py \
  -s A_STOCKS_EXCLUDE_ST -b 2022-01-01 -e 2025-12-31 \
  -a mom_window=60 \
  --winsorize mad --neutralize industry,mktcap --holdout 2025-01-01 \
  -S --overview factor_overview.png -o factor_report.json
```

---

## 2. Hard Invariants & Guardrails (Fail-Fast)

1. **Price-Basis Isolation**:
   - Forward-adjusted (`open`, `high`, `low`, `close`, `vwap`) -> price-volume indicators & forward total returns.
   - Unadjusted (`raw_close`, `raw_open`, ...) -> valuation multiples (`f003 / raw_close`) & market cap (`cap = raw_close * total_shares`).
   - Dividing per-share financials (`f001`–`f006`) by forward-adjusted `close` is automatically rejected by AST validation.
2. **Strict Causality (Zero Lookahead)**:
   - Lookahead operators (`fret`, `backset`) and negative lags (`ts_delay(close, -1)`) are rejected statically.
   - A dynamic **causality probe** runs by default: it truncates the panel at historical probe bars, re-runs `compute()`, and fails fast if any value changes when future bars are removed.
   - Use `^` (not `**`) for exponentiation in expressions.
3. **A-Share Tradability & Non-Inflating Quantile Returns**:
   - Default `--entry next_open` aligns signal at bar `t` close with entry at bar `t+1` open (matching `PortfolioStrategy(exec_mode="next_open")`) and excludes suspended & limit-up (`limited == 1`) entries.
   - Quantile NAV curves use overlapping equal-weight **tranches** (`1/p` capital rolled daily) so multi-day periods (`5d`, `10d`, `20d`) never suffer overlapping compounding inflation.

---

## 3. Key CLI Options

| Flag | Default | Description |
|---|---|---|
| `-s, --objs` | (required) | Candidate universe (`A_STOCKS_EXCLUDE_ST`, `A_STOCKS`, stock list file, or codes) |
| `-B, --benchmark` | `SH000300` | Calendar alignment baseline & excess return benchmark (excluded from cross-section) |
| `-b, --begin` / `-e, --end` | `2022-01-01` / `2025-12-31` | Evaluation window |
| `--periods` | `1,5,10,20` | Forward return holding horizons (bars) |
| `--primary-period` | `5` | Primary horizon for quantile, annual, holdout, and overview panels |
| `--quantiles` | `5` | Number of cross-sectional quantile groups (e.g. 5 or 10) |
| `--entry` | `next_open` | Alignment mode: `next_open`, `open_close` (`al.FRET`), or `close` |
| `--winsorize` | `none` | Outlier clipping: `none`, `mad`, `sigma`, `percentile` |
| `--neutralize` | `none` | Neutralization: `none`, `industry`, `mktcap`, or `industry,mktcap` |
| `--standardize` | `none` | Cross-sectional scaling: `none`, `zscore`, `rank` |
| `--cost-bps` | `0.0` | One-way transaction cost in bps deducted from tranche portfolio returns |
| `--holdout` | `None` | IS/OOS split date or trailing duration (e.g. `2025-01-01`, `1y`, `180d`) |
| `--by-industry` | `False` | Add Shenwan L1 per-industry RankIC table |
| `-S, --summary` | `False` | Print Markdown summary report to stdout |
| `-o, --output` | `""` | Export report to `.json`, `.md`, `.csv`, `.xlsx`, or `.png` |
| `--overview [path]` | `None` | Render Chinese high-DPI 4-panel tear-sheet PNG (`factor_overview.png`) |
| `--list-ops` | `False` | Print canonical expression operators, aliases, and data fields |
