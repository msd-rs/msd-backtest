# `benchmarks` 策略性能测试报告

> 注：本报告汇总 `msd-backtest` 基准测试套件在当前工作区环境下的实测性能数据。所有用例在同一台主机、同一 `.venv` / MSD 环境下各串行运行 3 次，时钟墙与各阶段耗时均取中位数。

## 1. 测试环境

| 项目 | 值 |
| :--- | :--- |
| 运行目录 | `.`（通过 `uv run` 调用本地源码树 `mbt` 与 `alpha-lib-rs` release 扩展） |
| Python | 3.14.4（`uv run`） |
| msd-backtest / py-alpha-lib | 0.2.1 / 0.5.1 |
| pymsd / polars / numpy | 0.1.27 / 1.42.1 / 2.5.1 |
| MSD 服务 | `http://127.0.0.1:50511` |
| CPU | 13th Gen Intel(R) Core(TM) i7-13700K |
| 计时方式 | 端到端进程时钟墙（含 `uv`/解释器启动、导入、取数、计算、报告）；分阶段耗时取 `mbt` 日志时间戳 |
| 重复次数 | 每项 3 次串行运行，取中位数 |

## 2. 统一测试参数

- 初始池：`A_STOCKS`，全市场 5226 只股票（回测加载含基准指数共 5227 个 obj）。
- 回测区间：`2020-01-01` ~ `2025-12-31`，共 1536 根日 K 线，预热 81 根（策略类声明 `warmup_bars = 60`）。
- 回测数据规模：5227 × 1536 = **8,028,672** 个 bar。
- 撮合模式：`next_open`；基准：`SH000300`（`.env` 默认）；初始资金：10,000,000；费率：默认配置。
- 选股器选股日：`2025-12-31`；所有用例初始池均为全市场 `A_STOCKS`。

## 3. 结果汇总

> 说明：回测类与资金账户类的「计算时间」= `init`（账户与运行器初始化）+ `strategy`（策略执行与撮合）；选股类的取数嵌入各 `step` 中（仅加载 K 线面板时会单独记录 `raw_data_build`），未单独生成 `Report`。

| 类别 | 用例 | 数据规模 | 时钟墙 | 取数时间 | 计算时间 | 报告时间 | 备注 |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 选股类 | `selector_price_volume.py` | 5226 → 600 → 20（600×120） | **0.335s** | 0.007s *(K线)* | 0.025s *(选股合计 0.032s)* | — | 纯价量；快照初筛后加载 600 只 K 线计算均线与量比 |
| 选股类 | `selector_fundamental.py` | 5226 → 828 → 20 | **0.363s** | *(含于步骤内)* | 0.029s *(选股合计)* | — | 财报法定披露日对齐 + 未复权 PB 估值；不加载 K 线面板 |
| 选股类 | `selector_fullmarket_resample.py` | 5226 → 20（全池 250 bar） | **0.550s** | 0.099s | 0.132s *(step01 合计 0.231s)* | — | 全市场重度选股；跳过初筛加载全池 K 线与 `fin_fields` + `Panel.resample("1W")` |
| 回测类 | `trading_signals_fast.py` | 5227 × 1536 (8.03M bar) | **1.950s** | 0.854s | 0.279s (`0.093 + 0.186`) | 0.443s | 无持仓状态；`signals()` 走 Rust `Simulator` 快路径（EMA 5/20 + MA120） |
| 回测类 | `trading_high_turnover.py` | 5227 × 1536 (8.03M bar) | **2.164s** | 0.834s | 0.309s (`0.095 + 0.214`) | 0.606s | 高换手快路径；EMA(2/5) 高频交叉（中位 130 笔/股），采用惰性 `Record` 回填 |
| 回测类 | `trading_position_aware.py` | 5227 × 1536 (8.03M bar) | **3.491s** | 0.817s | 1.764s (`0.093 + 1.671`) | 0.512s | 推荐范式；逐 bar `execute()` 使用 `holding_shares()` / `holding_rors()` 向量化掩码下单 |
| 回测类 | `trading_position_loop.py` | 5227 × 1536 (8.03M bar) | **15.796s** | 0.823s | 14.073s (`0.092 + 13.981`) | 0.515s | **反例**：与 `trading_position_aware.py` 逻辑相同，但用 `for g in range(n_objs): ctx.position(g)` 逐股循环 |
| 资金账户类 | `portfolio_scheduled_rebalance.py` | 5227 × 1536 (8.03M bar) | **1.411s** | 0.817s | 0.138s (`0.015 + 0.123`) | 0.058s | 轻载组合（Rust `PortfolioEngine`）；月度定时调仓、`top_k=10`、无 `fin_fields` |
| 资金账户类 | `portfolio_dynamic_risk.py` | 5227 × 1536 (8.03M bar) | **2.189s** | 1.044s | 0.719s (`0.016 + 0.703`) | 0.064s | 重载组合（Rust `PortfolioEngine`）；含 `fin_fields` 对齐 + 24 次季度 `run_selector()` + 每日持仓风控 |

## 4. 工作流、批量扫参及专项验证

| 场景 | 用例 / 参数 | 数据规模 | 时钟墙 | 阶段明细与输出特征 |
| :--- | :--- | :--- | ---: | :--- |
| 全量 JSON 导出 | `benchmarks/trading_signals_fast.py -o /tmp/mbt_bench_fast.json` | 5227 × 1536 (8.03M bar) | **2.832s** | 取数 0.830s + 计算 0.276s (`0.092 + 0.184`) + 报告 0.439s + JSON 导出 **0.924s**；产物 **580.74 MB** |
| 多起点 + 样本外 | `benchmarks/portfolio_scheduled_rebalance.py -b 2020-01-01,2022-01-01,2024-01-01 --holdout 2025-01-01 -S` | 3 起点合计 16.48M bar | **1.771s** | 单次 `load_data` + `_LazySlicedDict` 按需切片，单起点策略中位耗时 **0.157s** |
| 批量参数扫描 | `Runner.run_sweep`（10 组 EMA 参数，复用内存 `DataProvider`） | 10 × 8.03M bar | **2.598s** | 预载数据 0.832s 后，10 组全市场扫参合计 **2.598s**（单组边际 **0.260s**） |
| 官方模板验证 | `mbt/strategy_templates/fundamental_timing.py -s A_STOCKS -p -b 2020-01-01 -e 2025-12-31 -S` | 5227 × 1536 (8.03M bar) | **2.232s** | 全市场基本面季度选股 + 每日向量化止损跟踪 |
| 周线含财报重采样 | `load_kline(fin_fields=["f003","f205"])` 后直接 `panel.resample("1W")` | 5226 × 250 → 54 周 | **0.300s** | 取数对齐 0.201s + 含 `report_ts`（`datetime64[us]`）直接周线聚合 **0.099s** |

## 5. 关键性能结论

1. **组合策略与风控全市场保持在 2 秒级**：基于 Rust `PortfolioEngine` 与 2D NumPy 组合报表计算，月度调仓组合 `portfolio_scheduled_rebalance.py` 端到端仅 **1.411s**（组合报表 **0.058s**）；叠加季度 `ctx.run_selector()`、财报披露日对齐与每日 `holding_rors()` 风控的 `portfolio_dynamic_risk.py` 仅 **2.189s**。
2. **交易策略应优先采用向量化 API 与快速路径**：无持仓信号策略（`trading_signals_fast.py`）和高性能高换手策略（`trading_high_turnover.py`）分别在 **1.950s** 与 **2.164s** 完成 803 万 bar 回测；逐 bar 查询持仓时，`trading_position_aware.py`（**3.491s**）显著优于逐股循环反例 `trading_position_loop.py`（**15.796s**），策略开发应避免 `for g in range(n_objs): ctx.position(g)`。
3. **高密度工作流与扫参成本可控**：多起点组合回测借助单次取数与惰性切片在 **1.771s** 内完成 3 组起点评估；`Runner.run_sweep` 进行 10 组全市场扫参仅 **2.598s**（**0.260s/组**）。
4. **序列化产物注意体积与耗时平衡**：全量 JSON 导出额外耗时 **0.924s** 且输出文件达 **580.74 MB**；Agent 迭代调优阶段建议默认使用 `-S`（Markdown 摘要）。

## 6. 复现命令（在项目根目录 `.` 执行）

```bash
# 选股类
uv run python -m mbt.select.cli benchmarks/selector_price_volume.py \
  -s A_STOCKS -d 2025-12-31 -m http://127.0.0.1:50511 -f
uv run python -m mbt.select.cli benchmarks/selector_fundamental.py \
  -s A_STOCKS -d 2025-12-31 -m http://127.0.0.1:50511 -f
uv run python -m mbt.select.cli benchmarks/selector_fullmarket_resample.py \
  -s A_STOCKS -d 2025-12-31 -m http://127.0.0.1:50511 -f

# 回测类（单标的独立账户）
uv run python -m mbt.run.cli benchmarks/trading_signals_fast.py \
  -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
uv run python -m mbt.run.cli benchmarks/trading_high_turnover.py \
  -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
uv run python -m mbt.run.cli benchmarks/trading_position_aware.py \
  -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
uv run python -m mbt.run.cli benchmarks/trading_position_loop.py \
  -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S

# 资金账户类（共享现金组合）
uv run python -m mbt.run.cli benchmarks/portfolio_scheduled_rebalance.py \
  -s A_STOCKS -p -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
uv run python -m mbt.run.cli benchmarks/portfolio_dynamic_risk.py \
  -s A_STOCKS -p -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S

# 工作流与官方模板验证
uv run python -m mbt.run.cli benchmarks/trading_signals_fast.py \
  -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -o /tmp/mbt_bench_fast.json
uv run python -m mbt.run.cli benchmarks/portfolio_scheduled_rebalance.py \
  -s A_STOCKS -p -b 2020-01-01,2022-01-01,2024-01-01 -e 2025-12-31 --holdout 2025-01-01 -m http://127.0.0.1:50511 -S
uv run python -m mbt.run.cli mbt/strategy_templates/fundamental_timing.py \
  -s A_STOCKS -p -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
```
