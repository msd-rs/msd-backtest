# `msd-backtest (mbt.factor)` 与 `Microsoft Qlib` 因子构建与评估体系对比

本文基于 `msd-backtest` 仓库源码（[mbt/factor/](file:///home/jia/repo/msd-backtest/mbt/factor/__init__.py) 与 [factor.md](file:///home/jia/repo/msd-backtest/.agents/skills/msd-strategy-guide/references/factor.md)）及 Microsoft Qlib 官方文档/源码架构，从**设计哲学、数据底座、因子构建、防偏机制、截面预处理、评估体系、报告与工作流**七个维度进行系统性对比。

---

## 1. 核心定位与设计哲学对比

| 维度 | `msd-backtest` (`mbt.factor`) | `Microsoft Qlib` |
|---|---|---|
| **核心定位** | **Agent-First（AI Agent 优先）的确定性因子投研与防错引擎** | **AI/ML 优先的端到端量化研究平台** |
| **核心工作流** | 面向单因子/批量表达式的**极小验证闭环**：定义因子 → 自动依赖推断 → 因果与复权校验 → 截面评估 → 进入 `mbt.select` / `mbt.run` 策略 | 面向**机器学习全流程（`qrun`）**：原始/Alpha158 特征工程 → `DatasetH` 切分 → 模型训练（LightGBM/PyTorch）→ 信号预测 → `PortAnaRecord` 组合回测 |
| **错误处理哲学** | **Fail-Fast（零容忍静默降级）**：错误参数、未来函数、前复权混算基本面、样本不足等直接抛异常阻断 | **配置驱动与科研灵活性**：容忍宽泛的表达式与 `Processor` 组合，依赖研究者自身配置正确性（如自行区分 `infer` 与 `learn` 处理器） |
| **典型交互入口** | CLI 一行命令（`-E` 内联表达式、`.txt/.json` 批量文件、`.py` 子类）或 Python API (`evaluate_factor`) | YAML 配置文件（`qrun`）+ Python SDK (`D.features` / `DataHandlerLP` / `Recorder`) + Jupyter Notebook |

> [!NOTE]
> 两者的根本差异在于**抽象重心**：Qlib 围绕 **「高维特征矩阵 → ML 模型预测评分 (`score`) → 组合优化回测」** 构建；`mbt.factor` 则围绕 **「可解释因子本身 → 严格 A 股语义隔离与因果防线 → 统计与分层交易验证」** 构建，并与同仓的选股器 (`mbt.select`) 和撮合回测 (`mbt.run`) 共享同一套 `Panel` 矩阵与复权口径。

---

## 2. 数据底座与内存模型对比

| 维度 | `msd-backtest` (`mbt.factor`) | `Microsoft Qlib` |
|---|---|---|
| **存储与数据源** | 默认连接 MSD HTTP 数据库服务（`http://localhost:50510`），通过 Polars/NumPy 加载行情、股本、申万行业与财报 | 本地列式二进制存储（`.bin` 按股票-字段独立文件 + 日历/股票池索引），支持 `QlibDataLoader` 与 `StaticDataLoader` |
| **核心数据结构** | [Panel](file:///home/jia/repo/msd-backtest/mbt/select/panel.py#L16-L35)：严格矩形 `(n_objs, bars)` 2D 连续 NumPy 缓冲区（块布局 `i*bars .. (i+1)*bars`），携带 `objs` 元数据并支持零拷贝视图与 Polars 惰性物化 | `pandas.DataFrame`（通常为 `(datetime, instrument)` 或 `(instrument, datetime)` 双层 MultiIndex），配合 `DiskExpressionCache` / `DiskDatasetCache` 缓存表达式结果 |
| **复权价格体系** | **显式双轨制**（[PanelExecContext](file:///home/jia/repo/msd-backtest/mbt/factor/expr.py#L67-L135)）：<br>• 前复权：`open` / `high` / `low` / `close` / `vwap`<br>• 未复权：`raw_open` / `raw_high` / `raw_low` / `raw_close` (`price`) / `raw_vwap` | 通常默认存储**后复权或前复权价格 + `$factor` 复权因子**（`$close` 为复权价，真实未复权价格需由 `$close / $factor` 等方式换算） |
| **基准与日历对齐** | 强制以 `--benchmark`（默认 `SH000300`）作为交易日历对齐基准，在加载后将基准从截面中剥离（防止污染 IC/分位数），同时透传 `benchmark_close` / `benchmark_open` 给因子与超额基准 | 通过 `instruments`（如 `csi300`）与 `calendar` 对齐时间轴；跨标的引用基准需在表达式中显式使用 `ChangeInstrument("SH000300", ...)` |
| **Warmup 预热管理** | 表达式 AST **自动推断**滚动窗口所需的 `warmup_bars`（并在评估前自动加载、评估指标时自动剔除），支持 `--warmup-bars` 覆盖 | `QlibDataLoader` 按表达式依赖自动向前扩窗读取 `.bin` 历史片段，再裁剪到 `start_time ~ end_time` |

---

## 3. 因子构建方式与表达式引擎对比

### 3.1 因子定义方式

- **`mbt.factor`（双路径）**：
  1. **表达式路径（`ExpressionFactor`）**：支持 CLI `-E 'name=expr'` 多表达式并行、`.txt`（`name: expr` / `name = expr`）或 `.json` 批处理池文件，详见 [expr.py](file:///home/jia/repo/msd-backtest/mbt/factor/expr.py#L705-L811)。
  2. **Python 子类路径（`Factor`）**：继承 [Factor](file:///home/jia/repo/msd-backtest/mbt/factor/factor.py#L45-L113)，实现 `compute(self, panel: Panel) -> np.ndarray | dict[str, np.ndarray]`，支持一次返回多因子族字典、自定义 `direction`、声明 `fin_fields` 与 `finalize()` 诊断钩子，并通过 `-a key=value` 传入构造参数（未声明参数会 Fail-Fast 报错）。
- **`Qlib`（配置与 Handler 路径）**：
  1. **字符串表达式列表**：在 `D.features(instruments, fields)` 或 `QlibDataLoader(config={"feature": [...], "label": [...]})` 中编写表达式字符串列表。
  2. **内置高维特征模板（`DataHandlerLP` 子类）**：
     - **`Alpha158`**：预置 158 个价量 K 线形态、多窗口滚动统计（5/10/20/30/60 日均值、波动、回归斜率/R²/残差、量价相关性等）特征，常作为 GBDT/MLP 模型输入。
     - **`Alpha360`**：展开过去 60 个交易日的归一化 OHLCV 原始序列（共 360 维），常用于 LSTM/Transformer 等时序深度模型。
  3. **自定义算子扩展**：继承 `qlib.data.ops.ExpressionOps`（如 `ElemOperator`、`Rolling`、`PairRolling`）注册到表达式解析器。

### 3.2 表达式引擎与算子方言

| 特性 | `mbt.factor` (`py-alpha-lib` + `PanelExecContext`) | `Qlib` (`qlib.data.ops`) |
|---|---|---|
| **执行后端** | `alpha.lang.to_python` 将表达式编译为 Python/NumPy 调用，底层时序与截面核心算子由 **Rust (`py-alpha-lib`)** 加速执行，零拷贝绑定 `Panel` 缓冲区 | 表达式 AST 递归求值，核心 Rolling 算子由 **Cython** 实现（如 `rolling_slope`、`rolling_rsquare`、`rolling_resi`），支持多进程与磁盘缓存 |
| **语法与方言** | **多方言原生兼容**（[format_operator_catalog](file:///home/jia/repo/msd-backtest/mbt/factor/expr.py#L814-L846)）：<br>• **WorldQuant BRAIN 规范**：`ts_mean`, `ts_rank`, `group_neutralize`, `group_rank`, `rank`, `zscore`<br>• **WQ101 兼容**：`delay`, `delta`, `stddev`, `correlation`, `decay_linear`, `indneutralize`, `adv20`<br>• **GTJA191 兼容**：`tsmax`, `tsmin`, `tsrank`, `sma(x,n,m)`, `regbeta`, `regresi`, `SELF` 递归、三元运算 `? :` | **Qlib 自研表达式语法**：<br>• 字段带 `$` 前缀（如 `$close`, `$volume`），PIT 财务字段带 `$$` 前缀<br>• 算子首字母大写驼峰：`Ref`, `Mean`, `Std`, `Slope`, `Rsquare`, `Resi`, `IdxMax`, `Quantile`, `Corr`, `Rank`<br>• 跨标的上下文算子：`Mask`, `ChangeInstrument` |
| **自动依赖推断** | 编译 AST 时自动提取所需的 `fin_fields`（`f001..fNNN`）、`needs_industry`（`indclass.industry`）、`needs_cap` 以及最大 `suggested_warmup` | 表达式由 `QlibDataLoader` 显式列出，所需字段与窗口长度在解析表达式树时由 loader 获取 |

### 3.3 基本面 PIT 对齐与防未来函数机制（关键差异）

1. **财报 Point-In-Time (PIT) 对齐**：
   - **`mbt.factor`**：底层统一按**法定披露截止日**（04-30 / 08-31 / 10-31）做 backward asof 对齐，支持 `fin_only_year` 与 `fin_lookback_days`，无需策略作者手写 PIT 逻辑。
   - **`Qlib`**：提供独立的 `qlib.data.pit` 模块（`P()`、`PRef()` 算子与 `$$field_q` 语法），基于公告日 `date` 与报告期 `period` 构建双索引二进制存储；需要用户单独准备并转换 PIT 格式财务数据库。
2. **价格复权语义隔离（`mbt` 独有强校验）**：
   - **`mbt.factor`**：静态 AST 检查（[compile_expression](file:///home/jia/repo/msd-backtest/mbt/factor/expr.py#L563-L574) 与 [_FactorAntiPatternVisitor](file:///home/jia/repo/msd-backtest/mbt/factor/factor.py#L148-L165)）直接禁止将每股财务指标（`f001`–`f006`）与前复权价格（`close`/`open`/`high`/`low`）相除，强制要求使用 `raw_close`，从源头杜绝送转股导致的估值扭曲。
   - **`Qlib`**：框架层不做跨语义的 AST 报错拦截，若将复权 `$close` 与未复权财务数据直接混算，不会抛出异常。
3. **因果性与前视偏差防线**：
   - **`mbt.factor`（双重防线）**：
     - **静态拦截**：禁止 `FRET`、`BACKSET` 及任何负滞后参数（如 `ts_delay(close, -1)`、`al.REF(..., -1)`），并修复了 `PanelExecContext.SCALE` 为严格按单期截面归一化（避免跨全时间轴泄漏）。
     - **动态因果探针**：[check_factor_causality](file:///home/jia/repo/msd-backtest/mbt/factor/factor.py#L351-L474) 默认开启。在多个历史探测点截断未来 K 线（`panel.slice(0, t + 1)`）重新运行 `compute()`，若当前截面值发生任何变化立即 Fail-Fast 报错终止。
   - **`Qlib`**：
     - 表达式引擎本身允许 `Ref($close, -2)`（专门用于构造 `label` 未来收益），但在 PIT 算子中禁止对 `$$` 字段使用负偏移（如 `Ref($$roewa_q, -1)`）。
     - 没有自动化的截断因果探针，防泄漏主要靠 `DataHandlerLP` 严格切分 `fit_start_time` / `fit_end_time` 以及区分 `infer_processors` 与 `learn_processors`（如 `DropnaLabel.is_for_infer() == False`）。

---

## 4. 截面预处理管线对比

| 处理步骤 | `mbt.factor` ([preprocess.py](file:///home/jia/repo/msd-backtest/mbt/factor/preprocess.py)) | `Qlib` (`DataHandlerLP` `Processor` 链) |
|---|---|---|
| **执行顺序与方式** | 确定性固定管线：`mask -> winsorize -> neutralize -> standardize` | 可自由编排的 `Processor` 列表，分为 `infer_processors`（特征）与 `learn_processors`（标签，支持 `fit` / `transform` 持久化训练集统计量） |
| **去极值 (Winsorization)** | 支持 `mad`（中位数 ± $k \times 1.4826 \times \text{MAD}$，含零 MAD 防坍塌保护）、`sigma`（均值 ± $k\sigma$）、`percentile`（分位截尾） | 主要通过 `RobustZScoreNorm(..., clip_outlier=True)`（同样基于 Median + `1.4826 * MAD` 并默认裁剪到 `[-3, 3]`）或 `ProcessInf` 完成 |
| **中性化 (Neutralization)** | **一等公民内置支持**：`industry`（申万行业均值去中心化）、`mktcap`（对 $\ln(\text{raw\_close} \times \text{total\_shares})$ 做截面 OLS 残差）、以及 **`industry,mktcap` 基于 Frisch-Waugh-Lovell (FWL) 定理的精确联合正交投影** | **原生 `Processor` 链未内置行业+市值 FWL 正交中性化**；虽然表达式层有基础截面算子或需自行扩展 `Processor` / 依赖下游组合优化器控制风格暴露 |
| **标准化与缺失填充** | `zscore`（Rust `al.CC_ZSCORE`）、`rank`（Rust `al.CC_RANK`）；保留缺失掩码，**不**做均值填充以免引入伪信号 | `CSZScoreNorm`（截面 Z-Score）、`CSRankNorm`（截面排序）、`ZScoreNorm`（时序 fit 全局均值方差）、`MinMaxNorm`、`CSZFillna`（截面均值填充 `NaN`）、`DropnaLabel` |

---

## 5. 因子评估体系与统计方法对比

### 5.1 前瞻收益与 A 股可交易性建模

- **`mbt.factor`** ([returns.py](file:///home/jia/repo/msd-backtest/mbt/factor/returns.py))：
  - **三种交易对齐模式（`--entry`）**：
    - `next_open`（默认）：$t$ 日收盘产生信号 $\rightarrow t+1$ 日开盘买入 $\rightarrow t+1+p$ 日开盘卖出，与实盘引擎 `PortfolioStrategy(exec_mode="next_open")` 口径完全一致。
    - `open_close`：使用 `al.FRET` 计算 $t+1$ 开盘至 $t+p$ 收盘收益。
    - `close`：$t$ 日收盘买入的理论无延迟参考。
  - **内建 A 股涨停与停牌过滤**：信号端自动剔除停牌（`volume <= 0` 或价格无效）及次新股（`--min-listed-days`）；买入端默认排除涨停板（`limited == 1`），并在报告中统计 `suspended_ratio` 与 `limit_filtered_ratio`。
- **`Qlib`**：
  - 前瞻收益完全由用户在 `label` 中自定义表达式（常见模板为 `["Ref($close, -2) / Ref($close, -1) - 1"]`，即 $T+1$ 收盘买至 $T+2$ 收盘卖，或用 `$open` 替代）。
  - 信号分析阶段（`SigAnaRecord` / `calc_ic`）**不包含**涨跌停/停牌买不进的可交易性掩码；A 股涨跌停与成交量约束是在下游 `PortAnaRecord` 驱动 `Simulator`（`Exchange` 对象）做完整账户撮合回测时才生效。

### 5.2 分层测试、换手与统计指标对比

| 评估维度 | `mbt.factor` ([metrics.py](file:///home/jia/repo/msd-backtest/mbt/factor/metrics.py) & [evaluator.py](file:///home/jia/repo/msd-backtest/mbt/factor/evaluator.py)) | `Qlib` (`calc_ic` / `SigAnaRecord` / `analysis_model` / `PortAnaRecord`) |
|---|---|---|
| **IC / RankIC 检验** | 逐日截面 Pearson IC 与 Spearman RankIC（Rust `al.CC_RANK` 并列平均秩）；输出均值、标准差、ICIR、年化 RankICIR、胜率、显著占比（`\|RankIC\| > 0.02`）、普通 $t$ 值及 **Newey-West HAC 异方差自相关稳健 $t$ 值**（自动按持有期 `lag = period - 1` 调整重叠自相关） | `calc_ic(pred, label)` 计算逐日 Pearson IC 与 Rank IC；`SigAnaRecord` 记录 `IC`, `ICIR`, `Rank IC`, `Rank ICIR`；`model_performance_graph` 展示逐日 IC 序列与分布 |
| **分层组合回测** | 默认多周期（`1,5,10,20`）分层测试，采用 **重叠等权 Tranche 滚动模拟**（每日将 $1/p$ 资金滚入最新分位持仓 $p$ 天）：<br>• 消除多日持有期简单复利的虚高膨胀<br>• 消除调仓起始日偏差<br>• 支持 `--cost-bps` 按日换手扣除单边交易成本<br>• 输出 Q1..QN、多空（`QN-Q1`）、多头超额市场（`QN-Mkt`）、多头超额基准（`QN-Bench`）及分位单调性（Spearman） | 分为两层：<br>1. **轻量信号层**：`calc_long_short_return` 与 `analysis_model.model_performance_graph(graph_names=['group_return', ...])` 计算分位平均收益与多空收益。<br>2. **完整回测层**：`PortAnaRecord` 配合 `TopkDropoutStrategy` / `WeightStrategyBase` 运行真实资金撮合回测（`report_graph` + `risk_analysis`） |
| **换手与衰减** | 计算 Top/Bottom 分位日均单边换手率、多滞后阶 `(1,5,10,20)` 秩自相关性，并推算 **因子半衰期（`half_life_days`）** | `model_performance_graph` 中的 `pred_autocorr` 计算预测分数的自相关曲线（需指定 `lag`）；真实换手率由 `PortAnaRecord` 回测报表给出 |
| **结构化切片诊断** | 一体化内置：<br>• **分年度拆解**（每年 RankIC、多头超额）<br>• **IS/OOS Holdout 留存检验**（`--holdout`，计算 OOS/IS RankIC 留存率）<br>• **申万一级行业拆解**（`--by-industry`）<br>• **多因子 Rank 相关性矩阵** | 通常通过 `DatasetH` 的 `segments`（`train`/`valid`/`test`）分别运行并由 MLflow 实验管理对比，不自带一键分行业 RankIC 拆解与多因子共线性报告 |
| **自动化异常预警** | [_build_warnings](file:///home/jia/repo/msd-backtest/mbt/factor/evaluator.py#L97-L145) 自动输出 7 类诊断警告（如反向预测提示改 `direction`、对数市值暴露 `\|r\| > 0.30` 未中性化、覆盖率 `< 50%`、零方差截面天数过多、单调性弱、换手过高、OOS 严重衰减） | 无内建结构化启发式告警规则，依赖研究者查看图表与指标 |

---

## 6. 报告导出、可视化与生态工具对比

| 维度 | `msd-backtest` (`mbt.factor`) | `Microsoft Qlib` |
|---|---|---|
| **报告与导出格式** | [export_report](file:///home/jia/repo/msd-backtest/mbt/factor/report.py#L413-L476) 一键支持：<br>• **终端/文档**：Markdown 精简诊断摘要（`-S` / `.md`）<br>• **机器可读**：完整结构化 `.json`（含 `summary`、`diagnostics`、`warnings` 与日频序列）<br>• **表格分析**：`.csv` 与多 Sheet `.xlsx`（`leaderboard`, `correlation`, `ic_summary`, `quantiles`, `daily_series`, `annual`, `industry`） | 基于 **MLflow `Recorder`** 记录实验参数、指标与 pickle/csv 产物；支持在 Python/Jupyter 中加载 artifact 进一步分析 |
| **可视化图表** | `--overview`（[overview.py](file:///home/jia/repo/msd-backtest/mbt/factor/overview.py)）生成静态 **中文高 DPI 4 联图 Tear-sheet (`factor_overview.png`)**，包含顶部 KPI 横幅与多面板收益/IC/分位对比 | `qlib.contrib.report` 基于 **Plotly** 生成交互式 HTML/Notebook 图表：<br>• `analysis_model.model_performance_graph`（分位累计收益、预测 IC、自相关）<br>• `analysis_position.report_graph` / `score_ic_graph` / `cumulative_return_graph` |
| **辅助运维与文档** | `--list-ops`（实时打印规范算子/方言/字段字典）、`--doctor`（检查 MSD 连接与数据覆盖） | 官方 Sphinx 文档（ReadTheDocs）、`scripts/check_data_health.py`、以及配套生态 **RD-Agent**（自动假设生成与因子挖掘） |

---

## 7. 总结与适用场景建议

1. **选择 `mbt.factor` 的典型场景**：
   - 专注于 **A 股单因子挖掘、Alpha101/191 公式验证、基本面估值/质量因子检验**，需要极速从想法验证到策略落地（验证通过的 `Factor` 可无缝复用同一套 `Panel` 与 `alpha.lang` 算子进入 `mbt.select` / `mbt.run` 组合回测）。
   - **AI Agent 自动化投研闭环**：需要依靠 AST 静态拦截、动态截断因果探针、复权双轨校验、结构化 JSON/Markdown 诊断警告来防止 AI 幻觉与隐性前视偏差。
   - 需要开箱即用的 **A 股行业+市值 FWL 联合中性化**、**涨停/停牌过滤**、**Newey-West t 检验**与 **Tranche 非膨胀分层净值**。
2. **选择 `Qlib` 的典型场景**：
   - 需要 **大规模非线性机器学习/深度学习模型训练**（GBDT、MLP、LSTM、Transformer、图神经网络），利用 `Alpha158` / `Alpha360` 构造百维级特征工程并通过 `DatasetH` 管理训练-验证-测试流水线与 MLflow 实验追踪。
   - 需要基于预测分数结合凸优化/组合风控（`TopkDropoutStrategy`、`EnhancedIndexingStrategy`）进行中高频或跨市场（美股/A 股）端到端 AI 选股研究。
3. **协同参考方向**：
   - 若未来在 `mbt` 中引入多因子机器学习合成能力，可借鉴 Qlib 的 `DataHandlerLP`（将 `winsorize`/`neutralize`/`standardize` 的跨样本统计参数持久化分离 `fit` 与 `infer`）；反之，在 Qlib 中做传统单因子研究时，也可借鉴 `mbt.factor` 的 **动态截断因果探针**、**FWL 联合中性化** 与 **Tranche 分层收益** 设计来补齐信号验证阶段的严谨性。
