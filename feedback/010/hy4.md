# msd stack 是「AI Agent 优先」的量化策略框架

> 基于一次完整实证：在 `/workspace/strategy/011/davis_double_play.py` 中实现「戴维斯双击」选股 + 财报驱动组合调仓，并完成选股、回测、样本外与多起点验证。本文不复述体验，而是给出**判据 → 证据 → 反例 → 限定性结论**的论证。

## 0. 结论先行

msd stack 的设计优先服务于这样一类执行者：**没有跨会话隐性经验、只能依赖文本契约、通过工具调用迭代、且必须自我验证结果**。这恰是 AI Agent 的工作条件。因此"Agent 优先"不是"提供了 API"，而是"把正确性、知识、反馈从人的经验里搬到框架与工具链里"。

限定性结论：

- 在**策略构建与验证闭环**上，msd stack 已是明确的 Agent 优先框架，证据充分且可复现；
- 在**数据语义自描述与契约一致性**上仍存在缺口，这些缺口会让 Agent 依赖读源码与环境探查才能正确完成任务；
- 综合判断：**agent-first 的骨架已经成立，语义层尚未闭合**。用成熟度模型表述，它稳定处于 L2（可自证），部分达到 L3（可自描述）。

---

## 1. 判据：什么才算「Agent 优先」

先把命题变成可证伪的判据。Agent 与人类开发者的差异，直接推导出框架要求：

| Agent 的约束 | 对框架的要求 | 可检验判据 |
| :--- | :--- | :--- |
| 没有跨会话记忆，上下文有限 | 知识外置、按需加载、单一入口 | 是否存在可发现的能力清单与分层参考文档，而非散落的部落知识 |
| 依赖字面文本，会自信地编造标识符 | 契约显式、防幻觉规则 | 字段名/布局/口径是否有运行时可核对的单一事实源，并在文档中写成硬约束 |
| 不能看图表、不能凭盘感 | 反馈结构化、机器可读 | 每一步是否产出可解析的日志/因子表/诊断/报告，足以判断"跑通了没有" |
| 无法凭经验判断"看起来合理" | 正确性是结构性保证而非约定 | 无前视、复权、交易约束是否由框架强制，而不是靠调用者记得 |
| 一次会话内要多轮试错 | 迭代便宜、交付物单一 | 单次验证耗时、是否单文件可复现、是否有内建的稳健性验证开关 |
| 容易被"看似成功"的结果误导 | 结果自证机制 | 能否验证机制按设计运行（而非仅看收益），能否自动暴露过拟合 |

对应的成熟度模型：

- **L1 可执行**：CLI/API 可调用，Agent 能写出能跑的代码；
- **L2 可自证**：结构化反馈、诊断、报告让 Agent 能独立判断对错并迭代；
- **L3 可自描述**：数据覆盖、字段口径、对齐规则、边界行为机器可读，Agent 无需读实现源码或试探环境即可正确完成任务。

---

## 2. 正面论证

### 2.1 知识外置且按意图分派（判据：可发现性）

技能入口 `msd-strategy-guide/SKILL.md` 先给出技术栈与准备检查，再按**策略类型**分派到 `selector.md` / `trading.md` / `portfolio.md`，另有 `financial.md` / `algo.md` / `msd.md` / `optimize.md` 按需加载。本次任务是"横截面选股 + 组合调仓"，Agent 只需读 selector + portfolio + financial 三份文档即可确定架构：`Selector` 负责宽池排序，`PortfolioStrategy` 负责 `rebalance_objs` 调仓，单文件同时定义两者由 `mbt.run.cli` 自动串联。

这是典型的 Agent 优先信息架构：**入口唯一、按意图路由、上下文按最小必要加载**，而不是让 Agent 在一份大手册里检索。

### 2.2 防幻觉规则写进契约（判据：显式契约）

本次最有价值的两条硬约束都来自文档：

- `financial.md` 的 SSOT 规则：`f***` 字段编号**禁止凭记忆猜测**，必须现场执行 `msd shell ".schema stock_financial"` 核对。本次据此确认 `f000`（摊薄 EPS）、`f003`（每股净资产）、`f197`（净利润增长率）、`f205`（年化 ROE），并核对了数值语义（如 `f000` 为累计 EPS、`f197` 为同比百分比）。
- `selector.md` / `trading.md` 的复权警告：估值计算**必须**用未复权 `raw_close`，用前复权 `close` 算 PB 会在高送转股上产生"假低估"。

同时 `Panel` 把布局契约写进 docstring：`objs` 与 `(n_objs, bars)` 视图绑定，"绝不按入参顺序映射结果"。本次实测 `p.pick("SH600000")` 在某个切片里抛 `KeyError`，印证返回顺序确实会重排——**文档不只是说明，而是准确描述了会让 Agent 犯错的边界**。

把"Agent 最容易写错的地方"变成可检索的硬约束，是 agent-first 最直接的证据：它假定读者没有经验，只有文本。

### 2.3 正确性由框架结构保证（判据：结构性保证）

Agent 最危险的错误是"代码能跑、结果漂亮、但有前视偏差"。msd 把这类约束做进了执行路径：

- `ctx.run_selector` 只在**截至当前 bar** 的历史切片 `slice_panel` 上执行选股，动态轮动天然无未来函数；
- `load_data` 把财报报告期转换为法定披露截止日（04-30 / 08-31 / 10-31）再做 asof join，Agent 无需自己滞后报告日期；
- `PortfolioContext.set_universe` 白名单阻止买入池外标的；`rebalance_step` 先卖后买两遍执行；涨跌停不可买/不可卖、100 股取整、按日期规则计费、分红送转均由账户层处理；
- `Strategy` 类属性 `fin_fields` / `fin_only_year` / `warmup_bars` / `portfolio` 声明式驱动数据加载与运行模式，避免 Agent 漏配导致内存爆炸或模式错误。

结果：本次真正描述业务逻辑的调仓代码约 30 行，其余正确性不依赖 Agent 记得任何约定。

### 2.4 反馈闭环全部机器可读（判据：可自证）

这是本次体验中最强的证据。每一步都有无需人类解读的反馈：

| 环节 | 反馈形式 | Agent 可据此判断 |
| :--- | :--- | :--- |
| 选股执行 | 日志逐 step 输出耗时与 `remain N` | 哪一步把池子筛空了、过滤是否过严 |
| 因子检查 | `mbt.select.cli -f` 打印 polars 因子表（`pb/growth/roe/cheap/dull/quality/score`） | 打分方向、量纲、异常值是否符合经济含义 |
| 组合运行 | `## Rebalance Records` 逐笔给出 entry / exit / 持有盈亏 | 持有期是否足够长、"等待双击"是否真的发生 |
| 机制验证 | `finalize` 返回的 `diagnostics`（`rebalance_count=12`、`avg_holdings=20.0`、`buffer_exits`） | 调仓频率、分散度、缓冲带是否按设计生效 |
| 绩效评估 | `-S` markdown 报告含 Sharpe / MaxDD / fee_rate / trade_count | 单段行情是否只是噪声 |
| 过拟合检验 | `--holdout` 自动给出 IS/OOS 分割与 Overfit Ratio；`-b a,b,c` 多起点对比 | 结论是否依赖某一段行情 |

本次 `--holdout` 直接输出 `In-Sample Sharpe 1.20 / Out-of-Sample Sharpe 0.56 / Overfit Ratio 0.46`，迫使 Agent 如实报告"样本外明显走弱、价值风格暴露"，而不是包装成"策略表现优秀"。**能自动暴露自身弱点的框架，才适合交给不会自我怀疑的 Agent。**

### 2.5 迭代成本足以支撑会话内多轮闭环（判据：便宜、单一交付物）

实测数据（MSD_HOST 本地实例）：

- 全市场 5029 只 `load_snapshot` ≈ 0.04s；4824 只 × 60 bars 带 4 个财务字段 ≈ 0.08s；一次完整选股 ≈ 0.17s；
- 400 只 × 3.5 年组合回测数秒；多起点 3 次回测也在同一会话内完成；
- 交付物是**单个 `.py` 文件**，`uv run python -m mbt.run.cli <path> ...` 一条命令复现，无编译、无部署、无外部状态。

本次实际完成了"加载技能 → 核对 schema → 读契约 → 写文件 → 跑选股看因子 → 跑回测 → 读诊断 → 修正起点与警告 → 重跑样本外/多起点"的完整闭环，全部由工具调用与日志驱动，无需人工解释中间结果。这正是 Agent 优先框架的标志：**一次会话内可自证地收敛**。

---

## 3. 反论证：尚未闭合的部分

只列优点不构成论证。以下反例都来自本次实测，且性质一致——**缺口集中在 L3「语义自描述」，而非功能缺失**。

### 3.1 同一概念存在两套口径，文档未声明（最严重）

`load_data`（回测内）按**法定披露日**对齐财务；`MsdSelectorDataProvider.load_kline(fin_fields=...)`（起始日）按**报告期** `ts <= date` 做 backward join。同一个 `Selector` 在 `selector.execute(...)` 与 `ctx.run_selector(...)` 两处调用，2-4 / 6-8 / 10 月会看到尚未披露的财报。而 `financial.md` 只写"框架自动对齐，不要手动滞后"。

Agent 依赖文档作为契约，文档与实现的**静默偏差**（不报错、不 warning）会直接放大成看似合理的错误策略。本次只能靠外部手段缓解：把起点限定在 5/9/11 月、在策略里加 `SAFE_SELECT_MONTHS` 运行时警告、改写示例命令。这属于"Agent 自己补契约"，恰恰说明契约层未闭合。

### 3.2 版本耦合 bug 堵死文档推荐路径

polars 2.1.0 下 `MsdSelectorDataProvider.load_financial` 的 `pl.lit(self.date).cast(pl.Datetime)` 抛 `InvalidOperationError: casting from string to datetime is not supported`；而 `load_snapshot(fin_fields=[...])` 内部调用它，因此 CLI 传 `date=req.start` 时带财务字段的快照必崩。技能文档推荐的"先 `load_snapshot` 粗筛"快路径不可用，Agent 必须绕道 `load_kline(fin_fields=...)`。

需要读源码或试错才能绕过的知识，是 **agent-hostile** 的隐含知识。

### 3.3 数据覆盖不显式，错误表现不可定位

当前实例日线自 2022-09-08、财务自 2019-06-30、股本自 2021-09-30。用常规长周期起点 `-b 2015-05-04` 时，Selector 返回 0 只，CLI 抛 `ValueError: No objs provided`。错误信息指向"选股结果为空"而非"请求日期超出数据覆盖"，Agent 需额外执行多条探查命令才定位到根因。对比 3.2 的完整堆栈（可定位到具体行），这是反馈质量的两个极端：**执行层反馈优秀，数据层反馈缺失**。

### 3.4 财务历史窗口限制迫使策略降维

`load_data` 财务查询仅回溯 `query_start - 450` 天（年报 750 天），`ctx.run_selector` 的历史 DP 只能看到这些报告期。"盈利增长放缓"本应按时序刻画（当期增速相对前期回落），本次为避免口径差异与数据不足，简化为"当期增速处于横截面低分位"。这是框架的数据边界**改写了策略语义**，而这一边界并未在选股文档中说明。

### 3.5 关键边界行为只存在于实现层

本次为确认下列行为读取了 `site-packages` 中约 9 个源码文件（`selector.py`、`panel.py`、`portfolio.py`、`strategy.py`、`run/cli.py`、`run/backend.py`、`run/msd.py`、`run/report.py`、`select/msd.py`）：

- `try_load_selector` / `load_strategy` 取"第一个匹配子类"；
- universe guard 对退出股票 `min(0, curr_w) = 0` 仍会卖出；
- `dp.warmup_bars` 如何切分报告区间；
- `-a` 同名参数如何分发给 Selector 与 Strategy。

另有源码 docstring 残留 `PB = raw_close / f004`，而 `financial.md` 明确说明不存在 `f004`；`.schema` 必须用 `msd` 二进制，`pymsd.query_concat` 执行它会返回 `invalid table frame`。**知识留在实现层而非契约层，是 L3 未达标的直接度量。**

### 3.6 架构限制只在运行时事后提示

`PortfolioRunner` 的矩阵在起始日一次性分配，`run_selector` 只能在预加载池内过滤，无法选入池外新股；CLI 会打一条 warning，但此时数据已加载完毕。本次因此采用"起始日 400 只宽池 + 池内轮动"折中，并接受当前时点码表带来的幸存者偏差。对 Agent 而言，**事前约束优于事后警告**：后者只能写进交付说明，无法改变执行。

---

## 4. 综合判断

把证据按成熟度模型归类：

| 层级 | 状态 | 依据 |
| :--- | :--- | :--- |
| L1 可执行 | 已达成 | 单文件交付、一条命令复现、声明式配置 |
| L2 可自证 | 稳定达成，且超出同类框架 | step 日志、因子表、Rebalance Records、diagnostics、`--holdout`、多起点、结构化绩效报告 |
| L3 可自描述 | 部分达成 | SSOT 与 schema 核对是正面证据；asof 双口径、polars 兼容、覆盖区间不可查、边界行为需读源码是反面证据 |

因此命题应表述为：

> **msd stack 在"让 Agent 独立写出、跑通并验证一个量化策略"这条主链路上是 Agent 优先的；它尚未做到"让 Agent 在不读实现源码、不试探数据环境的前提下，确信自己写对了"。**

这个限定反而强化了论证：本次实证中，Agent 能在一次会话内完成从需求到样本外验证的闭环，且**能自行发现并定位上述所有缺口**——包括写出前视偏差警告、如实报告 OOS/IS = 0.46。一个框架能让 Agent 可靠地评估它自己，比"从不出错"更符合 Agent 优先的定义。

---

## 5. 可证伪的改进项与验收度量

若下列改动落地，L3 即成立；命题也可被量化检验。

1. **把对齐口径变成 API 参数**：`load_kline(..., fin_align="disclosure" | "period")`，默认 `disclosure`，消除"起始日必须落在 5/9/11 月"这类只能靠文档记忆的禁忌。
2. **提供机器可读的数据清单**：除 `.schema` 外，暴露每个表的 `min_ts/max_ts`、字段口径、对齐规则与单位，使覆盖区间成为一次查询可得的元数据；加载超出区间时直接报出可用范围。
3. **修复并守护推荐路径**：修正 polars 2.x 下 `load_financial` 的类型转换，并在 CI 冒烟测试中覆盖 `load_snapshot(fin_fields=[...])`。
4. **把边界行为提升到契约层**：明确写出"两处 Selector 调用的语义差异"、`run_selector` 仅池内过滤、warmup 切分语义、类发现顺序；提供 `mbt doctor` 输出环境/版本/数据覆盖的自检结果。
5. **清理契约内的矛盾**：统一 `f003` / `f004` 等字段示例，让 docstring 与参考文档互为可校验副本。

**验收度量**（可用同一任务重复评测）：让 Agent 在禁止读取 `site-packages` 的前提下完成同类双层策略，统计

- 首次运行成功率；
- 需要读取实现源码的次数（本次约 9 次）；
- 因数据口径/覆盖导致的返工次数（本次 2 次：起点越界、asof 不一致）；
- 从加载到产出样本外报告的轮次（本次 1 次会话内完成）。

这些数字若显著下降，"Agent 优先"就从定性评价变成可回归的工程指标。

---

## 附录：本次会话的 Agent 闭环轨迹（可复现）

```bash
# 1 加载能力与契约
use_skill msd-strategy-guide
msd shell -s "$MSD_HOST" ".schema stock_financial"      # SSOT：核对 f000/f003/f197/f205

# 2 单期选股与因子自检
uv run python -m mbt.select.cli strategy/011/davis_double_play.py -d 2024-05-10 -f

# 3 组合回测 + 机制诊断 + 样本内外
uv run python -m mbt.run.cli strategy/011/davis_double_play.py \
    -b 2023-05-04 -e 2026-11-10 -S --holdout 2025-01-01

# 4 多起点稳健性（不做参数寻优）
uv run python -m mbt.run.cli strategy/011/davis_double_play.py \
    -b 2023-05-04,2024-05-06,2025-05-06 -e 2026-11-10
```

| 轮次 | Agent 动作 | 框架反馈 | 决策 |
| :--- | :--- | :--- | :--- |
| 1 | 按类型读取 selector / portfolio / financial 文档 | 明确三层架构与单文件串联方式 | 选定 `Selector` + `PortfolioStrategy` |
| 2 | 现场核对 schema 与数值样例 | 字段中文名、累计 EPS / 同比增速语义 | 确定 4 个财务字段与 `raw_close/f003` 估值式 |
| 3 | 探测 `load_snapshot(fin_fields=...)` | polars 类型转换异常（完整堆栈） | 改用 `load_kline(fin_fields=...)` |
| 4 | 运行单期选股 `-f` | 400 行因子表、`remain` 逐步变化 | 确认打分方向与过滤强度 |
| 5 | 运行短周期回测 | Rebalance Records + diagnostics（6 次调仓、持仓 20） | 确认缓冲带/等权/财报日历生效 |
| 6 | 尝试 `-b 2015-05-04` | `No objs provided`（不可定位） | 探查各表最早 `ts`，改用 2023-05-04 起点 |
| 7 | 发现 asof 双口径 | 无报错、无提示（静默） | 限定安全月份、加运行时警告、修正示例 |
| 8 | 长周期 + holdout + 多起点 | OOS/IS Sharpe 0.46、3/3 跑赢基准 | 如实报告价值风格暴露，不调参 |

轨迹本身即论证：除第 3、6、7 轮需要额外探查（对应第 3 节的 L3 缺口）外，其余各轮都由框架的结构化反馈直接驱动 Agent 自主决策。
