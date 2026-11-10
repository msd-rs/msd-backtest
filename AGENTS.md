# msd stack: AI Agent 优先的量化策略框架

`msd-backtest` (mbt) 及其所属的 `msd stack` 是一个**面向 AI Agent 优先设计（Agent-First）**的高性能股票量化投研与回测框架。

---

## 🎯 核心哲学：什么是「AI Agent 优先」

> **核心矛盾：人类的「容错兜底」，是 Agent 的「致命盲区」。**
> 
> - **人类开发者**：具备跨会话常识、直觉与经验，遇到 warning 会下意识排查，面对数据缺失会主动询问。
> - **AI Agent**：没有跨会话隐性记忆，完全依赖字面契约与确定性规则；缺乏直觉自审能力；极容易被看似成功的结果误导（产生虚假自信）。
> - **设计结论**：**对 Agent 而言，一个会显式报错、大喊大叫的框架，远胜过一个体贴兜底的框架**。在数据不匹配或配置错误时，框架一旦静默兜底（如将选股池静默截断为首日 20 只、池外标的仅发 warning 丢弃、参数未识别被静默吞掉），Agent 会完全确信自己做对了，产出看似正确但逻辑失真的策略。
> 
> 因此，"Agent 优先" 的本质是：**把正确性保证、数据语义、防错机制与反馈闭环，从人类的经验直觉中，彻底下沉到框架和工具链的硬约束与自描述中。**

---

## 🏛️ 「Agent 优先框架」的六大核心特征

| # | 核心特征 | 内涵与标准 | 反例（人类可忍，Agent 致命） |
|---|---|---|---|
| ① | **契约同仓且可机读** | 接口约定、数据矩阵布局（Layout Contract）、字段口径内嵌在类 docstring 与同仓文档中，一条命令（如 `pydoc`）或代码跳转即可检索。 | 契约散落在外部 Wiki、口头传承，或代码与文档口径漂移 |
| ② | **极小迭代闭环** | 单文件纯 Python 交付（同文件定义 `Selector` 与 `Strategy`），单命令运行，秒级执行，无编译、无代码生成、无复杂部署。 | 修改一个逻辑需要改配置、编译、重启服务 |
| ③ | **正确性下沉为框架不变量** | 财报披露日对齐、回测内选股无前视切片、价格复权双轨制、两阶段撮合、整手与涨跌停约束全部由底层强制保证。 | 让 Agent 自己写时序对齐，策略写错也能跑且收益虚高 |
| ④ | **数据自描述与可抽样验证** | 表结构（Schema）与字段中文语义自描述，支持一条 SQL 现场抽样探查，数据边界与起止时间可查询。 | 字段名靠猜，不知道样本覆盖范围，盲目试错 |
| ⑤ | **全量结构化反馈与语义诊断** | 回测输出纯文本/Markdown 报表与 JSON，内置 `finalize() -> dict` 语义诊断通道，把策略行为转为可断言的数值。 | 只产出一张收益曲线图，Agent 无法解析运行细节 |
| ⑥ | **确定性与内建可自我证伪** | 消除隐藏随机源，同输入必同输出；内建多起点（`-b`）、样本外检验（`--holdout`）与预热（`--warmup-bars`），自动暴露过拟合。 | 无法划分样本内外，Agent 拿着单一过拟合区间自我欺骗 |

---

## 🛡️ 框架开发与维护的硬性设计规则 (Rules of Engagement)

在开发、扩展或修改 `mbt` 框架代码以及策略库时，必须严格遵守以下规则：

### 1. 零容忍静默降级（Fail-Fast 原则，最高准则）
- **严禁静默兜底**：当输入参数、数据区间或执行逻辑异常时，**必须显式抛出结构化异常**，严禁仅打印 Warning 后静默降级继续执行。
- **严禁静默吞参数**：命令行 `-a key=value` 若未被 `Selector` 或 `Strategy` 声明接收，必须报错中断，禁止静默忽略。
- **严禁隐蔽退化标的池**：组合策略回测必须明确要求传入标的池或全市场代码，严禁在缺少 `-s` 时静默退化为首日选出的固定标的。
- **池外标的越界报警**：运行期动态选股选出底层预加载池外的标的时，必须显式抛错或强制拦截并给出明确指引。

### 2. 严格的价格与财报语义隔离
- **复权价与不复权价双轨制**：
  - 前复权价格（`close`, `open`, `high`, `low`）：仅用于连续技术形态指标与动量计算。
  - 原始未复权价格（`raw_close` / `price`, `raw_open` 等）：必须用于基本面估值（PB = `raw_close / f003`、PE）及真实总市值计算。
- **无未来函数时间对齐**：
  - 选股端与回测端所有涉及财报数据的逻辑，必须严格基于**法定披露截止日**（04-30 / 08-31 / 10-31）做 backward asof 对齐，禁止按报告期直接对齐。

### 3. 文档契约即单一事实源（SSOT）
- `mbt` 的核心类与方法必须保持详尽、规范的 Python Docstring。
- Docstring 中的类型标注、矩阵布局约定与字段名称（如 `f003`）必须与真实实现绝对一致，禁止遗留错误字段或过期范例。

### 4. 报错信息具备即时可定位性（Actionable Errors）
- 错误信息必须直接说明根因并指明上下文。
- 例如：当请求的开始日期超出数据库历史范围时，必须明确提示 `Requested start '2015-05-04' is earlier than dataset min_ts '2022-09-08'`，严禁抛出无上下文的次生错误（如 `ValueError: No objs provided`）。

---

## 💻 技术栈与开发环境规范

- **包管理工具**：统一使用 `uv` 进行依赖管理与环境隔离。
  - 安装依赖：`uv add <pkg>`
  - 运行命令：`uv run <command>`
  - 启动 Python：使用 `uv run python` 或 `uv python`，不要直接使用裸 `python`。
- **语言规范**：Python 3.12+，代码缩进采用 **2 个空格**。
- **测试框架**：统一使用 `pytest`，测试代码置于 `tests/` 目录下，运行测试：`uv run pytest`。
- **算法与数据处理**：核心因子基于 `py-alpha-lib` 向量化算子库，结合 `NumPy` 和 `Polars` 进行高效计算。
- **数据库服务**：MSD 数据库服务默认地址为 `http://localhost:50511`。
- **标的代码规范**：标准 A 股代码格式，附带市场前缀（如 `SH600000`, `SZ000001`）。

---

## 📂 项目结构概览

- `mbt/`：框架核心代码库
  - `strategy.py`：策略基类（`Strategy`, `PortfolioStrategy`）与执行上下文（`Context`）
  - `account.py`：单标的账户、持仓与手续费（`Fee`）模型
  - `portfolio.py`：共享资金组合账户（`PortfolioAccount`）、调仓撮合引擎与 `PortfolioContext`
  - `data_provider.py`：数据提供抽象接口
  - `select/`：选股子系统
    - `selector.py`：选股器基类 `Selector` 与 `StaticSelector`
    - `panel.py`：2D 矩阵视图 `Panel` 与向量化算子计算
    - `msd.py`：选股端 MSD 行情与快照接入
    - `cli.py`：选股命令行工具 (`uv run python -m mbt.select.cli`)
  - `run/`：回测与交易执行子系统
    - `runner.py`：单标的回测推进核心
    - `cli.py`：回测命令行工具 (`uv run python -m mbt.run.cli`)
    - `report.py`：统计指标分析、Markdown 精简报告与数据导出
    - `msd.py`：交易端 MSD 数据加载与复权处理
    - `server.py`：HTTP REST 回测服务
  - `shared/`：公共常量与金融工具（财务字段定义、股票池预定义常量）
- `strategy/`：策略样例目录
- `tests/`：单元测试与性能基准测试集
- `etc/`：配置文件（如默认费率配置 `fee.json`）
- `pyproject.toml`：项目元数据与依赖定义

---

## 🚀 常用运行命令速查

### 1. 运行策略回测 (`mbt.run.cli`)
```bash
# 单标的/多标的回测，输出 Markdown 精简诊断 (-S)
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 SZ000001 -b 2023-01-01 -e 2023-12-31 -S

# 共享现金组合模式回测 (-p)，初始资金 100 万 (-c)
uv run python -m mbt.run.cli strategy/my_portfolio.py -s SH600000 SZ000001 -p -c 1000000 -b 2023-01-01 -e 2023-12-31 -S

# 单文件双层联动回测（自动先运行 Selector，再运行 Strategy）
uv run python -m mbt.run.cli strategy/davis_double_play.py -b 2023-05-04 -e 2026-11-10 -S

# 样本外验证 (--holdout) 与多起点稳定性检验 (-b 支持逗号分隔)
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 -b 2022-01-01 -e 2024-12-31 --holdout 2024-01-01 -S
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 -b 2022-01-01,2023-01-01,2024-01-01 -e 2024-12-31 -S
```

### 2. 运行单期选股 (`mbt.select.cli`)
```bash
# 运行选股器并打印中间因子表 (-f)
uv run python -m mbt.select.cli strategy/my_selector.py -d 2024-05-10 -f
```

### 3. API 与 Pydoc 查阅
```bash
# 查看模块或类的详细机读契约
uv run python -m pydoc mbt
uv run python -m pydoc mbt.Context
uv run python -m pydoc mbt.select.Panel

# 启动本地可视化文档服务
uv run python -m pydoc -p 8080
```
