# msd-backtest

`msd-backtest` 是一个基于 Python 的高性能、向量化量化投研与回测框架。框架专为现代 A 股量化研发打造，深度融合 [MSD (Market Strategy Data)](.agents/skills/msd-strategy-guide/references/msd.md) 数据服务与 [py-alpha-lib](.agents/skills/msd-strategy-guide/references/algo.md) 并行算法库，全面支持**横截面选股（Selector）**、**单标的独立交易策略（Trading Strategy）**以及**多标的多资产组合配置与动态调仓策略（Portfolio Strategy）**。

---

## 📖 极佳的代码自解释性与 Pydoc 支持

> [!TIP]
> **`mbt` 核心库自带极其详尽、规范的 Python Docstring (pydoc)。**
> 框架对各个类、方法、数据契约（Data Contracts）、字段语义（如复权与不复权价格、财报对齐规则）以及生命周期钩子都编写了深入的文档与代码示例。你无需频繁查阅外部手册，直接利用内置工具或 IDE 即可随时查阅最权威的 API 说明：

- **命令行查看模块与类文档：**
  ```bash
  # 查看整个 mbt 顶层包与核心类概览
  uv run python -m pydoc mbt

  # 查看策略上下文与交易操作 API
  uv run python -m pydoc mbt.Context
  uv run python -m pydoc mbt.PortfolioContext

  # 查看选股器与数据提供者（包含矩阵布局契约与字段语义）
  uv run python -m pydoc mbt.select.SelectorDataProvider
  uv run python -m pydoc mbt.select.Panel
  ```
- **启动本地 Web 文档服务器：**
  ```bash
  uv run python -m pydoc -p 8080
  # 随后在浏览器中打开 http://localhost:8080/mbt.html 即可通过可视化界面浏览完整 API
  ```
- **交互式环境与 IDE 支持：**
  在 Python REPL 中可直接调用 `help(mbt.Strategy)`；在 VS Code、Cursor、Antigravity、PyCharm 等主流现代 IDE 中，将鼠标悬停在类、方法或属性上，即可自动获得完整的参数签名、类型注解与中文/英文语义注解。

---

## 🌟 核心特性

- **三位一体的策略架构**：
  - **选股策略（`mbt.Selector`）**：多阶段漏斗式选股（`stepXX`），基于 `Panel` 二维矩阵与 Polars 进行高效率横截面因子筛选，无缝对齐财务指标与快照数据。
  - **交易策略（`mbt.Strategy`）**：单标的独立账户机制，用于细粒度验证个股时间序列上的择时信号与买卖点逻辑。
  - **组合策略（`mbt.PortfolioStrategy`）**：多标的共享单一现金池（`PortfolioAccount`），内置权重调仓（`ctx.rebalance`）、两阶段撮合（先卖后买）、交易限额、停牌与涨跌停约束。
- **单文件双层联合架构**：
  - 单个 Python 文件可同时定义 `Selector` 和 `PortfolioStrategy`（或 `Strategy`）。
  - 回测命令行自动识别：先由选股器筛选标的池，再自动挂载到交易策略执行回测；支持参数统一注入与自动派发。
- **严格杜绝未来函数（No-Lookahead Bias）**：
  - 核心执行基于单向时间步，策略在任意时间步仅能访问当前及历史截面数据。
  - 支持策略在运行期动态调用选股器（`ctx.run_selector`），历史数据按时间轴安全切片，杜绝未来信息穿越。
- **高性能向量化与三阶段生命周期**：
  - `prepare()`：全时序预计算技术指标与静态因子，将复杂度从 $O(N \times T^2)$ 降低到 $O(N \times T)$。
  - `execute()`：单向时间步推进，响应动态账户状态（如持仓收益率 `ror_hold`）、执行下单与调仓。
  - `finalize()`：回测收尾阶段统一统计诊断指标、平仓/退出原因归因，自动并入报告中。
- **严密的价格与基本面语义隔离**：
  - 前复权价格（`close`, `open`, `high`, `low`）：专用于平滑时序技术指标与动量计算。
  - 原始不复权价格（`raw_close` / `price`, `raw_open` 等）：强制用于基本面估值（如市净率 PB = `raw_close / f003`）与总市值计算，杜绝因高送转复权导致估值失真的严重陷阱。
- **开箱即用的专业回测生态**：
  - 内置 MSD 数据库无缝驱动与 `stock_financial` 财报数据 Asof 关联。
  - 完善的评估体系：样本外验证（Holdout / OOS）、多起点回测（Multi-start）、预热机制（Warmup Bars）、基准对照与紧凑 Markdown 诊断报告。

---

## 🚀 快速开始

### 1. 环境与依赖

推荐使用现代 Python 包管理器 `uv`（要求 Python 3.12+）：

```bash
# 克隆仓库
git clone https://github.com/msd-rs/msd-backtest.git
cd msd-backtest

# 安装核心及 CLI 依赖
uv sync --with cli

# 若需启动远程回测服务器（可选）
uv sync --with server
```

### 2. 环境配置

在项目根目录下创建或编辑 `.env` 文件，配置 MSD 数据库地址与费率规则：

```env
# MSD 服务端地址（必须）
MSD_HOST=http://localhost:50510

# 基准指数代码（可选，默认为上证指数 SH000001）
MSD_BT_BENCHMARK=SH000001

# 手续费配置文件（可选，未指定时默认使用 etc/fee.json 标准印花税与佣金）
# MSD_BT_FEE=etc/fee.json
```

---

## 🧩 策略开发指南

框架支持三种核心策略形态，均支持在单文件中单独或组合使用。

### 1. 选股策略 (`mbt.Selector`)

用于横截面数据处理与多条件逐步过滤。通过继承 `mbt.Selector` 并编写有序的 `stepXX` 步骤方法逐步缩小候选标的：

```python
import mbt
import alpha as al
import polars as pl
from mbt.select import A_STOCKS_EXCLUDE_ST

class MySelector(mbt.Selector):
  def __init__(self, pb_limit: float = 3.0, ma_window: int = 20):
    super().__init__()
    self.pb_limit = pb_limit
    self.ma_window = ma_window

  def step00(self, stocks: list[str]) -> list[str]:
    """步骤0：初始化股票池，剔除 ST 与停牌股票"""
    if not stocks:
      stocks = A_STOCKS_EXCLUDE_ST
    return stocks

  def step01(self, stocks: list[str]) -> list[str]:
    """步骤1：基于快照过滤低估值（注意使用 raw_close 进行估值）"""
    snap = self.dp.load_snapshot(stocks, fin_fields=["f003"]) # f003 为每股净资产
    snap = snap.with_columns((pl.col("raw_close") / pl.col("f003")).alias("pb"))
    filtered = snap.filter(pl.col("pb") < self.pb_limit)
    return filtered["obj"].to_list()

  def step02(self, stocks: list[str]) -> list[str]:
    """步骤2：基于历史 K 线 Panel 计算均线多头动量"""
    panel = self.dp.load_kline(stocks, lastN=self.ma_window + 10, baseline="SH000001")
    # 利用 panel.compute 向量化计算
    ma = panel.compute(al.MA, "close", self.ma_window)
    last_close = panel.last("close")
    last_ma = ma[:, -1]
    keep_indices = [i for i, (c, m) in enumerate(zip(last_close, last_ma)) if c > m]
    return [panel.objs[i] for i in keep_indices]
```

选股器可以通过命令行直接运行：
```bash
uv run python -m mbt.select.cli strategy/my_selector.py -d 2025-01-01 -f
```

---

### 2. 单标的独立交易策略 (`mbt.Strategy`)

适用于开发各标的独立核算的交易策略，或用于批量评估技术信号的通用有效性：

```python
import mbt
import alpha as al

class MyTradingStrategy(mbt.Strategy):
  fin_fields = ["f001"] # 声明所需财报字段，避免全量加载节省内存
  warmup_bars = 20      # 设置指标预热周期

  def __init__(self, period: int = 20):
    super().__init__()
    self.period = period

  def prepare(self, ctx: mbt.Context):
    # 静态指标在 prepare 阶段 O(N*T) 一次性全量预计算
    self.ma = al.MA(ctx.close, self.period)

  def execute(self, ctx: mbt.Context):
    # 逐日单向推进，获取当前截面信号
    current_close = ctx.close[:, ctx.bar]
    current_ma = self.ma[:, ctx.bar]
    
    buy_signals = current_close > current_ma
    sell_signals = current_close < current_ma
    
    ctx.buy(buy_signals, percent=1.0)
    ctx.sell(sell_signals, percent=1.0)

  def finalize(self, ctx: mbt.Context) -> dict | None:
    # 回测结束收尾诊断
    return {"total_bars_processed": ctx.bars}
```

---

### 3. 多标的组合配置策略 (`mbt.PortfolioStrategy`)

适用于实盘资产配置与投资组合管理。所有标的共享同一个大账户资金池，按目标权重自动执行调仓（先卖后买，释放现金）：

```python
import mbt
import numpy as np
import alpha as al

class MyPortfolioStrategy(mbt.PortfolioStrategy):
  def __init__(self, top_k: int = 5, rebalance_interval: int = 10):
    super().__init__()
    self.top_k = top_k
    self.rebalance_interval = rebalance_interval

  def prepare(self, ctx: mbt.PortfolioContext):
    # 预计算动量指标
    self.roc = al.ROC(ctx.close, 20)

  def execute(self, ctx: mbt.PortfolioContext):
    # 仅在调仓日进行重平衡
    if ctx.bar % self.rebalance_interval != 0:
      return

    # 获取当前横截面动量因子并选取 Top K
    current_roc = self.roc[:, ctx.bar]
    top_indices = np.argsort(current_roc)[-self.top_k:]

    # 构建目标权重（等权重配置）
    weights = np.zeros(ctx.n_objs)
    weights[top_indices] = 1.0 / self.top_k

    # 自动执行组合再平衡（卖出多余/不再持有的标的，腾出可用资金买入目标标的）
    ctx.rebalance(weights)
```

---

### 4. 选股与交易双层联合策略（单文件联动）

可以在同一个策略文件中同时编写 `Selector` 和 `PortfolioStrategy`（或 `Strategy`）。

#### 方式 A：静态两阶段串联
在同一个文件中定义两者，CLI 运行时无需传入 `-s`，系统将自动先运行 `Selector` 产生股票池，再作为输入运行 `Strategy`：
```bash
uv run python -m mbt.run.cli strategy/dual_momentum.py -b 2024-01-01 -e 2024-12-31 -S
```
传入的 `-a key=value` 参数会根据形参名自动精准派发给 `Selector` 与 `Strategy` 的 `__init__`。

#### 方式 B：动态轮动与在途淘汰（Dual-Layer Runoff）
策略在 `PortfolioStrategy.execute` 内部通过 `ctx.run_selector` 周期性安全刷新白名单，实现标的动态轮换与自然平仓退出：
```python
class AdvancedPortfolio(mbt.PortfolioStrategy):
  def __init__(self, selector: mbt.Selector, reselect_days: int = 20):
    super().__init__()
    self.selector = selector
    self.reselect_days = reselect_days

  def execute(self, ctx: mbt.PortfolioContext):
    # 1. 周期性使用无未来函数的历史切片刷新活跃标的池
    if ctx.bar % self.reselect_days == 0 or not ctx.universe:
      new_pool = ctx.run_selector(self.selector, lastN=120)
      ctx.set_universe(new_pool)

    # 2. 淘汰池（Runoff）保护：未在活跃池但仍有持仓的标的不会被买入，但可通过平仓安全退出
    for obj in ctx.account.runoff_objs(ctx.universe):
      ctx.sell(obj)

    # 3. 对活跃池内标的按信号买入
    for obj in ctx.universe:
      if not ctx.account.has_position(obj):
        ctx.buy(obj, percent=0.2)
```

---

## 💻 命令行运行与核心选项

### 运行交易/组合策略 (`mbt.run.cli`)

```bash
# 1. 对指定股票运行策略，并输出终端 Markdown 精简统计摘要 (-S)
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 SZ000001 -b 2023-01-01 -e 2023-12-31 -S

# 2. 结合独立选股脚本，先选股再回测
uv run python -m mbt.run.cli strategy/my_strategy.py -s strategy/my_selector.py -b 2023-01-01 -e 2023-12-31 -S

# 3. 运行共享现金组合回测模式 (-p)，指定初始资金为 100 万 (-c)
uv run python -m mbt.run.cli strategy/portfolio_strategy.py -s SH600000 SZ000001 -p -c 1000000 -b 2023-01-01 -e 2023-12-31 -S

# 4. 样本内/样本外验证 (Holdout)，指定后半段为测试集
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 -b 2022-01-01 -e 2024-12-31 --holdout 2024-01-01 -S

# 5. 多起点稳定性评估 (-b 支持多次传入或逗号分隔)
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 -b 2022-01-01,2023-01-01,2024-01-01 -e 2024-12-31 -S

# 6. 导出完整回测报告（支持 .xlsx, .json, .csv）
uv run python -m mbt.run.cli strategy/my_strategy.py -s SH600000 -b 2023-01-01 -e 2023-12-31 -o report.xlsx
```

**常用选项速查：**
| 参数 | 说明 |
| :--- | :--- |
| `strategy` | 策略文件路径（`.py` 或 `.json`） |
| `-s, --objs` | 标的代码列表（如 `SH600000`）或选股器路径（`.py`） |
| `-b, --begin` | 开始日期 (YYYY-MM-DD)，支持多次指定进行多起点回测 |
| `-e, --end` | 结束日期 (YYYY-MM-DD) |
| `-p, --portfolio` | 强制开启共享现金组合回测模式（`PortfolioStrategy` 默认自动开启） |
| `-c, --cash` | 初始资金（单标的模式默认 10,000，组合模式默认 1,000,000） |
| `-S, --summary` | 在终端打印包含收益率、夏普比率、最大回撤等指标的 Markdown 摘要 |
| `-a, --args` | 策略/选股器入参，以 `key=value` 格式传递 |
| `--holdout` | 样本外切分点（如 `2024-01-01` 或 `180d`），对比样本内外表现 |
| `--warmup-bars` | 指标预热 Bar 数量（预热期间不撮合订单） |
| `-o, --output` | 结果输出路径（支持 `.xlsx`, `.csv`, `.json`） |

---

## 🧠 AI 辅助策略编写与调优

本项目深度集成量化 AI 研发助手机制。项目中内置了专门的 Agent 技能指南：
- 核心技能定义：[`msd-strategy-guide`](.agents/skills/msd-strategy-guide/SKILL.md)
- 详细参考规范：
  - [选股器构建指引](.agents/skills/msd-strategy-guide/references/selector.md)
  - [交易策略构建指引](.agents/skills/msd-strategy-guide/references/trading.md)
  - [组合调仓策略指引](.agents/skills/msd-strategy-guide/references/portfolio.md)
  - [财务字段与对齐规范](.agents/skills/msd-strategy-guide/references/financial.md)
  - [策略调优与防过拟合指南](.agents/skills/msd-strategy-guide/references/optimize.md)

AI 助手能够基于自然语言描述，依据上述规范生成经过向量化优化、杜绝未来函数、严格对齐价格语义的高质量策略代码。

---

## 📂 项目结构

```text
.
├── mbt/                     # 核心量化回测与策略包
│   ├── account.py           # 单标的账户、持仓、手续费（Fee）模型
│   ├── portfolio.py         # 共享资金组合账户、调仓撮合与多资产上下文
│   ├── strategy.py          # 策略基类（Strategy, PortfolioStrategy）与 Context
│   ├── data_provider.py     # 数据源抽象接口
│   ├── select/              # 选股器模块
│   │   ├── selector.py      # Selector 基类与静态选股器
│   │   ├── panel.py         # 2D Panel 矩阵视图与向量化算子计算
│   │   ├── msd.py           # 选股端 MSD 行情与快照接入
│   │   └── cli.py           # 选股命令行工具
│   ├── run/                 # 交易执行与回测运行器
│   │   ├── runner.py        # 核心回测推进引擎
│   │   ├── cli.py           # 回测命令行界面
│   │   ├── report.py        # 报告生成与统计指标（Markdown / Excel / JSON）
│   │   ├── msd.py           # 交易端 MSD 数据加载与复权处理
│   │   └── server.py        # REST API 回测服务
│   └── shared/              # 共享金融工具与常数（财务字段、股票列表定义）
├── strategy/                # 策略与选股器示例
├── tests/                   # 单元测试与性能测试集
├── etc/                     # 默认配置（如 fee.json 费率定义）
└── pyproject.toml           # 项目配置与依赖定义
```

---

## 🛠️ 测试与开发

```bash
# 运行全部单元测试
uv run pytest

# 运行性能基准测试
uv run pytest --benchmark-only
```
