# msd-backtest

`msd-backtest` 是一个基于 Python 的高性能、向量化股票策略回测框架。它旨在为量化开发者提供一个简单、高效且灵活的工具，用于开发、测试和优化交易策略。

## 🌟 核心特性

- **高性能向量化计算**：核心数据处理基于 NumPy，支持大规模数据的快速回测。
- **多标的支持**：原生支持同时对多个股票或指数进行回测。
- **模块化架构**：
  - `Account`：模拟真实交易账户，处理资金、持仓、手续费和滑点。
  - `Strategy`：解耦策略逻辑，通过继承 `Strategy` 基类快速实现自定义逻辑。
  - `DataProvider`：灵活的数据提供接口，支持接入不同的行情数据源。
- **完善的报表输出**：支持生成 CSV、Excel (XLSX) 和 JSON 格式的回测结果报表。
- **内置 MSD 集成**：支持从 MSD (Market Strategy Data) 服务端直接拉取行情和财务数据。
- **远程回测支持**：内置 FastAPI 服务器，支持跨语言或远程调用回测引擎。

## 🚀 快速开始

### 安装

推荐使用 `uv` 进行依赖管理：

```bash
# 克隆仓库
git clone https://github.com/your-repo/msd-backtest.git
cd msd-backtest

# 安装依赖
uv sync

# 如果需要运行回测服务器
uv sync --with server
```

### 运行回测

你可以通过命令行工具直接运行已有的策略：

```bash
uv run python -m mbt.run.cli strategy/double_avg.py -s 000001.SZ 600000.SH --begin 2023-01-01 --end 2023-12-31 -o report.xlsx
```

**常用参数说明：**
- `strategy`: 策略文件路径（`.py` 或 `.json`）。
- `-s`, `--symbols`: 股票代码列表。
- `-b`, `--begin`: 开始日期 (YYYY-MM-DD)。
- `-e`, `--end`: 结束日期 (YYYY-MM-DD)。
- `-o`, `--output`: 输出文件路径（支持 `.csv`, `.xlsx`, `.json`）。

## 🛡️ 回测机制：杜绝未来数据

`msd-backtest` 采用**逐条数据迭代**（Bar-by-Bar Iteration）的执行机制，确保回测过程的严谨性和真实性。

- **单向时间线**：回测引擎（`mbt.Runner`）严格按照时间顺序，逐个时间步调用策略的 `execute` 方法。
- **数据隔离**：在任一时刻，策略仅能通过 `ctx` 访问当前及之前的数据，无法窥视未来的行情。
- **真实模拟**：所有的买卖指令都在当前 Bar 生成，并在当前或下一个 Bar 的价格基础上执行（取决于配置和滑点），完美模拟真实市场的环境。

## 💡 编写你的策略

编写策略非常简单，只需要继承 `mbt.Strategy` 并实现 `execute` 方法：

```python
from mbt import Strategy, Context

class MyStrategy(Strategy):
    def __init__(self, fast_period=5, slow_period=20):
        self.fast_period = fast_period
        self.slow_period = slow_period

    def execute(self, ctx: Context):
        # 获取当前收盘价序列
        close = ctx.close 
        
        # 计算移动平均线
        # 注意：ctx 提供的是 NumPy 数组，支持向量化操作
        # 这里仅为示意，实际开发中建议预计算指标
        
        # 买入逻辑示例
        # ctx.buy(buy_flags, percent=1.0)
        
        # 卖出逻辑示例
        # ctx.sell(sell_flags, percent=1.0)
        pass
```

## 🧠 AI 辅助策略编写

本项目深度集成 AI 辅助开发流程。利用 [msd_strategy](file:///.agents/skills/msd-strategy/SKILL.md) 技能，AI 可以根据用户描述自动生成符合规范的策略代码。

### 核心方法与目标
- **目标：** 通过自然语言描述，由 AI 生成高效、可运行的 Python 策略类。
- **技术栈：** Python 3.11+, NumPy, `py-alpha-lib`。
- **编码规范：**
  - 必须继承 `mbt.Strategy`。
  - 实现 `execute(self, ctx: mbt.Context)` 方法。
  - 使用 `ctx.buy()` 和 `ctx.sell()` 进行下单。

### 常用算子库 (alpha-lib)
为了支持 AI 高效构建因子，框架内置了丰富的 `alpha-lib` 算子，包括：
- **时间序列算子：** `MA`, `EMA`, `HHV`, `LLV`, `STDDEV`, `CORR`, `SLOPE` 等。
- **横截面算子：** `CC_RANK`, `CC_ZSCORE`, `GROUP_RANK`, `NEUTRALIZE` 等。
- **逻辑算子：** `CROSS`, `LONGCROSS`, `BARSLAST` 等。

AI 在编写策略时应优先使用这些向量化算子，以确保最佳性能。

## 📂 项目结构

```text
.
├── mbt/                # 核心回测包
│   ├── account.py      # 账户、持仓、手续费管理
│   ├── strategy.py     # 策略基类与上下文定义
│   ├── data_provider.py # 数据提供接口
│   └── run/            # 运行引擎
│       ├── runner.py   # 回测执行核心
│       ├── cli.py      # 命令行界面
│       ├── msd.py      # MSD 数据源接入
│       └── server.py    # REST API 服务
├── strategy/           # 示例策略目录
├── tests/              # 单元测试
└── pyproject.toml      # 项目配置与依赖说明
```

## 🔧 进阶配置

### 手续费配置 (Fee)

在 `mbt/account.py` 中支持自定义手续费。你可以通过 JSON 配置文件或代码中初始化 `Fee` 类来设置不同时间段的印花税、过户费及佣金。

### 复权处理

`msd-backtest` 默认在 `mbt/run/msd.py` 中实现了前/后复权的计算逻辑，确保回测收益计算的准确性。

## 🛠️ 开发与测试

运行单元测试：

```bash
uv run pytest
```

运行性能基准测试：

```bash
uv run pytest --benchmark-only
```

---

*更多详细文档请参考各模块代码注释。*
