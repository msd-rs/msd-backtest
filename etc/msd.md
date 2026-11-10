# MSD 安装与使用说明

`msd` 是单可执行文件、开箱即用（免复杂配置）的高性能量化时序数据库。
- 开源项目主站：https://github.com/msd-rs/msd
- 国内镜像仓库：https://cnb.cool/elsejj/msd-rs

---

## 数据库维护与管理（使用 `msd_setup.py`）

项目提供了自动化运维脚本 `msd_setup.py`（位于 `.agents/skills/msd-strategy-guide/scripts/msd_setup.py`），所有安装、后台守护、健康检查、建表与月度数据同步均通过该脚本完成，**切勿自行手动执行前台常驻命令或编写下载解压脚本**。

### 常用子命令

```bash
# 1. 查看数据库状态（目录、是否启动、MSD_HOST、最新数据时间）
uv run python .agents/skills/msd-strategy-guide/scripts/msd_setup.py info

# 2. 一键启动服务（若未安装则自动执行安装与全量导入）
uv run python .agents/skills/msd-strategy-guide/scripts/msd_setup.py up

# 3. 完整安装 MSD（下载二进制、启动服务、建表并导入自历史总包 2026-09 至当前月份的全部数据）
uv run python .agents/skills/msd-strategy-guide/scripts/msd_setup.py install

# 4. 更新最新月份数据（自动补齐自数据库最新交易日至当月的数据）
uv run python .agents/skills/msd-strategy-guide/scripts/msd_setup.py update

# 也可通过 -d 指定更新特定月份
uv run python .agents/skills/msd-strategy-guide/scripts/msd_setup.py update -d 2026-11
```

### 常用参数说明

- `-w, --workdir`: MSD 数据库存储目录，默认取环境变量 `$MSD_WORKDIR` 或用户级目录 `~/data/msd/`。
- `-H, --host`: MSD 服务地址，默认取环境变量 `$MSD_HOST` 或 `http://127.0.0.1:50510`。
- `install --force`: 强制重新下载并解压可执行文件。
- `install --skip-import`: 仅安装启动与建表，跳过数据导入。

---

## 客户端与 Python 调用

### 1. 命令行交互 CLI
进入数据库所在目录运行客户端：
```bash
./msd shell
# 执行单条命令或查询
./msd shell "select ts from stock_kline_1d where obj='SH000001' limit -1"
# 查看所有表
./msd shell ".tables"
```

### 2. Python 开发推荐（`pymsd`）
在 Python 策略中推荐使用 `pymsd` 进行高效查询与数据交互：
```python
import os
import pymsd

msd_host = os.environ.get("MSD_HOST", "http://127.0.0.1:50510")
client = pymsd.MsdConnection(msd_host)

# 查询日线最后 200 条记录
res = client.query_concat("select * from stock_kline_1d where obj in ('SH600000', 'SZ000001') limit -200")
```

---

## MSD 核心特性与 SQL 查询约束

MSD 的核心理念是：**既然 SQL 无法表达所有计算，就专注于最基础高效的过滤，并将数据快速下发给下游计算引擎（如 numpy, polars, pandas）**。

- **数据引擎**：
  - `ts`（时序引擎，默认）：必须有 `ts` 字段，表强制按 `obj` 分区（如日线、财务、分红）。
  - `kv`（键值引擎）：两列结构（`obj` 为键，另一列为值，如板块、股票码表）。
- **SQL WHERE 约束**：
  - 仅支持 `obj` 和 `ts` 过滤。
  - `ts` 支持比较运算符（`>`, `>=`, `=`, `<=`, `<`）及时间部分匹配（如 `ts.month=12` 可在财务表中直接过滤年报）。
  - `obj` 支持 `=`、`IN ("SH000001", "SZ000001")`、前缀通配符 `obj = "SH6000*"`、关联 KV 表项 `obj="@stock_block:上证A股"`。
- **LIMIT 负数语法**：
  - `LIMIT n`: 返回前 n 行。
  - `LIMIT -n`: 直接返回 ts 序的最末 n 行（获取最新行情无需先降序后升序）。
- **HTTP 服务接口**：
  - `/health`: 健康检查，返回 `200 OK`。
  - `/query`: POST 请求执行 SQL，返回 NDJSON 流。
  - `/table/{table_name}`: PUT 数据导入接口。
