这是高性能量化计算时序数据库 `msd` 的安装使用说明

- `msd` 是单可执行文件, 开箱即用(免初始配置), 其安装包约 13MB 左右, 解压后约 30MB
- `msd` 是开源项目, 主站是 https://github.com/msd-rs/msd , 通过 github action 按版本 v主.次.修 进行发布.
- `msd` 的中国国内镜像是在 https://cnb.cool/elsejj/msd-rs , 它会同步 github 的编译产物, 以便在 github 不可访问时进行镜像.


# 下载

## 下载地址

- Github: (最新版本)[https://github.com/msd-rs/msd/releases/latest]
- CNB: (最新版本l(https://cnb.cool/elsejj/msd-rs/-/releases/latest)

## 获取最新版版本号

请求(HTTP GET)上述的两个最新版地址, 其响应码是 302,307,301,308, 查询响应的 `Location`, 得到重定向的链接, 其中包含了带有最新版本号的链接, 不要直接访问重定向后的链接, 它是供人类查看的复杂WEB页面, 你不需要看它. 从链接中提取版本号, VERSION

例如:
- Github: `location: https://github.com/msd-rs/msd/releases/tag/v0.1.23` 得到版本号 `VERSION=v0.1.23`
- CNB: `location: /elsejj/msd-rs/-/releases/tag/v0.1.23` 得到版本号 `VERSION=v0.1.23`

## 根据TAG构造各个操作系统的下载直链

- Github
  - (Linux x64)[https://github.com/msd-rs/msd/releases/download/${VERSION}/msd-x86_64-unknown-linux-gnu.tar.gz]
  - (MacOS arm64)[https://github.com/msd-rs/msd/releases/download/${VERSION}/msd-aarch64-apple-darwin.tar.gz]
  - (Windows x64)[https://github.com/msd-rs/msd/releases/download/${VERSION}/msd-x86_64-pc-windows-msvc.zip]
- CNB
  - (Linux x64)[https://cnb.cool/elsejj/msd-rs/-/releases/download/${VERSION}/msd-x86_64-unknown-linux-gnu.tar.gz]
  - (MacOS arm64)[https://cnb.cool/elsejj/msd-rs/-/releases/download/${VERSION}/msd-aarch64-apple-darwin.tar.gz]
  - (Windows x64)[https://cnb.cool/elsejj/msd-rs/-/releases/download/${VERSION}/msd-x86_64-pc-windows-msvc.zip]

## 解压

请调用各个操作系统原生的解压缩机制根据下载文件的后缀进行解压缩, 解压后得到 `msd` 或 `msd.exe`

## 升级 

重复上述的过程即可获取最新的 `msd`

# 安装和启动服务

## 安装位置 (MSD_WORKDIR)

- 推荐将 msd 安装到一个用户级的独立数据库目录中, 如 (~/data/msd/), 因为它在运行时会生成数据文件, Windows 建议安装到非系统C盘. 确定后告知用户此目录作为启动目录
- 也可以将msd直接安装到当前工作目录中, 在工作目录下创建 msd 目录, 作为数据库目录

## 启动服务

- 开启日志(可选): 在 ${MSD_WORKDIR}/.env 中写入 `RUST_LOG=msd=info`, 服务启动时读取该文件, 默认日志等级是 warn

- **最重要: `msd server` 是常驻前台进程, 不会自行转后台.**
  直接运行 `./msd --log-dir ./logs server -p msd.pid` 会一直占用当前 shell/会话, 因此写在它后面的命令(探活、建表、导入)一行都不会执行;
  在"一次调用只执行一条命令并等待其结束"的自动化/agent 环境中, 表现为"命令长时间不返回、没有任何输出"。
  这不是启动失败, 而是被前台进程阻塞, **必须后台启动**. 可直接复制:

```bash
cd ${MSD_WORKDIR}
nohup ./msd --log-dir ./logs server -p msd.pid > /tmp/msd.out 2>&1 &
disown
# 有界轮询探活, 不要用固定 sleep 盲等
for i in $(seq 1 30); do
  curl -sf --max-time 1 http://127.0.0.1:50510/health > /dev/null && echo ready && break
  sleep 1
done
```

- 启动后的行为:
  - 日志写入 `--log-dir` 指定的目录(上例为 `./logs`), 每日轮换, 保留最近 7 天, 日志名形如 `msd.log.2026-10-24`
  - **进程的 stdout/stderr 通常没有任何输出**, 上例的 `/tmp/msd.out` 一般是 0 字节. 不要用"重定向文件 + tail"来判断启动是否成功, 那里永远是空的; 版本、数据库路径、监听地址、报错都在 `logs/msd.log.*` 中
  - 进程 ID 写入 `msd.pid`, 便于随后的管理
  - 默认监听 `127.0.0.1:50510`, 添加 `http://` 即客户端所需的 ${MSD_HOST}, 如 `http://127.0.0.1:50510`
  - 就绪判断: `curl -s http://127.0.0.1:50510/health`(返回 200 且内容为 `OK`). 根路径 `/` 返回 404 属正常, 不能用作探活; 也可用 `./msd shell ".tables"` 验证客户端连通
  - 运行 `./msd server --help` 查看选项

- 启动前检查与排障:
  - 先确认是否已在运行: `msd.pid` 存在且 `ps -p $(cat msd.pid)` 有输出, 或 `/health` 可访问
  - 端口被占用、或数据目录被上一个实例锁定(例如上次以前台方式启动后被 kill, 进程未完全退出)时, 服务会启动失败并退出而不是静默复用, 原因见 `logs/msd.log.*`
  - 停止: `kill $(cat msd.pid)`; 重启: 停止后按上面的后台方式重新启动
  - 判断"卡住"的经验规则: 若一条命令很久没有输出也没有报错, 先怀疑**前台常驻进程**. 取消后用 `nohup ... &` + 轮询重试, 不要重复执行原来那条命令


## 启动客户端CLI

- 运行 `./msd shell --help` 查看选项和说明

# 数据同步

以下的各命令均在安装后得到的 ${WORKDIR} 目录中运行

## 建表 (仅首次安装需要)

首次安装并启动服务服务后, 下载[建表SQL](https://cnb.cool/elsejj/msd/data/-/git/raw/main/create.sql), 然后运行
`./msd shell create.sql` 来创建数据表

## 更新数据

### 更新脚本示例

以下的 Bash 脚本是更新数据的示例, 无法运行 Bash 的操作系统可以此编写适当的脚本或直接运行相应的命令
```bash
!/bin/bash

DATE=$(date +"%Y-%m") # eg. 2026-09
BASE_URL=https://cnb.cool/elsejj/msd/data/-/releases/download/$DATE/ 
MSD=./msd

tables=(
"stock_capital_flow"
"stock_shares"
"stock_kline_1d"
"stock_infos"
"stock_financial"
"stock_dividend"
"stock_block"
)

for table in "${tables[@]}"
do
  $MSD shell ".import $BASE_URL$table.tbl.zstd $table"
done
```

### 获取当前数据库数据的最后时间

通过命令 `./msd shell "select ts from stock_kline_1d where obj='SH000001' limit -1"` 查询上证指数的最后日线时间, 即可得到数据最后的时间

### 更新历史总包(仅首次)

在上述的更新脚本示例中, 历史总包的 DATE 为 `2026-09`, 它下面放置了自 2021 年到 2026-10-01 的所有数据, 仅需更新一次.
自 2026-10 月后, 数据按月组织, 在获取了数据最后时候后, 可依次更新历史数据.

### 日常更新

当月的完整数据每日16:30后更新到其所在的目录.
可每日定时更新, 月内停更, 再次更新后会补足相应的数据.
多次更新是允许的, 会以最新的更新修改之前的数据, 所以当数据有误时, 再次更新即可.

### 导入常见问题

- `.import <URL> <table>` 的分工: **客户端(`msd shell`)负责下载 URL**, 再按 `PUT /table/{table}` 把数据流式推给服务端; **服务端负责解压、解析与入库**。`Import successful` 里的 `time_used_ms` / `rows_per_sec` 只是服务端入库耗时, **不含客户端下载时间**(实测 137MB 的 `stock_kline_1d` 总包: 服务端报 1.8~2.1s, 整体墙钟约 2.2s, 其中约 0.4s 是客户端下载; 服务端消耗约 5s CPU, 客户端几乎不耗 CPU)。由此:
  - URL 下载失败(超时、证书、代理)要在**运行 `msd shell` 的机器**上排查网络/代理, 而不是 server 侧
  - 解析或入库失败(表不存在、字段不匹配、磁盘满)要在**运行 server 的机器**上查 `logs/msd.log.*`, 服务端有 `msd::server::handlers::import: updater completed` 之类的记录
  - 大文件导入整体耗时数秒属正常, 不要因为"服务端耗时很短"就以为没有真正写入, 可用 `select ... limit -1` 校验
- 必须先执行建表 SQL 再导入, 否则 `.import` 会因表不存在而失败
- 官方更新脚本里的 `tables` 列表可按需裁剪; 本地导入 `stock_kline_1d`(约 1000 万行)只需数秒, 不必因为体量大而跳过
- 校验导入结果: `./msd shell "select ts from stock_kline_1d where obj='SH000001' limit -1"` 看最后交易日, `./msd shell ".tables"` 看表; 导入过程与错误都记在 `logs/msd.log.*`


# MSD 简介

MSD的设计理念是 **既然SQL无法表达所有的计算, 那么就不要让它做任何计算**, 所以它的 SQL 不是全功能的 SQL, 除了最基本的必要字段的过滤, 它不提供在SQL层面的计算, 而是数据高效的下发给下游的计算引擎, 如 numpy, pandas, polars 等.

- MSD 有两种数据引擎
  - ts: 这是默认的引擎, 数据为时序数据, 必须有`ts` 字段, 常用于各类时序数据, 如股票K线, 财务报表, 分红等. ts 表强制按 obj 分区, 创建表时不需指定此字段, 其名称必为 'obj'.
  - kv: 对于一些无时序性质的数据, 可以使用 kv引擎, 其只有两列, `obj` 是键, 另外的列为值, 可用于如股票码表, 板块列表等数据.
- SQL WHERE 只支持 obj 和 ts 两列的 WHERE 条件, 其中
  - ts
    - 支持 >, >=, =, <=, < 
    - 支持 ts.year, ts.month, ts.day 等时间日期部分匹配, 如 `ts.month=12` 可在财务表中过滤掉季报, 仅获取年报
  - obj
    - 支持 =, IN , 大量 obj 推荐使用 IN, 如 `obj in ("SH000001", "SH0000002")`
    - 支持通配符, 如 `obj = "SH6000*"`, 前缀匹配
    - 支持关联一个kv表项, 如 `obj="@stock_block:上证A股"`, 在一些场景, 可以避免大量的代码传递. 要求该该kv表项的内容为逗号分割的字符串或JSON字符串数组.
- 仅 ts 字段支持 ORDER BY, 默认数据按 ts 升序返回
- SQL LIMIT
  - 支持通常意义的 LIMIT n 格式, 仅最多返回 n 行数据, 此时n为正整数
  - 当n为负数时, 其含义为当前 ts 序的最多最后 n 行数据, 例如, 获取最新的 n 个K线数据, 使用 limit -n 即可, 无需 先降序再 limit n 再升序.
- INSERT 和数据更新
  - INSERT 可用于小批量数据插入, 但一般不推荐
  - 通过 `msd shell` 执行 `.import ` 是更加高效的方法, `.import` 可将一个本地或URL的数据导入到给定表, 导入支持
    - csv 格式: 前两列必须为 obj 和 ts, 其他列由表结构定义
    - tbl 格式: 这是 msd 专用的表格式, 类似 Apache Arrow, 它能几乎 zero memory copy 到 numpy, pandas, polars, arrow
  - 数据更新仅会新增大于数据时间的条目和更新已有的数据条目, 无法插入历史不存在的时间点数据, 此时应该先获取已有数据, 重新组织数据, 然后删除相应的obj, 最后再次插入, 此过程不会涉及到大量的IO, 可酌情使用.
- DELETE 仅支持删除给定 obj 的所有数据, 不支持按时间过滤删除.
- 支持 DROP TABLE
- 可以通过 ".table" "\dt" 列出当前的数据表
- 可以通过 ".schema <表名>", "DESC <表名>" 查看表的结构
- 对于 Python 来说, 推荐使用 `pymsd` 来与 msd 服务进行交交互, 创建 `pymsd.MsdConnect` 并调用其相关的方法, 获取, 聚合数据
- msd 服务以HTTP协议对外提供服务
  - '/query' : 用于执行SQL, 
    - 例如 `curl -X POST http://localhost:50511/query -H "Content-Type: application/json" -d '{"query": "select * from stock_kline_1d where obj=\"SH60000*\" limit -2"}'` 
    - 默认流式返回 NDJSON 格式的若干Table, 在做排查时可以使用. 但此格式对于大量数据效率较低, `pymsd.MsdConnect` 使用二进制协议, 更加高效.
    - 亦可使用 `msd shell` 直接执行SQL, 比使用 `curl` 更加方便
  - '/table/{table_name}': 用于给指定名称的表更新数据, 使用 'PUT' 方法, 发送的内容可以是 CSV或Tbl, 推荐使用 ``pymsd.MsdConnect` 或 `msd shell` 的 `.import` 来调用, 它们基于此API提供了更加易用的使用方法.

