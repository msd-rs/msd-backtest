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

- 开启日志: 在 ${MSD_WORKDIR}/.env 中, 添加 `RUST_LOG=msd=info` 以开启info日志, 默认的日志等级是 warn
- 启动服务: `./msd --log-dir ./logs server -p msd.pid`, 这将
  - 将日志写入  ${MSD_WORKDIR}/.logs 目录中, 每日轮换, 保持最近的7天, 日志名形如 `msd.log.2026-10-24`
  - 将进程ID 写入 `msd.pid` 中, 便于随后的管理
  - 启动后, 默认监听 '127.0.0.1:50510' 作为服务地址, 添加 `http://` 即客户端所需的 ${MSD_HOST}, 如 `http://127.0.0.1:50510`
  - 运行 `./msd server --help` 查看选项

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





