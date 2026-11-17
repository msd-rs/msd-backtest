"""小而美主板小市值轮动策略（打地鼠）。

选股：沪深主板、非 ST/*ST/退市、上市满一年、昨日未涨停、总市值 5 亿~50 亿，
      只留总市值最小的 5 只。
交易：每周二换仓到这 5 只；持仓跌 7% 止损；基准连跌 5% 清仓；
      昨日涨停的持仓若今日封板继续持有，开板即卖出。

说明：msd-backtest 以日线驱动，策略中的“周二上午 10 点”与“下午 2 点半”
      分别用周二日线 bar 和当日收盘涨停状态近似（分钟线数据当前不可用）。
      建议回测时加 --warmup-bars 250，以便准确判断“上市满一年”。

用法：
    python -m mbt.run.cli strategy/019/small_cap_rotation.py \
        -s A_STOCKS_EXCLUDE_ST SH000001 -B SH000001 \
        --warmup-bars 250 -b 2025-01-01 -e 2025-12-31 -c 1000000 -S
"""

from __future__ import annotations

import datetime
import logging

import numpy as np
import polars as pl

import mbt
from mbt.select import stocks as stock_meta

M_AUTHOR = "Tencent Hunyuan Hy"
logger = logging.getLogger("strategy.019")

MAINBOARD_PREFIXES = (
    "SH600", "SH601", "SH603", "SH605",
    "SZ000", "SZ001", "SZ002", "SZ003",
)
LISTING_LOOKBACK_BARS = 250
MIN_MARKET_CAP = 5e8
MAX_MARKET_CAP = 5e10


def is_mainboard(obj: str) -> bool:
    """仅保留沪深主板代码，排除科创板、创业板与北交所。"""
    return obj.startswith(MAINBOARD_PREFIXES)


class Selector(mbt.Selector):
    """筛选“小而美”的干净主板票，并按总市值升序输出目标池。"""

    def __init__(
        self,
        top_k: int = 5,
        min_market_cap: float = MIN_MARKET_CAP,
        max_market_cap: float = MAX_MARKET_CAP,
        benchmark: str = "SH000001",
    ):
        super().__init__()
        self.top_k = int(top_k)
        self.min_market_cap = float(min_market_cap)
        self.max_market_cap = float(max_market_cap)
        self.benchmark = benchmark
        self.factors = pl.DataFrame()

    def step00(self, stocks: list[str]) -> list[str]:
        """只留主板股票，剔除 ST、*ST、退市股及非股票标的。"""
        universe = list(stocks) if stocks else list(stock_meta.A_STOCKS_EXCLUDE_ST)
        if not universe:
            return []

        info = stock_meta.ALL_STOCKS.select("obj", "name", "kind", "status")
        df = pl.DataFrame({"obj": universe}, schema={"obj": pl.String}).join(
            info, on="obj", how="inner"
        )
        df = df.filter(
            (pl.col("kind") == 1)
            & (pl.col("status") == 0)
            & ~pl.col("name").str.contains("ST", literal=True)
            & ~pl.col("name").str.contains("退", literal=True)
        )
        return [obj for obj in df["obj"].to_list() if is_mainboard(obj)]

    def step01(self, stocks: list[str]) -> list[str]:
        """用未复权价乘总股本计算总市值，只保留 5 亿~50 亿。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        snap = self.dp.load_snapshot(stocks)
        if "raw_close" not in snap.columns or "total_shares" not in snap.columns:
            raise ValueError("snapshot must contain raw_close and total_shares")

        df = snap.select(
            pl.col("obj"),
            (pl.col("raw_close") * pl.col("total_shares")).alias("market_cap"),
        ).filter(
            pl.col("market_cap").is_not_null()
            & pl.col("market_cap").is_finite()
            & (pl.col("market_cap") >= self.min_market_cap)
            & (pl.col("market_cap") <= self.max_market_cap)
        )
        self.factors = df
        return df["obj"].to_list()

    def step02(self, stocks: list[str]) -> list[str]:
        """剔除上市不满一年及昨日涨停的票，按市值升序输出。"""
        if not stocks or len(self.factors) == 0:
            return []

        panel = self.dp.load_kline(
            stocks, lastN=LISTING_LOOKBACK_BARS, baseline="SH000001"
        )
        if panel.n_objs == 0 or panel.bars == 0:
            return []

        close = panel.column("close")
        limited = panel.column("limited")
        if panel.bars >= LISTING_LOOKBACK_BARS:
            listed_ok = ~np.isnan(close[:, 0])
        else:
            # 历史不足一年窗口时无法判断上市时间，交由调用方用 warmup-bars 补足。
            listed_ok = np.ones(panel.n_objs, dtype=bool)

        prev_limited = (
            limited[:, -2] if panel.bars >= 2 else np.zeros(panel.n_objs, dtype=np.float64)
        )
        flags = panel.cross_section(
            listed_ok=listed_ok, prev_limited=prev_limited
        )
        self.factors = (
            self.factors.join(flags, on="obj", how="inner")
            .filter(pl.col("listed_ok") & (pl.col("prev_limited") != 1))
            .sort("market_cap")
        )
        objs = self.factors["obj"].to_list()
        if self.is_backtest:
            return objs[: self.top_k]

        # 独立运行：返回完整候选池供 CLI 预加载，并带上熔断所需的基准指数。
        known = set(stock_meta.ALL_STOCKS["obj"].to_list())
        if self.benchmark in known and self.benchmark not in objs:
            objs.append(self.benchmark)
        return objs


def check_limit_up(ctx: mbt.PortfolioContext, obj: str) -> int:
    """检查昨日涨停的持仓今日是否封板。

    Returns:
        1: 昨日涨停且今日仍封板（死拿）；-1: 涨停板打开（卖出）；0: 不适用。
    """
    bar = ctx.bar
    if bar < 1:
        return 0
    group = ctx.index_of(obj)
    limited = ctx.data("limited")
    bars = ctx.bars
    if limited[group * bars + bar - 1] != 1:
        return 0
    return 1 if limited[group * bars + bar] == 1 else -1


class Strategy(mbt.PortfolioStrategy):
    """周二调仓 + 每日止损/熔断/涨停板监控的组合策略。"""

    def __init__(
        self,
        selector: Selector | None = None,
        stoploss_limit: float = 0.07,
        crash_limit: float = 0.05,
        top_k: int = 5,
        benchmark: str = "SH000001",
        selector_lastN: int = 300,
    ):
        super().__init__()
        self.selector = (
            selector if selector is not None else Selector(top_k=top_k, benchmark=benchmark)
        )
        self.stoploss_limit = float(stoploss_limit)
        self.crash_limit = float(crash_limit)
        self.benchmark = benchmark
        self.selector_lastN = int(selector_lastN)
        self._benchmark_idx: int | None = None
        self.exit_reasons = {
            "stop_loss": 0,
            "limit_up_break": 0,
            "market_crash": 0,
        }

    def prepare(self, ctx: mbt.PortfolioContext):
        """定位基准指数；未预加载基准时禁用熔断。"""
        idx = getattr(ctx.account, "benchmark_idx", None)
        if idx is None and self.benchmark in ctx.objs():
            idx = ctx.index_of(self.benchmark)
        self._benchmark_idx = idx
        if idx is None:
            logger.warning("benchmark %s not in pool, crash guard disabled", self.benchmark)

    def _market_crashed(self, ctx: mbt.PortfolioContext) -> bool:
        """基准连续下跌累计达到 crash_limit 时触发熔断。"""
        idx = self._benchmark_idx
        bar = ctx.bar
        if idx is None or bar < 1:
            return False
        close = ctx.data("close")
        bars = ctx.bars
        base = idx * bars
        series = close[base : base + bars]
        if np.isnan(series[bar]) or np.isnan(series[bar - 1]):
            return False

        start = bar
        while start > 0 and not np.isnan(series[start - 1]) and series[start] < series[start - 1]:
            start -= 1
        if start == bar:
            return False
        return bool(series[bar] / series[start] - 1.0 <= -self.crash_limit)

    def execute(self, ctx: mbt.PortfolioContext):
        """每日风控，周二按最新市值换仓到最小的 5 只。"""
        if ctx.is_warmup or ctx.bar < 1:
            return

        held = list(ctx.account.held_objs)

        if self._market_crashed(ctx):
            for obj in held:
                ctx.sell(obj)
                self.exit_reasons["market_crash"] += 1
            return

        to_sell: list[str] = []
        still_limit_up: set[str] = set()
        for obj in held:
            limit_state = check_limit_up(ctx, obj)
            if limit_state == -1:
                to_sell.append(obj)
                self.exit_reasons["limit_up_break"] += 1
            elif limit_state == 1:
                still_limit_up.add(obj)
            elif ctx.holding_ror(obj) <= -self.stoploss_limit:
                to_sell.append(obj)
                self.exit_reasons["stop_loss"] += 1
        for obj in to_sell:
            ctx.sell(obj)

        if datetime.date.fromisoformat(ctx.date).weekday() != 1:
            return

        pool = ctx.run_selector(self.selector, lastN=self.selector_lastN)
        ctx.set_universe(pool)
        if not pool:
            ctx.rebalance_objs({})
            return

        weight = 1.0 / len(pool)
        target = {obj: weight for obj in pool}
        current_weights = ctx.account.weights()
        for obj in still_limit_up:
            if obj not in target:
                # 涨停信仰：周二换仓也不动仍封板的持仓。
                target[obj] = float(current_weights[ctx.index_of(obj)])
        ctx.rebalance_objs(target)

    def finalize(self, ctx: mbt.PortfolioContext) -> dict:
        """输出卖出原因统计。"""
        return {"exit_reasons": dict(self.exit_reasons)}
