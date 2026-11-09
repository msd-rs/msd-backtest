"""第 03 轮：增强大盘状态判断、抑制换手与仓位连续化。

第 02 轮已把收益从 -66.74% 提升到 +41.85%（调参区间），但 2023-2024 弱势年份
仍为 -8.53%、回撤 -34.23%，手续费仍占初始资金约 10%。report.02.md 指出主要
短板是「单条 MA20 开关在震荡下行市里反复清仓/重建仓」和「排名抖动导致高换手」。
本轮在第 02 轮基础上增加：
  1. 大盘状态三重确认：市场宽度（池内站上MA20比例）、MA20斜率、连续N根确认；
  2. 弱势时按比例减仓而非一律清仓（exit_fraction）；
  3. 打分周度平滑（score_smooth）与最短持有期（min_hold_bars）抑制换手；
所有新参数默认值都等价于第 02 轮，便于逐项验证增量效果。
"""

from __future__ import annotations

import numpy as np
import polars as pl
import alpha as al
import mbt
from mbt.select import A_STOCKS_EXCLUDE_ST

CAP_LOW = 3e9
CAP_HIGH = 1e10
GENE_WINDOW = 120
RECENT_GENE_WINDOW = 60
HOT_WINDOW = 20
HIGH_WINDOW = 60


def _pct_rank(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把某个因子在合格股票截面内映射为 0~1 分位，值越大排名越靠前。"""
    out = np.zeros(len(values), dtype=np.float64)
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return out
    if len(idx) == 1:
        out[idx[0]] = 1.0
        return out
    v = np.nan_to_num(np.asarray(values[idx], dtype=np.float64), nan=0.0)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(idx), dtype=np.float64)
    ranks[order] = np.arange(len(idx), dtype=np.float64) / (len(idx) - 1)
    out[idx] = ranks
    return out


class Selector(mbt.Selector):
    """静态粗筛：非ST中盘股 + 上市够久 + 涨停基因 + 成交活跃。"""

    def __init__(
        self,
        cap_low: float = CAP_LOW,
        cap_high: float = CAP_HIGH,
        pre_cap_low: float = 2e9,
        pre_cap_high: float = 1.5e10,
        min_listed_bars: int = GENE_WINDOW,
        min_limit_up: int = 2,
        min_recent_limit_up: int = 1,
        min_amount: float = 1e8,
        min_turnover: float = 0.03,
        max_pool: int = 800,
    ):
        super().__init__()
        self.pre_cap_low = pre_cap_low if pre_cap_low > 0 else cap_low * 2 / 3
        self.pre_cap_high = pre_cap_high if pre_cap_high > 0 else cap_high * 1.5
        self.min_listed_bars = min_listed_bars
        self.min_limit_up = min_limit_up
        self.min_recent_limit_up = min_recent_limit_up
        self.min_amount = min_amount
        self.min_turnover = min_turnover
        self.max_pool = max_pool
        self.factors = pl.DataFrame()

    def step00(self, stocks: list[str]) -> list[str]:
        """用快照筛出非ST、未停牌、市值处于放宽后中盘带的股票，并记录总市值。"""
        objs = sorted(set(stocks)) if stocks else list(A_STOCKS_EXCLUDE_ST)
        objs = [s for s in objs if not s.startswith("BJ")]
        snap = self.dp.load_snapshot(objs)
        if len(snap) == 0:
            self.factors = pl.DataFrame(schema={"obj": pl.String, "market_cap": pl.Float64})
            return []
        snap = snap.filter(
            (pl.col("close") > 0)
            & (pl.col("volume") > 0)
            & (pl.col("total_shares") > 0)
            & (pl.col("tradable_shares") > 0)
        ).with_columns((pl.col("close") * pl.col("total_shares")).alias("market_cap"))
        snap = snap.filter(
            (pl.col("market_cap") >= self.pre_cap_low)
            & (pl.col("market_cap") <= self.pre_cap_high)
        )
        self.factors = snap.select("obj", "close", "market_cap", "amount")
        return snap["obj"].to_list()

    def step01(self, stocks: list[str]) -> list[str]:
        """用历史K线确认上市时长、涨停基因和成交热度，按20日成交额留最活跃的max_pool只。"""
        if not stocks:
            return []
        window = max(self.min_listed_bars, GENE_WINDOW, HOT_WINDOW)
        panel = self.dp.load_kline(stocks, lastN=window, baseline="SH000001")
        window = min(window, panel.bars)
        gene_window = min(GENE_WINDOW, window)
        recent_gene_window = min(RECENT_GENE_WINDOW, window)
        hot_window = min(HOT_WINDOW, window)

        close = panel.column("close")
        valid_bars = np.count_nonzero(close > 0, axis=1)
        prev_close = np.roll(close, 1, axis=1)
        prev_close[:, 0] = np.nan
        real_bar = (close > 0) & (prev_close > 0)
        limit_up = ((panel.column("limited") == 1) & real_bar).astype(np.float64).ravel()

        volume = panel.buffer("volume")
        tradable = panel.buffer("tradable_shares")
        turnover = np.divide(
            volume, tradable, out=np.full_like(volume, np.nan),
            where=np.isfinite(tradable) & (tradable > 0),
        )
        lu120 = panel.compute(al.SUM, limit_up, gene_window)[:, -1]
        lu60 = panel.compute(al.SUM, limit_up, recent_gene_window)[:, -1]
        amount20 = panel.compute(al.MA, "amount", hot_window)[:, -1]
        turn20 = panel.compute(al.MA, turnover, hot_window)[:, -1]

        df = panel.cross_section(
            valid_bars=valid_bars, lu120=lu120, lu60=lu60,
            amount20=amount20, turn20=turn20,
        ).join(self.factors.select("obj", "market_cap"), on="obj", how="left")
        df = df.filter(
            (pl.col("valid_bars") >= int(window * 0.9))
            & (pl.col("lu120") >= self.min_limit_up)
            & (pl.col("lu60") >= self.min_recent_limit_up)
            & (pl.col("amount20") >= self.min_amount)
            & (pl.col("turn20") >= self.min_turnover)
        ).sort("amount20", descending=True).head(self.max_pool)
        self.factors = df
        return df["obj"].to_list()


class Strategy(mbt.PortfolioStrategy):
    """周度轮动：前5买入、跌出前10卖出，打分改为均值回归方向并加风控。"""

    fin_fields = ["f001"]

    def __init__(
        self,
        cap_low: float = CAP_LOW,
        cap_high: float = CAP_HIGH,
        top_k: int = 5,
        hold_buffer: int = 10,
        min_limit_up: int = 2,
        min_recent_limit_up: int = 1,
        min_amount: float = 1e8,
        min_turnover: float = 0.03,
        min_drop: float = 0.10,
        max_drop: float = 0.70,
        min_rebound: float = -0.08,
        max_rebound: float = 0.20,
        require_up_day: int = 0,
        w_depth: float = 0.30,
        w_rebound: float = 0.0,
        w_mom20: float = -0.15,
        w_hot: float = -0.05,
        w_gene: float = -0.05,
        market_filter: int = 1,
        market_ma: int = 20,
        market_exit: int = 1,
        stickiness: float = 0.03,
        stop_loss: float = 0.08,
        breadth_min: float = 0.30,
        market_slope: int = 0,
        market_confirm: int = 2,
        exit_on_base: int = 1,
        exit_fraction: float = 1.0,
        min_hold_bars: int = 0,
        score_smooth: float = 0.0,
        warmup: int = GENE_WINDOW,
    ):
        super().__init__()
        self.cap_low = cap_low
        self.cap_high = cap_high
        self.top_k = top_k
        self.hold_buffer = hold_buffer
        self.min_limit_up = min_limit_up
        self.min_recent_limit_up = min_recent_limit_up
        self.min_amount = min_amount
        self.min_turnover = min_turnover
        self.min_drop = min_drop
        self.max_drop = max_drop
        self.min_rebound = min_rebound
        self.max_rebound = max_rebound
        self.require_up_day = int(require_up_day)
        self.w_depth = w_depth
        self.w_rebound = w_rebound
        self.w_mom20 = w_mom20
        self.w_hot = w_hot
        self.w_gene = w_gene
        self.market_filter = int(market_filter)
        self.market_ma = market_ma
        self.market_exit = int(market_exit)
        self.stickiness = stickiness
        self.stop_loss = stop_loss
        self.breadth_min = breadth_min
        self.market_slope = int(market_slope)
        self.market_confirm = max(1, int(market_confirm))
        self.exit_on_base = int(exit_on_base)
        self.exit_fraction = min(1.0, max(0.0, float(exit_fraction)))
        self.min_hold_bars = max(0, int(min_hold_bars))
        self.score_smooth = min(0.95, max(0.0, float(score_smooth)))
        self.warmup = warmup
        self._rebalance_bars: np.ndarray | None = None
        self._tradable_start = 0
        self._entry: np.ndarray | None = None
        self._hold_bar: np.ndarray | None = None
        self._prev_score: np.ndarray | None = None

    def prepare(self, ctx: mbt.Context):
        """预计算全历史因子，并标注每周首个交易日与大盘状态。"""
        panel = ctx.panel()
        self._tradable_start = int(getattr(getattr(ctx, "account", None), "tradable_start", 0))
        n = panel.n_objs

        volume = panel.buffer("volume")
        tradable = panel.buffer("tradable_shares")
        turnover = np.divide(
            volume, tradable, out=np.full_like(volume, np.nan),
            where=np.isfinite(tradable) & (tradable > 0),
        )
        self.close = panel.column("close")
        self.price = panel.column("price")
        self.shares = panel.column("total_shares")
        self.volume = panel.column("volume")
        self.limited = panel.column("limited")
        self._entry = np.full(n, np.nan)
        self._hold_bar = np.full(n, -1)
        self._prev_score = np.full(n, np.nan)

        prev_close = np.roll(self.close, 1, axis=1)
        prev_close[:, 0] = np.nan
        real_bar = (self.close > 0) & (prev_close > 0)
        limit_up = ((self.limited == 1) & real_bar).astype(np.float64).ravel()

        self.hh = panel.compute(al.HHV, "high", HIGH_WINDOW)
        self.ma5 = panel.compute(al.MA, "close", 5)
        ma20 = panel.compute(al.MA, "close", 20)
        self.ma60 = panel.compute(al.MA, "close", 60)
        self.v5 = panel.compute(al.MA, "volume", 5)
        self.v20 = panel.compute(al.MA, "volume", 20)
        self.ref_close1 = panel.compute(al.REF, "close", 1)
        self.ref_close5 = panel.compute(al.REF, "close", 5)
        self.ref_close20 = panel.compute(al.REF, "close", 20)
        self.amount20 = panel.compute(al.MA, "amount", HOT_WINDOW)
        self.turn20 = panel.compute(al.MA, turnover, HOT_WINDOW)
        self.lu120 = panel.compute(al.SUM, limit_up, GENE_WINDOW)
        self.lu60 = panel.compute(al.SUM, limit_up, RECENT_GENE_WINDOW)

        with np.errstate(divide="ignore", invalid="ignore"):
            self.depth = 1.0 - self.close / self.hh
            self.rebound = self.close / self.ref_close5 - 1.0
            self.mom20 = self.close / self.ref_close20 - 1.0
            self.volratio = self.v5 / self.v20 - 1.0
            self.pos60 = self.close / self.ma60 - 1.0

        # 大盘状态：指数在均线之上 + 市场宽度 + 均线斜率 + 连续确认
        bench_ma = panel.compute(al.MA, "close", self.market_ma)[0]
        above = self.close[0] >= bench_ma
        base_ok = above.copy()
        if self.breadth_min > 0:
            breadth = np.mean(
                (self.close[self._tradable_start:] > ma20[self._tradable_start:])
                & np.isfinite(ma20[self._tradable_start:]),
                axis=0,
            )
            base_ok &= breadth >= self.breadth_min
        if self.market_slope:
            slope_ok = bench_ma >= np.roll(bench_ma, 5)
            slope_ok[:5] = True
            base_ok &= slope_ok

        streak = np.zeros(panel.bars, dtype=np.int64)
        for k in range(1, panel.bars):
            streak[k] = streak[k - 1] + 1 if base_ok[k] else 0
        streak[0] = int(base_ok[0])
        self.market_ok = streak >= self.market_confirm
        self.market_bad = ~base_ok if self.exit_on_base else ~self.market_ok

        days = panel.timeline().astype("datetime64[D]")
        week = (
            (days - np.datetime64("1970-01-05", "D"))
            .astype("timedelta64[D]").astype(np.int64) // 7
        )
        rebalance = np.zeros(panel.bars, dtype=bool)
        rebalance[1:] = week[1:] != week[:-1]
        self._rebalance_bars = rebalance

    def _eligible(self, i: int) -> np.ndarray:
        """当日合格股票：中盘、有涨停基因、活跃、超跌但尚未大幅反弹、可买入。"""
        n = self.close.shape[0]
        ok = np.zeros(n, dtype=bool)
        ok[self._tradable_start:] = True

        close = self.close[:, i]
        price = self.price[:, i]
        cap = price * self.shares[:, i]
        depth = self.depth[:, i]
        rebound = self.rebound[:, i]
        amount = self.amount20[:, i]
        turn = self.turn20[:, i]

        ok &= (
            np.isfinite(close) & np.isfinite(price) & (price > 0)
            & np.isfinite(self.shares[:, i]) & np.isfinite(self.volume[:, i])
            & (self.volume[:, i] > 0)
        )
        ok &= (cap >= self.cap_low) & (cap <= self.cap_high)
        ok &= (
            (self.lu120[:, i] >= self.min_limit_up)
            & (self.lu60[:, i] >= self.min_recent_limit_up)
            & (amount >= self.min_amount)
            & (turn >= self.min_turnover)
            & np.isfinite(depth) & np.isfinite(rebound)
        )
        ok &= (
            (depth >= self.min_drop) & (depth <= self.max_drop)
            & (rebound >= self.min_rebound) & (rebound <= self.max_rebound)
        )
        if self.require_up_day:
            ok &= close >= self.ref_close1[:, i]
        ok &= self.limited[:, i] != 1
        return ok

    def execute(self, ctx: mbt.Context):
        """每周首个交易日：均值回归打分，前5补买、跌出前10卖出，附加大盘与止损。"""
        i = ctx.bar
        if i < self.warmup or self._rebalance_bars is None or not self._rebalance_bars[i]:
            return

        n = ctx.n_objs
        ok = self._eligible(i)
        market_ok = (self.market_filter == 0) or bool(self.market_ok[i])
        market_bad = bool(self.market_bad[i])

        score = (
            self.w_depth * _pct_rank(self.depth[:, i], ok)
            + self.w_rebound * _pct_rank(self.rebound[:, i], ok)
            + self.w_mom20 * _pct_rank(self.mom20[:, i], ok)
            + self.w_hot * 0.5 * (
                _pct_rank(self.amount20[:, i], ok) + _pct_rank(self.turn20[:, i], ok)
            )
            + self.w_gene * _pct_rank(self.lu120[:, i], ok)
        )

        acc = getattr(ctx, "account", None)
        held = set()
        if acc is not None:
            held = {
                g for g in range(self._tradable_start, n)
                if acc.positions[g].shares > 0
            }

        # 记录持仓起始bar，用于最短持有期
        if self._hold_bar is not None:
            for g in held:
                if self._hold_bar[g] < 0:
                    self._hold_bar[g] = i
            for g in range(self._tradable_start, n):
                if g not in held:
                    self._hold_bar[g] = -1

        # 打分周度平滑，减少排名抖动
        if self.score_smooth > 0 and self._prev_score is not None:
            a = self.score_smooth
            prev = self._prev_score
            score = np.where(np.isfinite(prev), (1.0 - a) * score + a * prev, score)

        # 持仓粘性：小幅抬高已持有股票的排名，降低无意义换手
        if self.stickiness:
            for g in held:
                if ok[g]:
                    score[g] += self.stickiness

        score[~ok] = -np.inf
        if self._prev_score is not None:
            self._prev_score = np.where(np.isfinite(score), score, np.nan)
        order = [
            g for g in np.argsort(-score, kind="stable")
            if np.isfinite(score[g])
        ][: max(self.top_k, self.hold_buffer)]
        top5 = order[: self.top_k]
        top10 = set(order[: self.hold_buffer])

        sell_full = np.zeros(n, dtype=bool)
        for g in held:
            if g not in top10:
                if (
                    self.min_hold_bars
                    and self._hold_bar is not None
                    and i - self._hold_bar[g] < self.min_hold_bars
                ):
                    continue
                sell_full[g] = True

        # 入场价止损（用前复权收盘价，自动兼容除权除息）
        if self.stop_loss > 0 and self._entry is not None:
            for g in held:
                e = self._entry[g]
                if np.isfinite(e) and e > 0:
                    if self.close[g, i] / e - 1.0 <= -self.stop_loss:
                        sell_full[g] = True

        # 大盘走弱：按比例减仓而非一律清仓
        sell_partial = np.zeros(n, dtype=bool)
        if self.market_exit and market_bad:
            for g in held:
                if not sell_full[g]:
                    sell_partial[g] = True

        buy = np.zeros(n, dtype=bool)
        if market_ok:
            room = self.hold_buffer - (len(held) - int(sell_full.sum()))
            for g in top5:
                if room <= 0:
                    break
                if g not in held:
                    buy[g] = True
                    room -= 1

        if self._entry is not None and buy.any():
            for g in np.flatnonzero(buy):
                self._entry[g] = self.close[g, i]
                if self._hold_bar is not None:
                    self._hold_bar[g] = i
        if self._entry is not None:
            for g in range(n):
                if g not in held and not buy[g]:
                    self._entry[g] = np.nan

        if sell_full.any():
            ctx.sell(sell_full, 1.0)
        if sell_partial.any() and self.exit_fraction > 0:
            ctx.sell(sell_partial, self.exit_fraction)
        if buy.any():
            ctx.buy(buy, 1.0)
