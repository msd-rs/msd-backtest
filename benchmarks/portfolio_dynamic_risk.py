"""组合重度压测策略：财务预加载 + 季度动态选股换血 + 每日持仓止损止盈 + 50只分散持仓。

对照 portfolio_scheduled_rebalance.py（月度纯触发、持仓仅10只、无财报字段）的轻载组合，
本策略覆盖 mbt 组合引擎的真实核心路径：
1. 声明 `fin_fields = ["f003", "f205"]` 与 `fin_only_year = False`，压测 `load_data()` 财报查询与披露日对齐；
2. 季度切换时在回测内调用 `ctx.run_selector()` + `ctx.set_universe()` 进行无未来函数动态换血；
3. 每个交易日调用 `ctx.holding_shares()` 与 `ctx.holding_rors()` 执行全池持仓止损/移动止盈/趋势离场；
4. 调仓日对筛选白名单按 60 日动量排序，取前 `top_k = 50` 只通过 `ctx.rebalance()` 等权配置。

运行示例：
    uv run python -m mbt.run.cli src/bench/portfolio_dynamic_risk.py \
        -s A_STOCKS -p -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
"""

from __future__ import annotations

import numpy as np
import polars as pl
import alpha as al
import mbt
from mbt.select import A_STOCKS_EXCLUDE_ST

M_AUTHOR = "Gemini 4 Hydrogen"


class DynamicFundamentalSelector(mbt.Selector):
    """内嵌基本面估值选股器：供回测期 `ctx.run_selector()` 按季度无前视调用。"""

    fin_fields: list[str] = ["f003", "f205"]
    fin_only_year: bool = False

    def __init__(
        self,
        min_roe: float = 6.0,
        max_pb: float = 8.0,
        candidate_limit: int = 300,
    ):
        super().__init__()
        self.min_roe = float(min_roe)
        self.max_pb = float(max_pb)
        self.candidate_limit = max(10, int(candidate_limit))
        self.factors = pl.DataFrame()

    def step00(self, stocks: list[str]) -> list[str]:
        if not stocks and not self.is_backtest:
            stocks = A_STOCKS_EXCLUDE_ST
        return [s for s in stocks if not s.startswith(("SH000", "SZ399"))]

    def step01(self, stocks: list[str]) -> list[str]:
        """读取无未来函数的财报快照，筛出 ROE 达标且每股净资产为正的标的。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        fin = self.dp.load_financial(
            stocks,
            fields=["f205", "f003"],
            only_year=False,
            lastN=1,
        )
        if isinstance(fin, dict):
            parts = []
            for sym, frame in fin.items():
                if frame is not None and not frame.is_empty():
                    parts.append(frame.with_columns(pl.lit(sym).alias("obj")))
            fin = pl.concat(parts) if parts else pl.DataFrame()

        if fin.is_empty() or not {"f205", "f003"}.issubset(fin.columns):
            self.factors = pl.DataFrame()
            return []

        picked = (
            fin.filter(pl.col("obj").is_in(stocks))
            .group_by("obj", maintain_order=True)
            .last()
            .filter(
                pl.col("f205").is_finite()
                & (pl.col("f205") >= self.min_roe)
                & pl.col("f003").is_finite()
                & (pl.col("f003") > 0.0)
            )
            .select([
                pl.col("obj"),
                pl.col("f205").alias("roe"),
                pl.col("f003").alias("bps"),
            ])
        )
        self.factors = picked
        return picked["obj"].to_list()

    def step02(self, stocks: list[str]) -> list[str]:
        """以当前 bar 未复权价格计算 PB，按 ROE 从高到低截断候选白名单。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        snap = self.dp.load_snapshot(stocks, fin_fields=[])
        if snap.is_empty():
            self.factors = pl.DataFrame()
            return []

        price_col = "raw_close" if "raw_close" in snap.columns else "close"
        merged = self.factors.join(
            snap.select([pl.col("obj"), pl.col(price_col).alias("price")]),
            on="obj",
            how="inner",
        )
        picked = (
            merged.with_columns((pl.col("price") / pl.col("bps")).alias("pb"))
            .filter(
                pl.col("pb").is_finite()
                & (pl.col("pb") > 0.0)
                & (pl.col("pb") <= self.max_pb)
            )
            .sort("roe", descending=True)
            .head(self.candidate_limit)
        )
        self.factors = picked
        return picked["obj"].to_list()


class Strategy(mbt.PortfolioStrategy):
    """季度动态选股换血 + 每日截面持仓风控 + 50 只等权再平衡组合策略。"""

    portfolio: bool = True
    exec_mode: str = "next_open"
    warmup_bars: int = 60
    fin_fields: list[str] = ["f003", "f205"]
    fin_only_year: bool = False

    def __init__(
        self,
        top_k: int = 50,
        ma_period: int = 20,
        momentum_period: int = 60,
        stop_loss: float = 0.08,
        trailing_trigger: float = 0.12,
        trailing_stop: float = 0.06,
        min_roe: float = 6.0,
        max_pb: float = 8.0,
        candidate_limit: int = 300,
    ):
        super().__init__()
        self.top_k = max(1, int(top_k))
        self.ma_period = max(2, int(ma_period))
        self.momentum_period = max(2, int(momentum_period))
        self.stop_loss = max(0.001, float(stop_loss))
        self.trailing_trigger = max(0.001, float(trailing_trigger))
        self.trailing_stop = max(0.001, float(trailing_stop))
        self.min_roe = float(min_roe)
        self.max_pb = float(max_pb)
        self.candidate_limit = max(self.top_k, int(candidate_limit))

        self.selector = DynamicFundamentalSelector(
            min_roe=self.min_roe,
            max_pb=self.max_pb,
            candidate_limit=self.candidate_limit,
        )

    def prepare(self, ctx: mbt.Context):
        """预计算均线趋势与动量得分，初始化持仓跟踪状态。"""
        with al.Context(
            groups=ctx.n_objs,
            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
            start=0,
            end=ctx.bars,
        ):
            ma = al.MA(ctx.close, self.ma_period)
            prev_close = al.REF(ctx.close, self.momentum_period)
            self.trend_ok = np.asarray(ctx.close > ma, dtype=bool)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.momentum = np.where(prev_close > 0.0, ctx.close / prev_close - 1.0, np.nan)

        self.offsets = np.arange(ctx.n_objs, dtype=np.int64) * ctx.bars
        self.tradable = np.array(
            [not obj.startswith(("SH000", "SZ399")) for obj in ctx.objs()],
            dtype=bool,
        )
        self.obj_to_idx = {obj: i for i, obj in enumerate(ctx.objs())}
        self.universe_mask = self.tradable.copy()
        self.peak_rors = np.full(ctx.n_objs, -np.inf, dtype=np.float64)
        self.pending_sell = np.zeros(ctx.n_objs, dtype=bool)
        self.last_quarter: tuple[int, int] | None = None
        self.rebalance_count = 0
        self.selector_runs = 0
        self.exit_counts = {
            "stop_loss": 0,
            "trailing_stop": 0,
            "trend_exit": 0,
        }

    def execute(self, ctx: mbt.Context):
        """每日查询持仓盈亏执行风控，季度初运行动态选股并重平衡。"""
        if ctx.is_warmup:
            return

        date_str = ctx.date
        need_rebalance = False
        if date_str and len(date_str) >= 7:
            year = int(date_str[:4])
            quarter = (int(date_str[5:7]) - 1) // 3
            cur_q = (year, quarter)
            if cur_q != self.last_quarter:
                self.last_quarter = cur_q
                need_rebalance = True

        if need_rebalance:
            picked = ctx.run_selector(self.selector, lastN=1)
            self.selector_runs += 1
            if picked:
                ctx.set_universe(picked)
                mask = np.zeros(ctx.n_objs, dtype=bool)
                for sym in picked:
                    g = self.obj_to_idx.get(sym)
                    if g is not None:
                        mask[g] = True
                self.universe_mask = mask & self.tradable

        shares = ctx.holding_shares()
        rors = np.nan_to_num(ctx.holding_rors(), nan=0.0)
        held = shares > 0.0

        self.pending_sell &= held
        self.peak_rors[~held] = -np.inf
        self.peak_rors[held] = np.maximum(self.peak_rors[held], rors[held])

        idx = self.offsets + ctx.bar
        manageable = held & ~self.pending_sell
        stop = manageable & (rors <= -self.stop_loss)
        trail = (
            manageable
            & ~stop
            & (self.peak_rors >= self.trailing_trigger)
            & ((self.peak_rors - rors) >= self.trailing_stop)
        )
        trend_exit = manageable & ~stop & ~trail & ~self.trend_ok[idx]
        sell = stop | trail | trend_exit

        if sell.any():
            ctx.sell(sell)
            self.pending_sell[sell] = True
            self.exit_counts["stop_loss"] += int(stop.sum())
            self.exit_counts["trailing_stop"] += int(trail.sum())
            self.exit_counts["trend_exit"] += int(trend_exit.sum())

        if need_rebalance:
            score = self.momentum[idx]
            close = ctx.close[idx]
            eligible = (
                self.universe_mask
                & self.trend_ok[idx]
                & np.isfinite(score)
                & (close > 0.0)
                & ~self.pending_sell
            )
            candidates = np.flatnonzero(eligible)
            if candidates.size > 0:
                k = min(self.top_k, candidates.size)
                top = candidates[np.argsort(score[candidates])[::-1][:k]]
                weights = np.zeros(ctx.n_objs, dtype=np.float64)
                weights[top] = 1.0 / k
                ctx.rebalance(weights, tolerance=0.01)
                self.rebalance_count += 1

    def finalize(self, ctx: mbt.Context) -> dict:
        """输出动态换血与风控出场诊断统计。"""
        return {
            "selector_runs": self.selector_runs,
            "rebalance_count": self.rebalance_count,
            "exit_stop_loss": self.exit_counts["stop_loss"],
            "exit_trailing_stop": self.exit_counts["trailing_stop"],
            "exit_trend_exit": self.exit_counts["trend_exit"],
            "top_k": self.top_k,
        }
