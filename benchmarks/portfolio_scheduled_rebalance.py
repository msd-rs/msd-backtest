"""资金账户类策略：共享现金池 + 定时调仓，用于性能测试。

默认每月首个交易日按 60 日动量截面排名等权调仓；也可切换为每 N 个交易日
定时调仓。使用 PortfolioStrategy + ctx.rebalance() 压测组合账户引擎。

运行示例：
    python -m mbt.run.cli src/bench/portfolio_scheduled_rebalance.py \
        -s SH600000 SZ000001 SH600519 SZ000858 -p -c 1000000 \
        -b 2024-01-01 -e 2025-12-31 -S
    python -m mbt.run.cli src/bench/portfolio_scheduled_rebalance.py \
        -s A_STOCKS -p -c 1000000 -b 2024-01-01 -e 2025-12-31 \
        -a schedule=interval -a rebalance_interval=20 -S
"""

from __future__ import annotations

import numpy as np
import alpha as al
import mbt

M_AUTHOR = "Tencent Hunyuan Hy"


class Strategy(mbt.PortfolioStrategy):
    """定时等权轮动：月度或固定交易日周期，按动量选取 Top K。"""

    portfolio = True
    exec_mode = "next_open"
    warmup_bars = 60

    def __init__(
        self,
        top_k: int = 10,
        schedule: str = "monthly",
        rebalance_interval: int = 20,
        momentum_period: int = 60,
    ):
        super().__init__()
        self.top_k = max(1, int(top_k))
        self.schedule = str(schedule).lower()
        if self.schedule not in ("monthly", "interval"):
            raise ValueError("schedule must be 'monthly' or 'interval'")
        self.rebalance_interval = max(1, int(rebalance_interval))
        self.momentum_period = max(2, int(momentum_period))

    def prepare(self, ctx: mbt.Context):
        """预计算动量因子，初始化调仓调度状态。"""
        with al.Context(
            groups=ctx.n_objs,
            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
            start=0,
            end=ctx.bars,
        ):
            prev_close = al.REF(ctx.close, self.momentum_period)
            with np.errstate(divide="ignore", invalid="ignore"):
                self.momentum = np.where(prev_close > 0.0, ctx.close / prev_close - 1.0, np.nan)

        self.offsets = np.arange(ctx.n_objs, dtype=np.int64) * ctx.bars
        self.tradable = np.array(
            [not obj.startswith(("SH000", "SZ399")) for obj in ctx.objs()],
            dtype=bool,
        )
        self.last_month: str | None = None
        self.last_interval_bar: int | None = None
        self.rebalance_count = 0

    def _is_rebalance_bar(self, ctx: mbt.Context) -> bool:
        """判断是否到达定时调仓日。"""
        if self.schedule == "monthly":
            month = str(ctx.date)[:7]
            if not month or month == "None":
                return False
            if month != self.last_month:
                self.last_month = month
                return True
            return False

        effective_bar = ctx.bar - ctx.warmup_bars
        if effective_bar <= 0:
            return False
        if self.last_interval_bar is None:
            self.last_interval_bar = effective_bar
            return True
        if effective_bar - self.last_interval_bar >= self.rebalance_interval:
            self.last_interval_bar = effective_bar
            return True
        return False

    def execute(self, ctx: mbt.Context):
        """调仓日截面打分并等权再平衡，非调仓日直接返回。"""
        if ctx.is_warmup or not self._is_rebalance_bar(ctx):
            return

        idx = self.offsets + ctx.bar
        score = self.momentum[idx]
        close = ctx.close[idx]
        candidates = np.flatnonzero(self.tradable & np.isfinite(score) & (close > 0.0))
        if candidates.size == 0:
            return

        k = min(self.top_k, candidates.size)
        top = candidates[np.argsort(score[candidates])[::-1][:k]]
        weights = np.zeros(ctx.n_objs, dtype=np.float64)
        weights[top] = 1.0 / k
        ctx.rebalance(weights, tolerance=0.01)
        self.rebalance_count += 1

    def finalize(self, ctx: mbt.Context) -> dict:
        """输出调仓诊断统计。"""
        return {
            "rebalance_count": self.rebalance_count,
            "schedule": self.schedule,
            "rebalance_interval": self.rebalance_interval,
            "top_k": self.top_k,
            "momentum_period": self.momentum_period,
        }
