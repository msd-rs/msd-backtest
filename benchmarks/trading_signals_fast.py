"""无持仓信息的交易策略：实现 `signals()` 走 Rust 向量化快速路径，用于性能测试。

信号只依赖行情，不查询 position/holding_ror：EMA 金叉且站上长均线、放量时买入；
死叉或跌破长均线卖出。信号在 T 日收盘生成，next_open 模式下于 T+1 开盘撮合。

运行示例：
    python -m mbt.run.cli src/bench/trading_signals_fast.py \
        -s SH600000 SZ000001 SH600519 -b 2024-01-01 -e 2025-12-31 -S
    python -m mbt.run.cli src/bench/trading_signals_fast.py -s A_STOCKS \
        -b 2024-01-01 -e 2025-12-31 -S
"""

from __future__ import annotations

import numpy as np
import alpha as al
import mbt

M_AUTHOR = "Tencent Hunyuan Hy"


class Strategy(mbt.Strategy):
    """纯行情 EMA 交叉择时，无持仓状态，走 fast-path 模拟引擎。"""

    exec_mode = "next_open"
    warmup_bars = 60

    def __init__(self, fast: int = 5, slow: int = 20, trend: int = 120, volume_window: int = 20):
        super().__init__()
        self.fast = max(2, int(fast))
        self.slow = max(3, int(slow))
        self.trend = max(self.slow + 1, int(trend))
        self.volume_window = max(2, int(volume_window))
        if self.fast >= self.slow:
            raise ValueError(f"fast({self.fast}) must be less than slow({self.slow})")

    def signals(self, ctx: mbt.Context) -> tuple[np.ndarray, np.ndarray]:
        """全区间一次性返回 (buy, sell) 信号块，长度 n_objs * bars。"""
        bars = ctx.bars
        with al.Context(
            groups=ctx.n_objs,
            flags=al.FLAG_SKIP_NAN,
            start=0,
            end=bars,
        ):
            fast_ma = al.EMA(ctx.close, self.fast)
            slow_ma = al.EMA(ctx.close, self.slow)
            trend_ma = al.MA(ctx.close, self.trend)
            volume_ma = al.MA(ctx.volume, self.volume_window)
            golden = np.asarray(al.CROSS(fast_ma, slow_ma), dtype=bool)
            death = np.asarray(al.RCROSS(fast_ma, slow_ma), dtype=bool)

        valid = np.isfinite(ctx.close) & (ctx.close > 0.0)
        warmup = np.tile(np.arange(bars) < self.warmup_bars, ctx.n_objs)
        tradable = np.repeat(
            np.array(
                [not obj.startswith(("SH000", "SZ399")) for obj in ctx.objs()],
                dtype=bool,
            ),
            bars,
        )
        base = valid & ~warmup & tradable
        buy = (golden & (ctx.close > trend_ma) & (ctx.volume > volume_ma) & base).astype(np.float64)
        sell = (death | (ctx.close < trend_ma)) & base
        return buy, sell.astype(np.float64)
