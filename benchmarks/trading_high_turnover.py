"""高换手快路径交易策略：短周期 EMA(2/5) 频繁交叉，用于压测高成交密度下的内核与报告阶段。

与 trading_signals_fast.py 同走 signals() Rust Simulator 快路径，但去除长均线与成交量门控，
使用超短均线产生约高一个数量级的交易笔数，重点评估：
1. Rust Simulator 高频撮合与交易流水回传；
2. `_hydrate_accounts()` 构造大量 Python `Record` 对象的开销；
3. `Report.calculate_metrics()` 统计大量持仓周期与 JSON 导出的成本。

运行示例：
    uv run python -m mbt.run.cli src/bench/trading_high_turnover.py \
        -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
"""

from __future__ import annotations

import numpy as np
import alpha as al
import mbt

M_AUTHOR = "Gemini 4 Hydrogen"


class Strategy(mbt.Strategy):
    """高频短周期 EMA 交叉（fast-path），压测大量交易流水下的模拟与报告性能。"""

    exec_mode = "next_open"
    warmup_bars = 60

    def __init__(self, fast: int = 2, slow: int = 5):
        super().__init__()
        self.fast = max(2, int(fast))
        self.slow = max(self.fast + 1, int(slow))

    def signals(self, ctx: mbt.Context) -> tuple[np.ndarray, np.ndarray]:
        """全区间一次性生成高频交叉买卖信号。"""
        bars = ctx.bars
        with al.Context(
            groups=ctx.n_objs,
            flags=al.FLAG_SKIP_NAN,
            start=0,
            end=bars,
        ):
            fast_ma = al.EMA(ctx.close, self.fast)
            slow_ma = al.EMA(ctx.close, self.slow)
            golden = np.asarray(al.CROSS(fast_ma, slow_ma), dtype=bool)
            death = np.asarray(al.RCROSS(fast_ma, slow_ma), dtype=bool)

        valid = np.isfinite(ctx.close) & (ctx.close > 0.0)
        warmup = np.tile(np.arange(bars) < ctx.warmup_bars, ctx.n_objs)
        tradable = np.repeat(
            np.array(
                [not obj.startswith(("SH000", "SZ399")) for obj in ctx.objs()],
                dtype=bool,
            ),
            bars,
        )
        base = valid & ~warmup & tradable
        buy = (golden & base).astype(np.float64)
        sell = (death & base).astype(np.float64)
        return buy, sell
