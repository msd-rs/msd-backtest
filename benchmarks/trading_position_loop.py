"""持仓感知逐股循环对照组：逻辑与 trading_position_aware.py 一致，改用逐股循环写法。

用于对比 `ctx.holding_shares()` / `ctx.holding_rors()` 截面向量化写法
与常见策略（如 `cta_momentum.py` 模版）中 `for g in range(ctx.n_objs): ctx.position(g)`
写法的实际性能差异。

运行示例：
    uv run python -m mbt.run.cli src/bench/trading_position_loop.py \
        -s A_STOCKS -b 2020-01-01 -e 2025-12-31 -m http://127.0.0.1:50511 -S
"""

from __future__ import annotations

import numpy as np
import alpha as al
import mbt

M_AUTHOR = "Gemini 4 Hydrogen"


class Strategy(mbt.Strategy):
    """均线金叉入场，逐股循环查询 ctx.position(g) / ctx.holding_ror(g) 止损止盈。"""

    exec_mode = "next_open"
    warmup_bars = 60

    def __init__(self, ma_period: int = 20, stop_loss: float = 0.07, take_profit: float = 0.18):
        super().__init__()
        self.ma_period = max(2, int(ma_period))
        self.stop_loss = max(0.001, float(stop_loss))
        self.take_profit = max(self.stop_loss + 0.001, float(take_profit))

    def prepare(self, ctx: mbt.Context):
        """全区间预计算均线交叉与趋势离场信号。"""
        with al.Context(
            groups=ctx.n_objs,
            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
            start=0,
            end=ctx.bars,
        ):
            ma = al.MA(ctx.close, self.ma_period)
            self.entry_signal = np.asarray(al.CROSS(ctx.close, ma), dtype=bool) & (ctx.close > ma)
            self.trend_exit = np.asarray(ctx.close < ma, dtype=bool)

        self.tradable = [not obj.startswith(("SH000", "SZ399")) for obj in ctx.objs()]
        self.pending_sell = [False] * ctx.n_objs
        self.pending_buy = [False] * ctx.n_objs
        self.exit_counts = {"stop_loss": 0, "take_profit": 0, "trend_exit": 0}

    def execute(self, ctx: mbt.Context):
        """逐股循环读取账户持仓与收益率，按单标的提交买卖指令。"""
        if ctx.is_warmup:
            return

        n_objs = ctx.n_objs
        bars = ctx.bars
        bar = ctx.bar
        entry_sig = self.entry_signal
        trend_sig = self.trend_exit

        for g in range(n_objs):
            pos = ctx.position(g)
            held = pos.shares > 0.0

            if not held:
                self.pending_sell[g] = False

            pending_buy_prev = self.pending_buy[g]
            self.pending_buy[g] = False
            flat_idx = g * bars + bar

            if held:
                if self.pending_sell[g]:
                    continue
                ror = ctx.holding_ror(g)
                if ror <= -self.stop_loss:
                    ctx.sell(g)
                    self.pending_sell[g] = True
                    self.exit_counts["stop_loss"] += 1
                elif ror >= self.take_profit:
                    ctx.sell(g)
                    self.pending_sell[g] = True
                    self.exit_counts["take_profit"] += 1
                elif bool(trend_sig[flat_idx]):
                    ctx.sell(g)
                    self.pending_sell[g] = True
                    self.exit_counts["trend_exit"] += 1
            elif self.tradable[g] and not pending_buy_prev and bool(entry_sig[flat_idx]):
                ctx.buy(g)
                self.pending_buy[g] = True

    def finalize(self, ctx: mbt.Context) -> dict:
        """输出离场原因统计。"""
        return {
            "exit_stop_loss": self.exit_counts["stop_loss"],
            "exit_take_profit": self.exit_counts["take_profit"],
            "exit_trend_exit": self.exit_counts["trend_exit"],
            "ma_period": self.ma_period,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }
