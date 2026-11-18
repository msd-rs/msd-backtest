"""包含持仓信息的交易策略：用 holding_shares / holding_rors 做止损止盈，用于性能测试。

与 fast-path 策略相对，本策略必须逐 bar 读取账户状态，走 Python execute() 路径；
所有下单决策仍按截面 NumPy 掩码向量化提交，避免逐股票循环。

运行示例：
    python -m mbt.run.cli src/bench/trading_position_aware.py \
        -s SH600000 SZ000001 SH600519 -b 2024-01-01 -e 2025-12-31 -S
    python -m mbt.run.cli src/bench/trading_position_aware.py -s A_STOCKS \
        -b 2024-01-01 -e 2025-12-31 -a stop_loss=0.08 -a take_profit=0.20 -S
"""

from __future__ import annotations

import numpy as np
import alpha as al
import mbt

M_AUTHOR = "Tencent Hunyuan Hy"


class Strategy(mbt.Strategy):
    """均线金叉入场，按持仓收益率止损/止盈，跌破均线离场。"""

    exec_mode = "next_open"
    warmup_bars = 60

    def __init__(self, ma_period: int = 20, stop_loss: float = 0.07, take_profit: float = 0.18):
        super().__init__()
        self.ma_period = max(2, int(ma_period))
        self.stop_loss = max(0.001, float(stop_loss))
        self.take_profit = max(self.stop_loss + 0.001, float(take_profit))

    def prepare(self, ctx: mbt.Context):
        """全区间预计算均线信号，并初始化跨 bar 的挂单状态。"""
        with al.Context(
            groups=ctx.n_objs,
            flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE,
            start=0,
            end=ctx.bars,
        ):
            ma = al.MA(ctx.close, self.ma_period)
            self.entry_signal = np.asarray(al.CROSS(ctx.close, ma), dtype=bool) & (ctx.close > ma)
            self.trend_exit = np.asarray(ctx.close < ma, dtype=bool)

        self.offsets = np.arange(ctx.n_objs, dtype=np.int64) * ctx.bars
        self.tradable = np.array(
            [not obj.startswith(("SH000", "SZ399")) for obj in ctx.objs()],
            dtype=bool,
        )
        self.pending_sell = np.zeros(ctx.n_objs, dtype=bool)
        self.pending_buy = np.zeros(ctx.n_objs, dtype=bool)
        self.exit_counts = {"stop_loss": 0, "take_profit": 0, "trend_exit": 0}

    def execute(self, ctx: mbt.Context):
        """读取当前持仓截面，提交止损/止盈卖出与金叉买入。"""
        if ctx.is_warmup:
            return

        idx = self.offsets + ctx.bar
        shares = ctx.holding_shares()
        rors = np.nan_to_num(ctx.holding_rors(), nan=0.0)
        held = shares > 0.0

        # next_open 模式下当日挂单尚未成交，避免同一标的重复发单。
        self.pending_sell &= held
        pending_buy_prev = self.pending_buy.copy()
        self.pending_buy[:] = False

        manageable = held & ~self.pending_sell
        stop = manageable & (rors <= -self.stop_loss)
        take = manageable & (rors >= self.take_profit)
        trend = manageable & self.trend_exit[idx]
        sell = stop | take | trend

        buy = (
            self.entry_signal[idx]
            & self.tradable
            & ~held
            & ~self.pending_sell
            & ~pending_buy_prev
        )

        if sell.any():
            ctx.sell(sell)
        if buy.any():
            ctx.buy(buy)

        self.pending_sell[sell] = True
        self.pending_buy[buy] = True
        self.exit_counts["stop_loss"] += int(stop.sum())
        self.exit_counts["take_profit"] += int(take.sum())
        self.exit_counts["trend_exit"] += int(trend.sum())

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
