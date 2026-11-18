"""纯价量选股器：快照流动性初筛 + 历史 K 线趋势/放量确认，用于性能测试。

只使用前复权价与成交量，不读取任何财务字段；`step02` 通过 `load_kline`
做历史序列计算，便于压测选股器的面板加载与 alpha-lib 向量化路径。

运行示例：
    python -m mbt.select.cli src/bench/selector_price_volume.py \
        -s A_STOCKS_EXCLUDE_ST -m http://127.0.0.1:50510 -f
    python -m mbt.select.cli src/bench/selector_price_volume.py \
        -s SH600000 SZ000001 SH600519 -d 2025-06-30 -a top_k=5 -f
"""

from __future__ import annotations

import numpy as np
import polars as pl
import alpha as al
import mbt
from mbt.select import A_STOCKS_EXCLUDE_ST, Panel

M_AUTHOR = "Tencent Hunyuan Hy"


class Selector(mbt.Selector):
    """价量动量选股：价格站上多头排列均线且近期温和放量上攻。"""

    fin_fields: list[str] = []

    def __init__(self, max_candidates: int = 600, top_k: int = 20, lookback: int = 120):
        super().__init__()
        self.max_candidates = max(1, int(max_candidates))
        self.top_k = max(1, int(top_k))
        self.lookback = max(61, int(lookback))
        self.factors = pl.DataFrame()
        self._panel: Panel | None = None

    def step00(self, stocks: list[str]) -> list[str]:
        """初始化股票池：独立运行时使用全 A 股（剔除 ST）。"""
        if not stocks and not self.is_backtest:
            stocks = A_STOCKS_EXCLUDE_ST
        return stocks

    def step01(self, stocks: list[str]) -> list[str]:
        """快照初筛：剔除低价、低成交额与停牌标的，保留流动性前 N 名。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        snapshot = self.dp.load_snapshot(stocks, fin_fields=[])
        if snapshot.is_empty():
            self.factors = pl.DataFrame()
            return []

        price_col = "raw_close" if "raw_close" in snapshot.columns else "close"
        filtered = snapshot.filter(
            (pl.col(price_col) >= 3.0)
            & (pl.col("amount") >= 2.0e7)
            & (pl.col("volume") > 0.0)
        )
        if filtered.is_empty():
            self.factors = pl.DataFrame()
            return []

        filtered = filtered.sort("amount", descending=True).head(self.max_candidates)
        self.factors = filtered.select([
            pl.col("obj"),
            pl.col(price_col).alias("close"),
            pl.col("amount"),
        ])
        return filtered["obj"].to_list()

    def step02(self, stocks: list[str]) -> list[str]:
        """历史确认：MA20 > MA60、收盘站上 MA20、20 日涨幅 5%~60% 且量比不低于 1.2。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        panel = self.dp.load_kline(
            stocks,
            lastN=self.lookback,
            fin_fields=[],
            fin_only_year=True,
            baseline="SH000001",
        )
        self._panel = panel
        if panel.n_objs == 0 or panel.bars < 61:
            self.factors = pl.DataFrame()
            return []

        # 严格周期窗口：不足 60 根有效 K 线的标的保持 NaN，不参与下面的过滤。
        # alpha-lib 接收一维块布局 buffer，结果再还原为 (n_objs, bars) 视图。
        with panel.ctx(flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE):
            close_buf = panel.buffer("close")
            volume_buf = panel.buffer("volume")
            shape = (panel.n_objs, panel.bars)
            ma20 = np.asarray(al.MA(close_buf, 20)).reshape(shape)
            ma60 = np.asarray(al.MA(close_buf, 60)).reshape(shape)
            vol_ma20 = np.asarray(al.MA(volume_buf, 20)).reshape(shape)
            prev_close = np.asarray(al.REF(close_buf, 20)).reshape(shape)
            last_close = panel.last("close")

            with np.errstate(divide="ignore", invalid="ignore"):
                ret20 = last_close / prev_close[:, -1] - 1.0
                vol_ratio = panel.last("volume") / vol_ma20[:, -1]

        factors = panel.cross_section(
            close=last_close,
            ma20=ma20[:, -1],
            ma60=ma60[:, -1],
            ret20=ret20,
            vol_ratio=vol_ratio,
        )
        picked = (
            factors.filter(
                (pl.col("close") > pl.col("ma20"))
                & (pl.col("ma20") > pl.col("ma60"))
                & pl.col("ret20").is_between(0.05, 0.60)
                & (pl.col("vol_ratio") >= 1.2)
            )
            .sort("ret20", descending=True)
            .head(self.top_k)
        )
        self.factors = picked
        return picked["obj"].to_list()
