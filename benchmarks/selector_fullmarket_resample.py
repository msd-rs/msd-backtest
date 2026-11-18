"""全市场重度选股器：跳过快照初筛，全池 load_kline + 财报字段对齐 + 周线重采样。

用于压测 Selector 在无前置瘦身时的重负载路径：
1. 全市场 5226 只股票直接加载 250 根日 K 线 + 财务字段（f003/f205 法定披露日对齐）；
2. 调用 `panel.resample("1W")` 进行全市场周线聚合；
3. 结合周线均线多头与日线 MA250、未复权 PB 与 ROE 筛选。

运行示例：
    uv run python -m mbt.select.cli src/bench/selector_fullmarket_resample.py \
        -s A_STOCKS -d 2025-12-31 -m http://127.0.0.1:50511 -f
"""

from __future__ import annotations

import numpy as np
import polars as pl
import alpha as al
import mbt
from mbt.select import A_STOCKS, Panel

M_AUTHOR = "Gemini 4 Hydrogen"


class Selector(mbt.Selector):
    """全市场长序列面板 + 周线多周期重采样 + 财报估值复合选股器。"""

    fin_fields: list[str] = ["f003", "f205"]
    fin_only_year: bool = False

    def __init__(
        self,
        lookback: int = 250,
        min_roe: float = 8.0,
        max_pb: float = 6.0,
        top_k: int = 20,
    ):
        super().__init__()
        self.lookback = max(120, int(lookback))
        self.min_roe = float(min_roe)
        self.max_pb = float(max_pb)
        self.top_k = max(1, int(top_k))
        self.factors = pl.DataFrame()

    def step00(self, stocks: list[str]) -> list[str]:
        """独立运行时默认全 A 股；不做快照预过滤，直接传递全池至 step01。"""
        if not stocks and not self.is_backtest:
            stocks = A_STOCKS
        return stocks

    def step01(self, stocks: list[str]) -> list[str]:
        """全池加载 250 日 K 线与财报字段，重采样周线后联合筛选。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        panel: Panel = self.dp.load_kline(
            stocks,
            lastN=self.lookback,
            fin_fields=self.fin_fields,
            fin_only_year=self.fin_only_year,
            baseline="SH000001",
        )
        if panel.n_objs == 0 or panel.bars < 60:
            self.factors = pl.DataFrame()
            return []

        shape = (panel.n_objs, panel.bars)
        ma_win = min(250, panel.bars)
        with panel.ctx(flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE):
            close_buf = panel.buffer("close")
            d_ma_long = np.asarray(al.MA(close_buf, ma_win)).reshape(shape)[:, -1]
            d_prev60 = np.asarray(al.REF(close_buf, 60)).reshape(shape)[:, -1]
            d_close = panel.last("close")
            with np.errstate(divide="ignore", invalid="ignore"):
                ret60 = np.where(d_prev60 > 0.0, d_close / d_prev60 - 1.0, np.nan)

        weekly = panel.slice(
            0, panel.bars, columns=["ts", "open", "high", "low", "close", "volume"]
        ).resample("1W")
        if weekly.bars < 20:
            self.factors = pl.DataFrame()
            return []

        w_shape = (weekly.n_objs, weekly.bars)
        with weekly.ctx(flags=al.FLAG_SKIP_NAN | al.FLAG_STRICTLY_CYCLE):
            w_close_buf = weekly.buffer("close")
            w_ma10 = np.asarray(al.MA(w_close_buf, 10)).reshape(w_shape)[:, -1]
            w_ma20 = np.asarray(al.MA(w_close_buf, 20)).reshape(w_shape)[:, -1]
            w_close = weekly.last("close")

        raw_close = panel.last("raw_close") if "raw_close" in panel.columns else d_close
        bps = panel.last("f003") if "f003" in panel.columns else np.full(panel.n_objs, np.nan)
        roe = panel.last("f205") if "f205" in panel.columns else np.full(panel.n_objs, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            pb = np.where((raw_close > 0.0) & (bps > 0.0), raw_close / bps, np.nan)

        factors = panel.cross_section(
            close=d_close,
            raw_close=raw_close,
            ma_long=d_ma_long,
            w_close=w_close,
            w_ma10=w_ma10,
            w_ma20=w_ma20,
            ret60=ret60,
            roe=roe,
            pb=pb,
        )
        picked = (
            factors.filter(
                pl.col("close").is_finite()
                & pl.col("ma_long").is_finite()
                & (pl.col("close") > pl.col("ma_long"))
                & pl.col("w_ma10").is_finite()
                & pl.col("w_ma20").is_finite()
                & (pl.col("w_close") > pl.col("w_ma10"))
                & (pl.col("w_ma10") > pl.col("w_ma20"))
                & pl.col("roe").is_finite()
                & (pl.col("roe") >= self.min_roe)
                & pl.col("pb").is_finite()
                & (pl.col("pb") > 0.0)
                & (pl.col("pb") <= self.max_pb)
            )
            .sort("ret60", descending=True)
            .head(self.top_k)
        )
        self.factors = picked
        return picked["obj"].to_list()
