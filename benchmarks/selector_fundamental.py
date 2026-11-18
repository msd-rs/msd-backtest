"""财报选股器：按法定披露日对齐的季度财报筛选 + 未复权价格估值，用于性能测试。

字段经 MSD 实例校验：f205 年化 ROE、f197 净利润增长率、f223 资产负债率、
f003 每股净资产。估值统一使用未复权 raw_close，避免复权价格扭曲 PB。

运行示例：
    python -m mbt.select.cli src/bench/selector_fundamental.py \
        -s A_STOCKS_EXCLUDE_ST -m http://127.0.0.1:50510 -f
    python -m mbt.select.cli src/bench/selector_fundamental.py \
        -s SH600000 SZ000001 SH600519 -d 2025-06-30 -a top_k=5 -f
"""

from __future__ import annotations

import polars as pl
import mbt
from mbt.select import A_STOCKS_EXCLUDE_ST

M_AUTHOR = "Tencent Hunyuan Hy"


class Selector(mbt.Selector):
    """盈利质量 + 成长 + 低杠杆 + 合理 PB 的多条件财报选股器。"""

    fin_fields: list[str] = ["f205", "f197", "f223", "f003"]
    fin_only_year: bool = False

    def __init__(
        self,
        min_roe: float = 8.0,
        min_stable_roe: float = 5.0,
        min_growth: float = 10.0,
        max_debt: float = 70.0,
        max_pb: float = 6.0,
        report_history: int = 4,
        top_k: int = 20,
    ):
        super().__init__()
        self.min_roe = float(min_roe)
        self.min_stable_roe = float(min_stable_roe)
        self.min_growth = float(min_growth)
        self.max_debt = float(max_debt)
        self.max_pb = float(max_pb)
        self.report_history = max(1, int(report_history))
        self.top_k = max(1, int(top_k))
        self.factors = pl.DataFrame()

    def step00(self, stocks: list[str]) -> list[str]:
        """初始化股票池：独立运行时使用全 A 股（剔除 ST）。"""
        if not stocks and not self.is_backtest:
            stocks = A_STOCKS_EXCLUDE_ST
        return stocks

    def step01(self, stocks: list[str]) -> list[str]:
        """财报初筛：最近报告 ROE/增速/负债率达标，且近 N 期 ROE 均不为负。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        fin = self.dp.load_financial(
            stocks,
            fields=self.fin_fields,
            only_year=False,
            lastN=self.report_history,
        )
        if isinstance(fin, dict):
            parts = []
            for sym, frame in fin.items():
                if frame is not None and not frame.is_empty():
                    parts.append(frame.with_columns(pl.lit(sym).alias("obj")))
            fin = pl.concat(parts) if parts else pl.DataFrame()

        if fin.is_empty() or not set(self.fin_fields).issubset(fin.columns):
            self.factors = pl.DataFrame()
            return []

        latest = (
            fin.filter(pl.col("obj").is_in(stocks))
            .sort(["obj", "report_ts"])
            .group_by("obj", maintain_order=True)
            .tail(self.report_history)
        )
        picked = (
            latest.group_by("obj", maintain_order=True)
            .agg([
                pl.col("f205").last().alias("roe"),
                pl.col("f205").min().alias("min_roe"),
                pl.col("f197").last().alias("profit_growth"),
                pl.col("f223").last().alias("debt_ratio"),
                pl.col("f003").last().alias("bps"),
            ])
            .filter(
                (pl.col("roe") >= self.min_roe)
                & (pl.col("min_roe") >= self.min_stable_roe)
                & (pl.col("profit_growth") >= self.min_growth)
                & (pl.col("debt_ratio") <= self.max_debt)
                & (pl.col("bps") > 0.0)
            )
        )
        self.factors = picked
        return picked["obj"].to_list()

    def step02(self, stocks: list[str]) -> list[str]:
        """估值复筛：以未复权价格计算 PB，按 ROE 从高到低取前 K 名。"""
        if not stocks:
            self.factors = pl.DataFrame()
            return []

        snapshot = self.dp.load_snapshot(stocks, fin_fields=[])
        if snapshot.is_empty():
            self.factors = pl.DataFrame()
            return []

        price_col = "raw_close" if "raw_close" in snapshot.columns else "close"
        merged = self.factors.join(
            snapshot.select([pl.col("obj"), pl.col(price_col).alias("price")]),
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
            .head(self.top_k)
        )
        self.factors = picked
        return picked["obj"].to_list()
