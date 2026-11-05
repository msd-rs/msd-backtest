import logging
import numpy as np
import polars as pl
import alpha as al
import mbt
from mbt.select import A_STOCKS_EXCLUDE_ST, Panel

logger = logging.getLogger("selector")


class VolumeBreakoutMonthlyYangSelector(mbt.Selector):
  """
  放量突破且月线三连阳选股器

  选股逻辑：
  1. 基础过滤：剔除 ST 股票、停牌股及无成交股票，要求当日收阳（收盘价 >= 开盘价）。
  2. 月线三连阳：最近 3 个自然月（包含当前月）每个月的月线均为阳线（收盘价 > 开盘价）。
  3. 放量突破：当日收盘价突破过去 N 日（默认 20 日）最高价，且当日成交量达到前 5 日均量的指定倍数（默认 >= 1.5 倍）且大于昨日成交量。
  """

  def __init__(
    self,
    breakout_days: int = 20,
    volume_ratio: float = 1.5,
    volume_ma_days: int = 5,
    monthly_yang_count: int = 3,
    today_yang: bool = True,
  ):
    super().__init__()
    self.breakout_days = breakout_days
    self.volume_ratio = volume_ratio
    self.volume_ma_days = volume_ma_days
    self.monthly_yang_count = monthly_yang_count
    self.today_yang = today_yang
    self.factors: pl.DataFrame = pl.DataFrame()

  def step00(self, stocks: list[str]) -> list[str]:
    """初步过滤：剔除ST股票、停牌股及无成交股票，可选过滤当日收阳标的。"""
    if not stocks:
      stocks = A_STOCKS_EXCLUDE_ST

    snap = self.dp.load_snapshot(stocks)
    if snap.is_empty():
      self.factors = pl.DataFrame()
      return []

    cond = (pl.col("volume") > 0) & (pl.col("close") > 0)
    if self.today_yang:
      cond = cond & (pl.col("close") >= pl.col("open"))

    filtered = snap.filter(cond)
    selected = filtered["obj"].to_list()

    cols_to_keep = [c for c in ["obj", "close", "open", "volume", "amount"] if c in filtered.columns]
    self.factors = filtered.select(cols_to_keep)
    return selected

  def step01(self, stocks: list[str]) -> list[str]:
    """月线三连阳筛选：最近 3 个自然月（包含当前月）每个月的收盘价均高于开盘价。"""
    if not stocks:
      return []

    # 加载约半年(120根)日K线以覆盖最近至少 3 个自然月
    panel = self.dp.load_kline(stocks, lastN=120, baseline="SH000001")
    if panel.bars < 60:
      return []

    # 使用 Panel 多周期重采样为月线面板
    month_panel = panel.resample("1M")
    if month_panel.bars < self.monthly_yang_count:
      return []

    count = self.monthly_yang_count
    # 获取最近 N 个月的开盘价与收盘价 (n_objs, count)
    m_opens = month_panel.column("open")[:, -count:]
    m_closes = month_panel.column("close")[:, -count:]

    # 连续 count 个月收阳 (收盘价 > 开盘价 且 非 NaN)
    valid_data = ~np.isnan(m_closes).any(axis=1)
    is_all_yang = valid_data & (m_closes > m_opens).all(axis=1)

    passed_indices = np.where(is_all_yang)[0]
    selected = [month_panel.objs[idx] for idx in passed_indices]

    # 装配月度涨幅因子
    month_timeline = month_panel.timeline()[-count:]
    month_gain_cols: dict[str, np.ndarray] = {}
    for i in range(count):
      m_str = str(np.datetime64(month_timeline[i], "M"))
      o_i = m_opens[:, i]
      c_i = m_closes[:, i]
      gain = (c_i - o_i) / np.where(o_i > 0, o_i, np.nan)
      month_gain_cols[f"m{i+1}_{m_str}_gain"] = np.round(gain * 100, 2)

    yang_factors = month_panel.cross_section(**month_gain_cols).filter(pl.col("obj").is_in(selected))
    if not self.factors.is_empty():
      self.factors = self.factors.filter(pl.col("obj").is_in(selected)).join(
        yang_factors, on="obj", how="left"
      )
    else:
      self.factors = yang_factors

    return selected

  def step02(self, stocks: list[str]) -> list[str]:
    """放量突破筛选：当日收盘价突破前 N 日最高价，且当日成交量较 5 日均量显著放大。"""
    if not stocks:
      return []

    bars_needed = max(self.breakout_days + self.volume_ma_days + 5, 60)
    panel = self.dp.load_kline(stocks, lastN=bars_needed, baseline="SH000001")
    if panel.bars < self.breakout_days + 1:
      return []

    hhv = panel.compute(al.HHV, "high", self.breakout_days)
    vol_ma = panel.compute(al.MA, "volume", self.volume_ma_days)

    close = panel.last("close")
    prev_high = hhv[:, -2]  # 前 N 日最高价（不含当日）
    vol = panel.last("volume")
    prev_vol = panel.offset("volume", -2)
    prev_vol_ma = vol_ma[:, -2]  # 前 5 日成交量均值（不含当日）

    vol_ratio = vol / np.where(prev_vol_ma > 0, prev_vol_ma, np.nan)
    breakout_pct = (close - prev_high) / np.where(prev_high > 0, prev_high, np.nan)

    breakout_cond = (
      (close >= prev_high)
      & (vol_ratio >= self.volume_ratio)
      & (vol > prev_vol)
      & (~np.isnan(prev_high))
      & (~np.isnan(vol_ratio))
    )

    passed_indices = np.where(breakout_cond)[0]
    selected = [panel.objs[idx] for idx in passed_indices]

    breakout_factors = panel.cross_section(
      prev_high=np.round(prev_high, 2),
      breakout_pct=np.round(breakout_pct * 100, 2),
      vol_ratio=np.round(vol_ratio, 2),
      prev_vol_ma=np.round(prev_vol_ma, 0),
    ).filter(pl.col("obj").is_in(selected))

    if not self.factors.is_empty():
      self.factors = self.factors.filter(pl.col("obj").is_in(selected)).join(
        breakout_factors, on="obj", how="left"
      )
    else:
      self.factors = breakout_factors

    return selected

