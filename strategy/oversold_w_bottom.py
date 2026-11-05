"""超跌 + W底 + 归母净利润高增长 + 低换手 选股器

选股思路: 找基本面仍在高增长(归母净利润同比 > 46%), 但股价已被打到低位、
近期成交清淡(近 5 日日均换手率 < 5%), 且在低位刚刚做出 W 底(双底)形态的股票,
即"业绩好 + 跌得深 + 卖压枯竭 + 形态转强"。

  step00  基础股票池: 默认全部 A 股(剔除 ST), 剔除 B 股与北交所
  step01  业绩成长: 归母净利润(f097)同比增长率 > min_growth(默认 46%),
          报告期相对全市场最新报告期的滞后不超过 max_report_lag 天
  step02  规模与流动性: 剔除停牌/无成交, 成交额与总市值下限(默认很宽松, 仅做数据有效性过滤)
  step03  交易清淡 + 超跌: 近 5 日日均换手率 < max_turnover(默认 5%),
          且收盘价相对 lookback 日最高收盘的回撤 >= min_drawdown(默认 25%)
  step04  W 底: 最近 lookback 日窗口内存在两个相近低点(左底/右底),
          两底时间间隔在 [min_gap, max_gap] 内, 右底距今 <= recent_days,
          两底高度接近(|差| <= tol_pct), 中间反弹高点(颈线)高出底部 [neck_pct, max_neck_pct],
          两底贴近窗口最低点(<= low_tol_pct), 且当前价已自右底反弹但尚未大幅突破颈线
  step05  按右底新鲜度、距右底涨幅排序, 取前 top_n 只

用法:
  python -m mbt.select.cli selectors/oversold_w_bottom.py
  python -m mbt.select.cli selectors/oversold_w_bottom.py -f           # 输出因子明细
  python -m mbt.select.cli selectors/oversold_w_bottom.py -a min_growth=30 max_turnover=8
  python -m mbt.select.cli selectors/oversold_w_bottom.py -s SZ000651 SH601398
  python -m mbt.select.cli selectors/oversold_w_bottom.py -d 2026-09-30
"""

import logging
from typing import Sequence

import mbt
import numpy as np
import polars as pl
import alpha as al
import time

from mbt.select import ALL_STOCKS, A_STOCKS_EXCLUDE_ST

logger = logging.getLogger("selector")

# 单次批量请求的股票数量, 仅用于 load_kline:
# Panel 是 (n_objs, bars) × 全部列的 float64 定宽矩阵, 800 只约 18MB,
# 一次取 5000 只则要 370MB 且并不更快(实测 220ms/150MB vs 248ms/368MB)。
# 财务与快照是"每只十几行/一行"的小载荷, 分批只会多付 RPC 往返, 因此不分批。
CHUNK_SIZE = 800
# 作为对齐基准的指数(交易日历完整)
BASELINE = "SH000001"
# 归母净利润字段
FIN_NET_PROFIT = "f097"
# 换手率统计窗口(交易日)
TURNOVER_BARS = 5
# 上年同期亏损、本期盈利(扭亏)时使用的增速替代值(%)
TURNAROUND_GROWTH = 999.99


def _chunks(items: Sequence[str], size: int = CHUNK_SIZE):
  for i in range(0, len(items), size):
    yield list(items[i:i + size])


def _finite_mean_last(arr: np.ndarray, periods: int) -> np.ndarray:
  """按行取末尾 periods 列的有效值均值, 全为 NaN 时返回 NaN"""
  tail = arr[:, -periods:]
  ok = np.isfinite(tail)
  cnt = ok.sum(axis=1)
  total = np.where(ok, tail, 0.0).sum(axis=1)
  with np.errstate(divide="ignore", invalid="ignore"):
    return np.where(cnt > 0, total / np.maximum(cnt, 1), np.nan)


GROWTH_SCHEMA = {
  "obj": pl.String,
  "np_yoy_pct": pl.Float64,
  "np_cur": pl.Float64,
  "np_prev": pl.Float64,
  "report_ts": pl.Datetime("us"),
}

KLINE_SCHEMA = {
  "obj": pl.String,
  "close": pl.Float64,
  "valid_bars": pl.Int32,
  "dd_pct": pl.Float64,
  "turnover_5d": pl.Float64,
  "turnover_last": pl.Float64,
  "ma5": pl.Float64,
  "ma10": pl.Float64,
  "w_found": pl.Boolean,
  "w_gap": pl.Int32,
  "w_right_age": pl.Int32,
  "w_l1": pl.Float64,
  "w_l2": pl.Float64,
  "w_diff_pct": pl.Float64,
  "w_neck_pct": pl.Float64,
  "w_near_low_pct": pl.Float64,
  "w_from_low_pct": pl.Float64,
  "w_break_pct": pl.Float64,
}

# W 底因子列, 与 KLINE_SCHEMA 中的 w_* 对应
W_FIELDS = tuple(k for k in KLINE_SCHEMA if k.startswith("w_"))


class OversoldWBottomSelector(mbt.Selector):
  """超跌且 W 底、归母净利润高增长、换手率低"""

  def __init__(
    self,
    min_growth: float = 46.0,
    growth_mode: str = "yoy",
    allow_turnaround: bool = True,
    max_report_lag: int = 200,
    max_turnover: float = 5.0,
    lookback: int = 120,
    min_drawdown: float = 25.0,
    trough_span: int = 3,
    min_gap: int = 10,
    max_gap: int = 60,
    recent_days: int = 30,
    tol_pct: float = 6.0,
    neck_pct: float = 6.0,
    max_neck_pct: float = 40.0,
    low_tol_pct: float = 5.0,
    rebound_min: float = 2.0,
    max_break_pct: float = 8.0,
    require_ma_up: bool = False,
    min_bars: int = 150,
    min_amount: float = 1e7,
    min_market_cap: float = 0.0,
    top_n: int = 30,
  ):
    """
    Args:
      min_growth: 归母净利润增长率下限(%), 即"归母净利润增长率 > min_growth"
      growth_mode: 增速口径, "yoy" 为最新报告期同比(单季/累计均可),
                   "annual" 为最近两期年报同比
      allow_turnaround: 上年同期亏损、本期盈利(扭亏)是否算作达标, 记为 TURNAROUND_GROWTH(%)
      max_report_lag: 最新报告期相对全市场最新报告期的最大滞后天数, 避免用过期财报
      max_turnover: 近 TURNOVER_BARS 日日均换手率上限(%)
      lookback: 超跌与 W 底形态的观察窗口(交易日)
      min_drawdown: 最小回撤(%), 收盘价相对窗口内最高收盘的跌幅
      trough_span: 局部低点判定半径, 低点需低于左右各 trough_span 根K线的最低价
      min_gap: 左底与右底的最小间隔(交易日)
      max_gap: 左底与右底的最大间隔(交易日), 避免把相隔过远的两个低点当成同一个W
      recent_days: 右底距今的最大交易日数, 保证形态是"刚刚形成"的
      tol_pct: 两个低点的高度差容忍度(%), 左右底价格需接近
      neck_pct: 颈线(两底之间的最高收盘)相对底部的最小抬升幅度(%)
      max_neck_pct: 颈线相对底部的最大抬升幅度(%), 过大说明是"暴涨后暴跌"而非双底
      low_tol_pct: 两底相对窗口最低点的最大允许高出幅度(%), 保证形态处在低位
      rebound_min: 当前价相对右底的最小反弹幅度(%), 确认已从右底转强
      max_break_pct: 当前价相对颈线的最大超出幅度(%), 避免形态已被大幅突破
      require_ma_up: 是否额外要求收盘价站上 MA5(短期转强确认)
      min_bars: 最少有效K线数, 剔除次新股与长期停牌
      min_amount: 最近交易日最低成交额(元)
      min_market_cap: 最低总市值(元)
      top_n: 最终输出股票数量
    """
    super().__init__()
    self.min_growth = min_growth
    self.growth_mode = growth_mode
    self.allow_turnaround = allow_turnaround
    self.max_report_lag = max_report_lag
    self.max_turnover = max_turnover
    self.lookback = lookback
    self.min_drawdown = min_drawdown
    self.trough_span = trough_span
    self.min_gap = min_gap
    self.max_gap = max_gap
    self.recent_days = recent_days
    self.tol_pct = tol_pct
    self.neck_pct = neck_pct
    self.max_neck_pct = max_neck_pct
    self.low_tol_pct = low_tol_pct
    self.rebound_min = rebound_min
    self.max_break_pct = max_break_pct
    self.require_ma_up = require_ma_up
    self.min_bars = min_bars
    self.min_amount = min_amount
    self.min_market_cap = min_market_cap
    self.top_n = top_n

    # K线长度需覆盖形态窗口、均线与数据完整性检查
    self.history_bars = max(lookback + 30, min_bars + 20)

    self.factors: pl.DataFrame = pl.DataFrame()
    self._snap: pl.DataFrame | None = None
    self._snap_fetched: set[str] = set()
    self._growth: pl.DataFrame | None = None
    self._growth_fetched: set[str] = set()
    self._kline: pl.DataFrame | None = None
    self._kline_fetched: set[str] = set()

  # ------------------------------------------------------------------ 数据获取

  def _snapshots(self, objs: list[str]) -> pl.DataFrame:
    """获取最新快照(行情+股本), 结果带缓存; 载荷很小, 一次取完"""
    missing = [s for s in objs if s not in self._snap_fetched]
    if missing:
      df = self.dp.load_snapshot(missing)
      self._snap = df if self._snap is None else pl.concat([self._snap, df])
      self._snap_fetched.update(missing)
    if self._snap is None:
      return pl.DataFrame(schema={"obj": pl.String})
    return self._snap.filter(pl.col("obj").is_in(objs))

  def _growth_factors(self, objs: list[str]) -> pl.DataFrame:
    """获取财务并计算归母净利润增长率, 结果带缓存; 载荷很小, 一次取完"""
    missing = [s for s in objs if s not in self._growth_fetched]
    if missing:
      df = self._growth_batch(missing)
      if df is not None and df.height > 0:
        self._growth = df if self._growth is None else pl.concat([self._growth, df])
      self._growth_fetched.update(missing)
    if self._growth is None:
      return pl.DataFrame(schema=GROWTH_SCHEMA)
    return self._growth.filter(pl.col("obj").is_in(objs))

  def _growth_batch(self, batch: list[str]) -> pl.DataFrame:
    """计算一批股票的归母净利润同比增速

    财务数据为季度累计值(03-31/06-30/09-30/12-31), 同比即与去年同期同一报告期比较。
    上年同期为亏损(<=0)时增速无意义: 本期盈利按"扭亏"处理(增长率记为 TURNAROUND_GROWTH),
    两期均亏损则记为 NaN 并淘汰。

    实现上不逐股循环: mbt 0.1.12 起 `load_financial` 直接返回长表(obj, ts, *fields),
    用 (obj, ts - 1 年) 自连接取上年同期, 分组取最新一期, 一次算完。
    """
    t1 = time.time()
    fin = self.dp.load_financial(batch, [FIN_NET_PROFIT], only_year=False, lastN=12)
    logger.info(f"load financial used: {time.time() - t1:.3f}s")
    
    if fin is None or fin.height == 0:
      return pl.DataFrame(schema=GROWTH_SCHEMA)
    if not {"obj", "ts", FIN_NET_PROFIT}.issubset(fin.columns):
      return pl.DataFrame(schema=GROWTH_SCHEMA)

    flat = (
      fin.select(["obj", "ts", FIN_NET_PROFIT])
      .drop_nulls(FIN_NET_PROFIT)
      .unique(subset=["obj", "ts"], keep="last")
      .sort(["obj", "ts"])
    )
    if self.growth_mode == "annual":
      flat = flat.filter(pl.col("ts").dt.month() == 12)
    if flat.height == 0:
      return pl.DataFrame(schema=GROWTH_SCHEMA)

    latest = flat.group_by("obj").agg(
      pl.col("ts").sort_by("ts").last().alias("report_ts"),
      pl.col(FIN_NET_PROFIT).sort_by("ts").last().alias("np_cur"),
    )
    prev = flat.select(
      pl.col("obj"),
      pl.col("ts").alias("prev_ts"),
      pl.col(FIN_NET_PROFIT).alias("np_prev"),
    )
    # 注意 polars 的 NaN 语义: NaN > 0 为 True, 与 Python float 相反。
    # 这里显式排除 NaN, 保证与逐股版本(nan > 0 -> False)的判断完全一致。
    prev_positive = (pl.col("np_prev") > 0) & pl.col("np_prev").is_not_nan()
    cur_positive = (pl.col("np_cur") > 0) & pl.col("np_cur").is_not_nan()

    latest = (
      latest
      .with_columns(pl.col("report_ts").dt.offset_by("-1y").alias("prev_ts"))
      .join(prev, on=["obj", "prev_ts"], how="left")
      # 没有上年同期(新股/缺季报)的股票直接淘汰, 与原实现一致
      .filter(pl.col("np_prev").is_not_null())
      .with_columns(
        pl.when(prev_positive)
        .then((pl.col("np_cur") / pl.col("np_prev") - 1) * 100)
        .when((~prev_positive) & cur_positive & pl.lit(self.allow_turnaround))
        .then(pl.lit(TURNAROUND_GROWTH))
        .otherwise(pl.lit(float("nan")))
        .alias("np_yoy_pct")
      )
    )

    return latest.select(
      pl.col("obj"),
      pl.col("np_yoy_pct").cast(pl.Float64),
      pl.col("np_cur").cast(pl.Float64),
      pl.col("np_prev").cast(pl.Float64),
      pl.col("report_ts").cast(pl.Datetime("us")),
    ).sort("obj")

  def _kline_factors(self, objs: list[str]) -> pl.DataFrame:
    """分批计算换手率、超跌幅度、W 底形态, 结果带缓存"""
    missing = [s for s in objs if s not in self._kline_fetched]
    if missing:
      frames = [self._kline_batch(batch) for batch in _chunks(missing)]
      frames = [f for f in frames if f is not None and f.height > 0]
      if frames:
        df = pl.concat(frames)
        self._kline = df if self._kline is None else pl.concat([self._kline, df])
      self._kline_fetched.update(missing)
    if self._kline is None:
      return pl.DataFrame(schema=KLINE_SCHEMA)
    return self._kline.filter(pl.col("obj").is_in(objs))

  def _kline_batch(self, batch: list[str]) -> pl.DataFrame:
    """一批股票的 K 线因子: 换手率(volume/tradable_shares)、回撤、W 底"""
    panel = self.dp.load_kline(
      batch, [self.history_bars, 2, 2], baseline=BASELINE
    )
    n, bars = panel.n_objs, panel.bars

    with panel.ctx():
      close = panel.buffer("close").astype(np.float64)
      hhv = al.HHV(close, self.lookback).reshape(n, bars)
      ma5 = al.MA(close, 5).reshape(n, bars)
      ma10 = al.MA(close, 10).reshape(n, bars)

    close_2d = panel.column("close")
    low_2d = panel.column("low")
    vol_2d = panel.column("volume")
    shares_2d = panel.column("tradable_shares")

    with np.errstate(divide="ignore", invalid="ignore"):
      turnover = np.where(shares_2d > 0, vol_2d / shares_2d * 100, np.nan)

    close_last = close_2d[:, -1]
    with np.errstate(divide="ignore", invalid="ignore"):
      dd_pct = np.where(hhv[:, -1] > 0, (close_last / hhv[:, -1] - 1) * 100, np.nan)

    turnover_5d = _finite_mean_last(turnover, TURNOVER_BARS)

    df = pl.DataFrame({
      "obj": panel.objs,
      "close": close_last,
      "valid_bars": np.isfinite(close_2d).sum(axis=1),
      "dd_pct": dd_pct,
      "turnover_5d": turnover_5d,
      "turnover_last": turnover[:, -1],
      "ma5": ma5[:, -1],
      "ma10": ma10[:, -1],
      **self._w_bottom_batch(close_2d, low_2d),
    })
    return df.select([pl.col(k).cast(v) for k, v in KLINE_SCHEMA.items()])

  # ------------------------------------------------------------------ W 底识别

  def _w_bottom(self, close_row: np.ndarray, low_row: np.ndarray) -> dict:
    """单个股票的 W 底特征(批量实现的行版本, 便于单股调试与合成用例)"""
    res = self._w_bottom_batch(
      np.asarray(close_row, dtype=np.float64)[None, :],
      np.asarray(low_row, dtype=np.float64)[None, :],
    )
    row = {}
    for k in W_FIELDS:
      v = res[k][0]
      if k == "w_found":
        row[k] = bool(v)
      elif k in ("w_gap", "w_right_age"):
        row[k] = int(v)
      else:
        row[k] = float(v)
    return row

  def _w_bottom_batch(self, close_2d: np.ndarray, low_2d: np.ndarray) -> dict:
    """W 底特征的批量版本: (n_objs, bars) 一次算完

    语义与逐股版本完全相同: 取窗口内最近的两次局部低点作为左底/右底,
    左底取"与右底价格接近的最近低点"而不是"窗口最深低点",
    末端 trough_span 根K线因缺少右侧邻域不判为低点。
    1019 只 × 120 bar 约 5ms(逐股约 24ms)。
    """
    close_2d = np.asarray(close_2d, dtype=np.float64)
    low_2d = np.asarray(low_2d, dtype=np.float64)
    n, bars = close_2d.shape
    w = min(self.lookback, bars)
    close = close_2d[:, bars - w:]
    low = low_2d[:, bars - w:]

    out = {k: np.full(n, np.nan) for k in W_FIELDS}
    out["w_found"] = np.zeros(n, dtype=bool)
    out["w_gap"] = np.full(n, -1, dtype=np.int64)
    out["w_right_age"] = np.full(n, -1, dtype=np.int64)
    if n == 0 or w < 3:
      return out

    filled = np.where(np.isfinite(low) & (low > 0), low, np.inf)
    span = max(1, self.trough_span)
    # 局部低点: 不高于左右各 span 根K线的最低价
    is_low = np.ones((n, w), dtype=bool)
    for d in range(1, span + 1):
      if d >= w:
        break
      is_low[:, d:] &= filled[:, d:] <= filled[:, :-d]
      is_low[:, :-d] &= filled[:, :-d] <= filled[:, d:]
    is_low &= np.isfinite(filled) & (filled < np.inf)
    # 末端 span 根K线缺少右侧邻域, 不能确认是低点(否则持续下跌的最后一根会被当成低点)
    if w > span:
      is_low[:, w - span:] = False

    n_lows = is_low.sum(axis=1)
    rows = np.arange(n)
    right = w - 1 - np.argmax(is_low[:, ::-1], axis=1)
    l2 = low[rows, right]

    # 左底: 间隔在 [min_gap, max_gap] 内、且与右底高度接近的最近一个低点
    ar = np.arange(w)[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
      near = np.abs(low / l2[:, None] - 1) * 100 <= self.tol_pct
    cand = (
      is_low
      & (n_lows >= 2)[:, None]
      & (ar <= right[:, None] - self.min_gap)
      & (ar >= right[:, None] - self.max_gap)
      & near
    )
    left = np.where(cand, ar, -1).max(axis=1)
    l1 = low[rows, np.clip(left, 0, w - 1)]

    ok = (left >= 0) & (n_lows >= 2) & np.isfinite(close[:, -1])
    ok &= np.isfinite(l2) & (l2 > 0) & np.isfinite(l1) & (l1 > 0)

    # 颈线 = 两底之间的最高收盘; 区间变长, 逐股取 max(每行几十个元素, 可忽略)
    neck = np.full(n, np.nan)
    for i in np.flatnonzero(ok):
      seg = close[i, left[i]:right[i] + 1]
      if np.isfinite(seg).any():
        neck[i] = np.nanmax(seg)

    llv = np.where(np.isfinite(low), low, np.inf).min(axis=1)
    base = np.minimum(l1, l2)
    ok &= np.isfinite(neck) & (neck > 0) & (llv > 0) & (llv < np.inf) & (base > 0)

    last = close[:, -1]
    with np.errstate(invalid="ignore", divide="ignore"):
      out["w_diff_pct"] = np.where(ok, (l2 / l1 - 1) * 100, np.nan)
      out["w_neck_pct"] = np.where(ok, (neck / base - 1) * 100, np.nan)
      out["w_near_low_pct"] = np.where(ok, (base / llv - 1) * 100, np.nan)
      out["w_from_low_pct"] = np.where(ok, (last / l2 - 1) * 100, np.nan)
      out["w_break_pct"] = np.where(ok, (last / neck - 1) * 100, np.nan)
    out["w_found"] = ok
    out["w_gap"] = np.where(ok, right - left, -1)
    out["w_right_age"] = np.where(ok, w - 1 - right, -1)
    out["w_l1"] = np.where(ok, l1, np.nan)
    out["w_l2"] = np.where(ok, l2, np.nan)
    return out

  # ------------------------------------------------------------------ 选股步骤

  def step00(self, stocks: list[str]) -> list[str]:
    """基础股票池: 默认全部A股(剔除ST), 剔除B股与北交所"""
    if len(stocks) == 0:
      picked = sorted(A_STOCKS_EXCLUDE_ST)
    else:
      picked = sorted(set(stocks))

    picked = [
      s for s in picked
      if not (s.startswith("SH900") or s.startswith("SZ200") or s.startswith("BJ"))
    ]
    self.factors = (
      pl.DataFrame({"obj": picked})
      .join(ALL_STOCKS.select(["obj", "name"]).rename({"obj": "obj"}), on="obj", how="left")
    )
    logger.info(f"step00 universe: {len(picked)}")
    return picked

  def step01(self, stocks: list[str]) -> list[str]:
    """业绩成长: 归母净利润增长率 > min_growth, 且财报不过期"""
    g = self._growth_factors(stocks)
    if g.height == 0:
      self.factors = self.factors.head(0)
      logger.info("step01 after net profit growth: 0")
      return []

    newest = g["report_ts"].max()
    g = g.with_columns(
      (newest - pl.col("report_ts")).dt.total_days().alias("report_lag_days")
    )

    alive = g.filter(
      # polars 中 NaN 参与比较时被视为"最大", 必须显式剔除(polars 的 NaN 不是 null)
      pl.col("np_yoy_pct").is_not_null()
      & ~pl.col("np_yoy_pct").is_nan()
      & (pl.col("np_yoy_pct") > self.min_growth)
      & (pl.col("report_lag_days") <= self.max_report_lag)
    ).get_column("obj").to_list()

    self.factors = self.factors.join(
      g.select(["obj", "np_yoy_pct", "np_cur", "np_prev", "report_lag_days"]),
      on="obj", how="left",
    ).filter(pl.col("obj").is_in(alive))
    logger.info(f"step01 after net profit growth>{self.min_growth}%: {len(alive)}")
    return alive

  def step02(self, stocks: list[str]) -> list[str]:
    """规模与流动性: 剔除停牌/无成交, 成交额与总市值下限"""
    if len(stocks) == 0:
      logger.info("step02 after size/liquidity: 0")
      return []
    snap = self._snapshots(stocks).select(["obj", "close", "amount", "total_shares"])
    snap = snap.with_columns([
      (pl.col("close") * pl.col("total_shares") / 1e8).alias("mktcap_yi"),
      (pl.col("amount") / 1e8).alias("amount_yi"),
    ])

    alive = snap.filter(
      pl.col("close").is_finite()
      & (pl.col("close") > 0)
      & pl.col("total_shares").is_finite()
      & (pl.col("total_shares") > 0)
      & pl.col("amount").is_finite()
      & (pl.col("amount") >= self.min_amount)
      & (pl.col("close") * pl.col("total_shares") >= self.min_market_cap)
    ).get_column("obj").to_list()

    self.factors = self.factors.join(
      snap.select(["obj", "mktcap_yi", "amount_yi"]), on="obj", how="left"
    ).filter(pl.col("obj").is_in(alive))
    logger.info(f"step02 after size/liquidity: {len(alive)}")
    return alive

  def step03(self, stocks: list[str]) -> list[str]:
    """交易清淡 + 超跌: 近5日日均换手率 < max_turnover 且回撤 >= min_drawdown"""
    k = self._kline_factors(stocks)
    df = self.factors.join(k, on="obj", how="left")

    alive = df.filter(
      (pl.col("valid_bars") >= self.min_bars)
      & pl.col("turnover_5d").is_finite()
      & (pl.col("turnover_5d") < self.max_turnover)
      & pl.col("dd_pct").is_finite()
      & (pl.col("dd_pct") <= -self.min_drawdown)
    ).get_column("obj").to_list()

    self.factors = df.filter(pl.col("obj").is_in(alive))
    logger.info(
      f"step03 after turnover<{self.max_turnover}% and drawdown>={self.min_drawdown}%: {len(alive)}"
    )
    return alive

  def step04(self, stocks: list[str]) -> list[str]:
    """W 底形态: 双底相近、中间有反弹、贴近窗口低位、右底新鲜且已初步转强"""
    df = self.factors

    cond = (
      pl.col("w_found")
      & (pl.col("w_gap") >= self.min_gap)
      & (pl.col("w_gap") <= self.max_gap)
      & (pl.col("w_right_age") >= 0)
      & (pl.col("w_right_age") <= self.recent_days)
      & (pl.col("w_diff_pct").abs() <= self.tol_pct)
      & (pl.col("w_neck_pct") >= self.neck_pct)
      & (pl.col("w_neck_pct") <= self.max_neck_pct)
      & (pl.col("w_near_low_pct") <= self.low_tol_pct)
      & (pl.col("w_from_low_pct") >= self.rebound_min)
      & (pl.col("w_break_pct") <= self.max_break_pct)
    )
    if self.require_ma_up:
      cond = cond & (pl.col("close") >= pl.col("ma5")) & (pl.col("ma5") >= pl.col("ma10"))

    alive = df.filter(cond).get_column("obj").to_list()

    self.factors = df.filter(pl.col("obj").is_in(alive))
    logger.info(f"step04 after W bottom: {len(alive)}")
    return alive

  def step05(self, stocks: list[str]) -> list[str]:
    """排序: 右底越新鲜、距右底涨幅越小(越贴近底部)越优, 取前 top_n 只"""
    df = (
      self.factors
      .sort(["w_right_age", "w_from_low_pct"], descending=[False, False])
      .head(self.top_n)
    )
    self.factors = df
    picked = df.get_column("obj").to_list()
    logger.info(f"step05 final: {len(picked)}")
    return picked
