"""Panel: 二维面板视图, 让一维连续缓冲区自带 objs 元数据"""

from __future__ import annotations

import contextlib
from typing import Iterator

import alpha as al
import numpy as np
import polars as pl


class Panel:
  """行情/因子面板

  `SelectorDataProvider.load_kline` 返回的是一维连续缓冲区(块布局), 本对象把它
  包装成 (n_objs, bars) 的二维视图, 并**始终携带 objs 元数据**: 行号与股票
  代码绑定, 调用方不再需要用入参顺序去猜行序。

  内部布局: block i 占用 [i*bars, (i+1)*bars), 对应 objs[i]。

  Attributes:
    frame: 底层 pl.DataFrame, 每列长度都等于 n_objs * bars
    objs: 行序对应的股票代码(实现可能重排, 以本属性为准)
    bars: 每只股票的K线根数

  兼容旧签名(仍可解包):
    >>> data, objs, bars = dp.load_kline(objs, 250)
  """

  __slots__ = ("frame", "objs", "bars", "_index")

  def __init__(self, frame: pl.DataFrame, objs: list[str], bars: int):
    n = len(objs)
    if bars <= 0:
      raise ValueError("bars must be positive")
    if n * bars != len(frame):
      raise ValueError(f"layout mismatch: {n} objs x {bars} bars != {len(frame)} rows")
    self.frame = frame
    self.objs = list(objs)
    self.bars = bars
    self._index = {s: i for i, s in enumerate(self.objs)}

  # ---------------------------------------------------------------- 元信息

  @property
  def n_objs(self) -> int:
    return len(self.objs)

  def __len__(self) -> int:
    return self.n_objs

  def __repr__(self) -> str:
    return f"Panel(n_objs={self.n_objs}, bars={self.bars}, columns={self.frame.columns})"

  def __iter__(self) -> Iterator:
    """兼容旧的三元组解包: data, objs, bars = dp.load_kline(...)"""
    return iter((self.frame, self.objs, self.bars))

  def pick(self, obj: str) -> int:
    """股票代码 -> 块行号; 不存在时抛 KeyError 而不是静默返回错误下标"""
    return self._index[obj]

  # ---------------------------------------------------------------- 二维视图

  def buffer(self, name: str) -> np.ndarray:
    """一维连续缓冲区(块布局), 即 alpha-lib 直接消费的形态"""
    return self.frame.get_column(name).to_numpy()

  def column(self, name: str) -> np.ndarray:
    """(n_objs, bars) 二维视图; 与 buffer() 共享同一块内存"""
    return self.buffer(name).reshape(self.n_objs, self.bars)

  def last(self, name: str) -> np.ndarray:
    """每只股票最后一根的值, 形状 (n_objs,)"""
    return self.column(name)[:, -1]

  def offset(self, name: str, k: int) -> np.ndarray:
    """每只股票第 k 根的值(k 可为负), 形状 (n_objs,)"""
    return self.column(name)[:, k]

  def timeline(self) -> np.ndarray:
    """日期轴 (bars,), 由 asof 对齐基准决定; 全市场共享同一条时间轴"""
    return self.frame.get_column("ts").to_numpy()[: self.bars]

  # ---------------------------------------------------------------- alpha 上下文

  def set_ctx(self, flags: int | None = None) -> None:
    """设置 alpha-lib 分组上下文

    groups 恒等于 n_objs, 由本对象保证, 避免调用方手写 groups 时错配
    (错配不会报错, 只会把相邻股票串在一起计算)。
    """
    al.set_ctx(
      groups=self.n_objs,
      flags=al.FLAG_SKIP_NAN if flags is None else flags,
      start=0,
      end=0,
    )

  @contextlib.contextmanager
  def ctx(self, flags: int | None = None):
    """with panel.ctx(): 把"设置上下文"与"调用 alpha 函数"绑定在同一作用域

    alpha-lib 的分组信息是进程级全局状态, 而 load_kline 会改写它, 因此每次
    计算前都应显式落在这个上下文里:

    >>> with panel.ctx():
    ...   ma = al.MA(panel.column("close"), 60)
    """
    self.set_ctx(flags)
    yield self

  def compute(self, fn, *args, **kwargs) -> np.ndarray:
    """在正确的分组上下文里调用 alpha 函数, 并把结果还原成二维视图

    参数中的列名字符串会自动解析为对应的 buffer, 因此不必关心一维/二维转换:

    >>> div_ttm = panel.compute(al.SUM, "dividend", 250)[:, -1]
    >>> ma250 = panel.compute(al.MA, "close", 250)[:, -1]

    适用于逐元素/滚动类函数(输入输出等长); 横截面类函数(CC_RANK 等)同样可用,
    但请确认其输出长度与输入一致。
    """
    def resolve(a):
      if isinstance(a, str) and a in self.frame.columns:
        return self.buffer(a)
      return a

    args = tuple(resolve(a) for a in args)
    kwargs = {k: resolve(v) for k, v in kwargs.items()}
    with self.ctx():
      out = fn(*args, **kwargs)
    return np.asarray(out).reshape(self.n_objs, self.bars)

  # ---------------------------------------------------------------- 截面与导出

  def resample(self, freq: str = "1M") -> Panel:
    """多周期重采样: 将日K线面板聚合为周K线、月K线、季K线或年K线面板

    返回一个完整的 Panel 对象 (n_objs, new_bars), 保留底层所有列, 包含正确的
    OHLCV 聚合与状态列继承。

    Args:
      freq: 重采样周期。支持:
        - "W" / "1W" / "week" / "weekly": 周线 (周一至周日为一个周期)
        - "M" / "1M" / "month" / "monthly": 月线 (自然月)
        - "Q" / "1Q" / "quarter" / "quarterly": 季线 (自然季)
        - "Y" / "1Y" / "year" / "yearly": 年线 (自然年)
        - 或其他 Polars dt.truncate 支持的字符串 (如 "1w", "1mo", "1q", "1y")

    Returns:
      新的 Panel 对象, 每只股票包含 new_bars 根重采样后的 K 线。
    """
    if "ts" not in self.frame.columns:
      raise ValueError("cannot resample Panel without 'ts' column")

    freq_norm = freq.strip().lower()
    mapping = {
      "w": "1w",
      "1w": "1w",
      "week": "1w",
      "weekly": "1w",
      "m": "1mo",
      "1m": "1mo",
      "month": "1mo",
      "monthly": "1mo",
      "q": "1q",
      "1q": "1q",
      "quarter": "1q",
      "quarterly": "1q",
      "y": "1y",
      "1y": "1y",
      "year": "1y",
      "yearly": "1y",
    }
    rule = mapping.get(freq_norm, freq_norm)

    tl_series = pl.Series("ts", self.timeline())
    try:
      truncated = tl_series.dt.truncate(rule).to_numpy()
    except Exception as e:
      raise ValueError(f"invalid resampling frequency '{freq}': {e}") from e

    _, first_indices = np.unique(truncated, return_index=True)
    starts = np.sort(first_indices)
    ends = np.append(starts[1:], len(truncated))
    m_bars = len(starts)
    n_sym = self.n_objs

    # 重采样面板的时间轴取每个周期的最后一个交易日
    timeline_arr = np.array([self.timeline()[e - 1] for e in ends])
    new_cols: dict[str, pl.Series] = {
      "ts": pl.Series("ts", np.tile(timeline_arr, n_sym))
    }

    sum_cols = {"volume", "amount", "dividend", "transfer_shares", "right_shares"}

    for name in self.frame.columns:
      if name == "ts":
        continue
      col_series = self.frame.get_column(name)
      dtype = col_series.dtype

      arr_2d = self.column(name)
      res_2d = np.empty((n_sym, m_bars), dtype=arr_2d.dtype)
      is_float = np.issubdtype(arr_2d.dtype, np.floating)

      for k, (s, e) in enumerate(zip(starts, ends)):
        sub = arr_2d[:, s:e]
        valid = ~np.isnan(sub) if is_float else np.ones(sub.shape, dtype=bool)
        has_valid = valid.any(axis=1)

        if name == "open":
          idx = np.where(has_valid, valid.argmax(axis=1), 0)
          res_2d[:, k] = np.where(has_valid, sub[np.arange(n_sym), idx], np.nan if is_float else 0)
        elif name == "close":
          idx = (e - s) - 1 - np.where(has_valid, valid[:, ::-1].argmax(axis=1), 0)
          res_2d[:, k] = np.where(has_valid, sub[np.arange(n_sym), idx], np.nan if is_float else 0)
        elif name == "high":
          if is_float:
            res_2d[:, k] = np.where(has_valid, np.nanmax(np.where(valid, sub, -np.inf), axis=1), np.nan)
          else:
            res_2d[:, k] = np.max(sub, axis=1)
        elif name == "low":
          if is_float:
            res_2d[:, k] = np.where(has_valid, np.nanmin(np.where(valid, sub, np.inf), axis=1), np.nan)
          else:
            res_2d[:, k] = np.min(sub, axis=1)
        elif name in sum_cols:
          if is_float:
            res_2d[:, k] = np.where(has_valid, np.nansum(sub, axis=1), 0.0)
          else:
            res_2d[:, k] = np.sum(sub, axis=1)
        else:
          # 其他列(如股本、财务状态等)默认继承该周期最后一个有效值
          idx = (e - s) - 1 - np.where(has_valid, valid[:, ::-1].argmax(axis=1), 0)
          res_2d[:, k] = np.where(has_valid, sub[np.arange(n_sym), idx], np.nan if is_float else 0)

      new_cols[name] = pl.Series(name, res_2d.ravel(), dtype=dtype)

    new_frame = pl.DataFrame(new_cols)
    return Panel(new_frame, self.objs, m_bars)

  def cross_section(self, **columns) -> pl.DataFrame:
    """把若干 (n_objs,) 因子装配成 n 行截面表 —— 选股器真正需要的形态

    >>> panel.cross_section(close=panel.last("close"), vol_ratio=vr)
    """
    data: dict[str, object] = {"obj": self.objs}
    for name, value in columns.items():
      arr = np.asarray(value)
      if arr.shape != (self.n_objs,):
        raise ValueError(
          f"column '{name}' must have shape ({self.n_objs},), got {arr.shape}"
        )
      data[name] = arr
    return pl.DataFrame(data)

  def to_frame(self, key: str = "enum") -> pl.DataFrame:
    """展开成长表, 仅用于调试/导出; 计算路径请用 column()/last()/cross_section()

    5000 只 x 1000 根 = 500 万行, obj 列的承载方式(key):
      "enum": pl.Enum(objs), 字典编码约 2 字节/行; 字典即股票列表本身,
              不需要额外维护 obj<->id 映射。默认项。
      "str" : 普通字符串列, 约 38MB / 500 万行, 仅在需要对外交付时使用。
    """
    if key not in ("enum", "str"):
      raise ValueError("key must be 'enum' or 'str'")
    sym = pl.Series("obj", np.repeat(np.asarray(self.objs, dtype=object), self.bars))
    if key == "enum":
      sym = sym.cast(pl.Enum(self.objs))
    return pl.DataFrame({"obj": sym}).hstack(self.frame)
