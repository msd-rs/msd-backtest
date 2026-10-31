"""Panel: 二维面板视图, 让一维连续缓冲区自带 symbols 元数据"""

from __future__ import annotations

import contextlib
from typing import Iterator

import alpha as al
import numpy as np
import polars as pl


class Panel:
  """行情/因子面板

  `SelectorDataProvider.load_kline` 返回的是一维连续缓冲区(块布局), 本对象把它
  包装成 (n_symbols, bars) 的二维视图, 并**始终携带 symbols 元数据**: 行号与股票
  代码绑定, 调用方不再需要用入参顺序去猜行序。

  内部布局: block i 占用 [i*bars, (i+1)*bars), 对应 symbols[i]。

  Attributes:
    frame: 底层 pl.DataFrame, 每列长度都等于 n_symbols * bars
    symbols: 行序对应的股票代码(实现可能重排, 以本属性为准)
    bars: 每只股票的K线根数

  兼容旧签名(仍可解包):
    >>> data, symbols, bars = dp.load_kline(symbols, 250)
  """

  __slots__ = ("frame", "symbols", "bars", "_index")

  def __init__(self, frame: pl.DataFrame, symbols: list[str], bars: int):
    n = len(symbols)
    if bars <= 0:
      raise ValueError("bars must be positive")
    if n * bars != len(frame):
      raise ValueError(f"layout mismatch: {n} symbols x {bars} bars != {len(frame)} rows")
    self.frame = frame
    self.symbols = list(symbols)
    self.bars = int(bars)
    self._index = {s: i for i, s in enumerate(self.symbols)}

  # ---------------------------------------------------------------- 元信息

  @property
  def n_symbols(self) -> int:
    return len(self.symbols)

  def __len__(self) -> int:
    return self.n_symbols

  def __repr__(self) -> str:
    return f"Panel(n_symbols={self.n_symbols}, bars={self.bars}, columns={self.frame.columns})"

  def __iter__(self) -> Iterator:
    """兼容旧的三元组解包: data, symbols, bars = dp.load_kline(...)"""
    return iter((self.frame, self.symbols, self.bars))

  def pick(self, symbol: str) -> int:
    """股票代码 -> 块行号; 不存在时抛 KeyError 而不是静默返回错误下标"""
    return self._index[symbol]

  # ---------------------------------------------------------------- 二维视图

  def buffer(self, name: str) -> np.ndarray:
    """一维连续缓冲区(块布局), 即 alpha-lib 直接消费的形态"""
    return self.frame.get_column(name).to_numpy()

  def column(self, name: str) -> np.ndarray:
    """(n_symbols, bars) 二维视图; 与 buffer() 共享同一块内存"""
    return self.buffer(name).reshape(self.n_symbols, self.bars)

  def last(self, name: str) -> np.ndarray:
    """每只股票最后一根的值, 形状 (n_symbols,)"""
    return self.column(name)[:, -1]

  def offset(self, name: str, k: int) -> np.ndarray:
    """每只股票第 k 根的值(k 可为负), 形状 (n_symbols,)"""
    return self.column(name)[:, k]

  def timeline(self) -> np.ndarray:
    """日期轴 (bars,), 由 asof 对齐基准决定; 全市场共享同一条时间轴"""
    return self.frame.get_column("ts").to_numpy()[: self.bars]

  # ---------------------------------------------------------------- alpha 上下文

  def set_ctx(self, flags: int | None = None) -> None:
    """设置 alpha-lib 分组上下文

    groups 恒等于 n_symbols, 由本对象保证, 避免调用方手写 groups 时错配
    (错配不会报错, 只会把相邻股票串在一起计算)。
    """
    al.set_ctx(
      groups=self.n_symbols,
      flags=al.FLAG_SKIP_NAN if flags is None else flags,
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
    return np.asarray(out).reshape(self.n_symbols, self.bars)

  # ---------------------------------------------------------------- 截面与导出

  def cross_section(self, **columns) -> pl.DataFrame:
    """把若干 (n_symbols,) 因子装配成 n 行截面表 —— 选股器真正需要的形态

    >>> panel.cross_section(close=panel.last("close"), vol_ratio=vr)
    """
    data: dict[str, object] = {"obj": self.symbols}
    for name, value in columns.items():
      arr = np.asarray(value)
      if arr.shape != (self.n_symbols,):
        raise ValueError(
          f"column '{name}' must have shape ({self.n_symbols},), got {arr.shape}"
        )
      data[name] = arr
    return pl.DataFrame(data)

  def to_frame(self, key: str = "enum") -> pl.DataFrame:
    """展开成长表, 仅用于调试/导出; 计算路径请用 column()/last()/cross_section()

    5000 只 x 1000 根 = 500 万行, symbol 列的承载方式(key):
      "enum": pl.Enum(symbols), 字典编码约 2 字节/行; 字典即股票列表本身,
              不需要额外维护 symbol<->id 映射。默认项。
      "str" : 普通字符串列, 约 38MB / 500 万行, 仅在需要对外交付时使用。
    """
    if key not in ("enum", "str"):
      raise ValueError("key must be 'enum' or 'str'")
    sym = pl.Series("symbol", np.repeat(np.asarray(self.symbols, dtype=object), self.bars))
    if key == "enum":
      sym = sym.cast(pl.Enum(self.symbols))
    return pl.DataFrame({"symbol": sym}).hstack(self.frame)
