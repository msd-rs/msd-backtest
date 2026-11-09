import mbt
import numpy as np
import alpha as al


class Strategy(mbt.Strategy):
  """
  德瑞曼逆向价值与高成长策略 (Dreman Contrarian Value & Growth Strategy)

  策略哲学与理论背景：
  本策略基于华尔街著名逆向投资大师大卫·德瑞曼 (David Dreman) 在经典著作《逆向投资策略》中的核心理念：
  “业绩为王，逆向低估”。

  市场投资者往往受情绪与跟风驱动，给高热门股票过高估值溢价，而冷落、低估被忽视的稳健企业。
  然而单纯挑选低估值容易落入“低估值陷阱 (Value Trap)”。
  因此德瑞曼强调选股的核心铁律：
  1. 必须有强劲扎实的真实业绩支撑（净利润同比大增、净资产收益率健康）；
  2. 估值处于合理低估区间（市盈率 PE 处于中低水平，具有坚实的安全边际）；
  3. 右侧趋势突破入场，享受“估值修复 + 业绩成长”双重戴维斯双击（Davis Double Play）。

  核心选股与交易规则：
  1. 业绩为王 (Earnings Growth & Quality)：
     - 净利润同比大增：f197 (净利润同比增长率) >= min_profit_growth (默认 15.0%)
     - 资本回报稳健：f205 (年化净资产收益率 ROE) >= min_roe (默认 5.0%)，杜绝偶发性微利扭亏
  2. 估值低估 (Low Valuation)：
     - 市盈率 (P/E) 处于合理低估区间：0 < PE <= max_pe (默认 25.0)，排除高估泡沫股与亏损股
  3. 技术面右侧顺势介入 (Right-Side Trend Timing)：
     - 价格向上穿越 20 日均线 (CROSS(close, MA20)) 或 5 日均线上穿 20 日均线 (CROSS(MA5, MA20))，
       或价格处于多头趋势中 (close > MA20 且 MA5 >= MA20)
     - 价格运行在中长期生命线 60 日均线上方 (close > MA60)，确保右侧顺势进场，避免左侧接飞刀
  4. 估值修复与动态风控出场 (Exit & Risk Management)：
     - 估值修复到位/过热：PE > pe_exit (默认 40.0)，戴维斯双击兑现后获利了结
     - 基本面转弱：净利润同比增速转负 (f197 < 0.0)
     - 趋势走坏破位：收盘价跌破 60 日生命线 (close < MA60)
     - 硬止损：单笔持仓浮亏达到 stop_loss_pct (默认 -8.0%)
     - 移动止盈 (Trailing Stop)：持仓最高盈利达到 trailing_trigger (默认 10.0%) 后，
       若从持仓最高点回撤超过 trailing_stop_pct (默认 8.0%)，触发移动止盈锁定胜果
     - 冷却周期保护：平仓或止损后设定 cooldown_bars (默认 5 根 K 线)，避免震荡市频繁反复磨损。
  """

  def __init__(
    self,
    *args,
    max_pe: float = 25.0,
    min_profit_growth: float = 15.0,
    min_roe: float = 5.0,
    pe_exit: float = 40.0,
    stop_loss_pct: float = 0.08,
    trailing_trigger: float = 0.10,
    trailing_stop_pct: float = 0.08,
    cooldown_bars: int = 5,
    **kwargs,
  ):
    super().__init__()
    params = {
      "max_pe": max_pe,
      "min_profit_growth": min_profit_growth,
      "min_roe": min_roe,
      "pe_exit": pe_exit,
      "stop_loss_pct": stop_loss_pct,
      "trailing_trigger": trailing_trigger,
      "trailing_stop_pct": trailing_stop_pct,
      "cooldown_bars": cooldown_bars,
    }

    param_keys = list(params.keys())
    for i, a in enumerate(args):
      if isinstance(a, str) and "=" in a:
        k, v = a.split("=", 1)
        k = k.strip()
        if k in params:
          params[k] = float(v.strip())
      elif i < len(param_keys):
        params[param_keys[i]] = float(a)

    for k, v in kwargs.items():
      if k in params:
        params[k] = float(v)

    self.max_pe = params["max_pe"]
    self.min_profit_growth = params["min_profit_growth"]
    self.min_roe = params["min_roe"]
    self.pe_exit = params["pe_exit"]
    self.stop_loss_pct = params["stop_loss_pct"]
    self.trailing_trigger = params["trailing_trigger"]
    self.trailing_stop_pct = params["trailing_stop_pct"]
    self.cooldown_bars = int(params["cooldown_bars"])

    # 预计算变量
    self.buy_signals = None
    self.static_exit = None
    self.peak_rors = None
    self.last_sell_bars = None

  def prepare(self, ctx: mbt.Context):
    """
    预计算静态技术指标与基本面估值因子。
    在 prepare 阶段将 alpha-lib 上下文扩展至完整区间 [0, bars)，以 O(N * T) 复杂度批量执行。
    """
    with al.Context(groups=ctx.dp.groups, flags=al.FLAG_SKIP_NAN, start=0, end=ctx.dp.bars):
      close = ctx.data("close")
      price = ctx.data("price")

      # 1. 均线系统
      ma20 = al.MA(close, 20)
      ma60 = al.MA(close, 60)
      ma5 = al.MA(close, 5)

      # 2. 财务估值与业绩指标
      # f003: 每股净资产 (BPS)
      # f205: 年化净资产收益率 (ROE, %)
      # f197: 净利润同比增长率 (%)
      f003 = ctx.data("f003") if "f003" in ctx.dp.data else np.full_like(close, np.nan)
      f205 = ctx.data("f205") if "f205" in ctx.dp.data else np.full_like(close, np.nan)
      f197 = ctx.data("f197") if "f197" in ctx.dp.data else np.full_like(close, np.nan)

      # 计算市净率 PB 与市盈率 PE (PE = PB / (ROE / 100))
      pb = np.where(f003 > 0, price / np.where(f003 > 0, f003, 1.0), np.nan)
      roe_ratio = f205 / 100.0
      pe = np.where((roe_ratio > 0) & (pb > 0), pb / np.where(roe_ratio > 0, roe_ratio, 1.0), np.nan)

      # 3. 德瑞曼核心基本面条件：低市盈率 + 净利润同比大增 + 稳健 ROE
      val_undervalued = (pe > 0.0) & (pe <= self.max_pe)
      profit_strong = f197 >= self.min_profit_growth
      roe_healthy = f205 >= self.min_roe
      fundamental_buy = val_undervalued & profit_strong & roe_healthy

      # 4. 技术面右侧趋势确认：突破 20 日均线、MA5 上穿 MA20 或多头趋势，且位于 60 日均线上方
      cross_ma20 = al.CROSS(close, ma20) > 0
      cross_ma5 = al.CROSS(ma5, ma20) > 0
      trend_up = (close > ma20) & (ma5 >= ma20)
      technical_buy = (cross_ma20 | cross_ma5 | trend_up) & (close > ma60)

      # 综合买入信号
      self.buy_signals = fundamental_buy & technical_buy

      # 5. 静态卖出条件：估值过热、业绩转弱、均线破位
      pe_overvalued = pe > self.pe_exit
      profit_deteriorating = f197 < 0.0
      trend_broken = close < ma60
      self.static_exit = pe_overvalued | profit_deteriorating | trend_broken

      # 初始化持仓最高收益率跟踪数组与上次卖出 bar 索引
      self.peak_rors = np.full(ctx.dp.groups, np.nan)
      self.last_sell_bars = np.full(ctx.dp.groups, -999, dtype=np.int32)

  def execute(self, ctx: mbt.Context):
    """
    逐 bar 执行买卖决策与动态账户风控。
    """
    idx = ctx._group_offsets + ctx.dp.i

    # 获取当前各组账户的持仓状态与持仓盈亏比例
    if hasattr(ctx, "accounts"):
      is_holding = np.array([acc.position.shares > 0 for acc in ctx.accounts], dtype=bool)
      ror_hold = np.array([acc.ror_hold[ctx.dp.i] for acc in ctx.accounts], dtype=np.float64)
    elif hasattr(ctx, "account") and hasattr(ctx.account, "positions"):
      is_holding = np.asarray(ctx.account.positions > 0, dtype=bool)
      ror_hold = np.asarray(ctx.account.ror_hold, dtype=np.float64) if hasattr(ctx.account, "ror_hold") else np.zeros(len(is_holding))
    else:
      is_holding = np.zeros(ctx.dp.groups, dtype=bool)
      ror_hold = np.zeros(ctx.dp.groups, dtype=np.float64)

    # 更新持仓最高收益率，已平仓标的重置为 NaN
    self.peak_rors = np.where(
      is_holding,
      np.fmax(np.where(np.isnan(self.peak_rors), -np.inf, self.peak_rors), ror_hold),
      np.nan,
    )

    # 1. 动态卖出风控
    # 固定止损
    stop_loss = is_holding & (ror_hold <= -self.stop_loss_pct)

    # 移动止盈：持仓收益率达到 trailing_trigger 后，从高点回撤 trailing_stop_pct 触发
    trailing_stop = (
      is_holding
      & (self.peak_rors >= self.trailing_trigger)
      & (((1.0 + ror_hold) / (1.0 + self.peak_rors)) <= (1.0 - self.trailing_stop_pct))
    )

    # 基本面/趋势出场信号
    exit_cond = is_holding & self.static_exit[idx]

    sell_signals = is_holding & (stop_loss | trailing_stop | exit_cond)

    if np.any(sell_signals):
      self.last_sell_bars = np.where(sell_signals, ctx.dp.i, self.last_sell_bars)

    # 2. 买入信号：满足德瑞曼业绩与低估值右侧信号，未持仓且不在平仓冷却期内
    in_cooldown = (ctx.dp.i - self.last_sell_bars) <= self.cooldown_bars
    buy_signals = self.buy_signals[idx] & (~is_holding) & (~in_cooldown)

    # 3. 提交交易委托
    ctx.sell(sell_signals, 1.0)
    ctx.buy(buy_signals, 1.0)
