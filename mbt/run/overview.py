# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Modern high-DPI strategy overview chart renderer."""

from __future__ import annotations

import copy
import dataclasses
import datetime
import inspect
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyBboxPatch
import numpy as np

from mbt.run.report import Report, _format_date


# Modern Light Financial Dashboard Palette (A-share Red-Up / Green-Down)
PALETTE = {
  "bg": "#F8FAFC",
  "card": "#FFFFFF",
  "border": "#E2E8F0",
  "grid": "#F1F5F9",
  "text_primary": "#0F172A",
  "text_secondary": "#334155",
  "text_muted": "#475569",
  "strategy": "#2563EB",
  "strategy_fill": "#DBEAFE",
  "benchmark": "#94A3B8",
  "up": "#DC2626",
  "up_bg": "#FEF2F2",
  "down": "#059669",
  "down_bg": "#ECFDF5",
  "drawdown": "#EF4444",
  "drawdown_fill": "#FCA5A5",
  "rebal": "#D97706",
  "rebal_bg": "#FFFBEB",
  "rebal_border": "#F59E0B",
  "badge_bg": "#F1F5F9",
  "accent_blue_bg": "#EFF6FF",
}

_CJK_FONT_CANDIDATES = [
  "Source Han Sans SC",
  "Noto Sans CJK SC",
  "WenQuanYi Micro Hei",
  "PingFang SC",
  "Hiragino Sans GB",
  "Microsoft YaHei",
  "SimHei",
  "DejaVu Sans",
]


@dataclasses.dataclass
class RebalanceEvent:
  """Single bar rebalance summary for chart annotation and timeline."""
  bar_idx: int
  date: str
  buys: list[tuple[str, float]]
  sells: list[tuple[str, float]]
  turnover: float
  short_label: str
  detail_text: str


@dataclasses.dataclass
class StrategyMeta:
  """Display metadata for the strategy overview header and universe card."""
  title: str
  brief: str
  universe_headline: str
  universe_detail: str
  mode_label: str
  date_range: str
  benchmark_label: str
  initial_cash_label: str
  author: str = "AI Agent"


def _configure_fonts() -> str:
  """Select the best available CJK-capable sans-serif font."""
  available = {f.name for f in fm.fontManager.ttflist}
  chosen = "DejaVu Sans"
  for candidate in _CJK_FONT_CANDIDATES:
    if candidate in available:
      chosen = candidate
      break
  plt.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
  plt.rcParams["axes.unicode_minus"] = False
  return chosen


def _clean_doc(doc: str | None, max_len: int = 120) -> str:
  """Extract the first meaningful paragraph from a docstring."""
  if not doc:
    return ""
  lines = [ln.strip() for ln in inspect.cleandoc(doc).splitlines()]
  para: list[str] = []
  for ln in lines:
    if not ln:
      if para:
        break
      continue
    if ln.startswith((":", "Args:", "Returns:", "Attributes:", "Raises:")):
      break
    para.append(ln)
  text = " ".join(para).strip()
  if len(text) > max_len:
    text = text[: max_len - 1].rstrip() + "…"
  return text


def _parse_dates(date_arr: np.ndarray) -> tuple[list[datetime.date], list[str]]:
  """Convert Report.date (YYYYMMDD int, unix ts, or datetime64) to dates and strings."""
  dates: list[datetime.date] = []
  labels: list[str] = []
  for i, raw in enumerate(date_arr):
    s = _format_date(i, raw, date_arr)
    labels.append(s)
    try:
      dates.append(datetime.date.fromisoformat(s[:10]))
    except ValueError:
      dates.append(datetime.date(2020, 1, 1) + datetime.timedelta(days=i))
  return dates, labels


def _fmt_money(val: float) -> str:
  """Format RMB currency compactly."""
  if abs(val) >= 1e8:
    return f"¥{val / 1e8:.2f}亿"
  if abs(val) >= 1e4:
    return f"¥{val / 1e4:.1f}万"
  return f"¥{val:,.0f}"


def _extract_short_title(text: str, max_len: int = 18) -> str:
  """Extract a concise headline from a strategy brief or docstring."""
  if not text:
    return ""
  first = text.splitlines()[0].strip()
  for sep in ("：", ":", "。", "，", ","):
    if sep in first:
      head = first.split(sep, 1)[0].strip()
      if len(head) >= 2:
        first = head
        break
  if len(first) > max_len:
    first = first[: max_len - 1].rstrip() + "…"
  return first


def extract_strategy_meta(
  report: Report,
  strategy: Any = None,
  selector: Any = None,
  account: Any = None,
  req: Any = None,
  is_portfolio: bool = False,
  multi_count: int = 1,
  title: str | None = None,
  brief: str | None = None,
  universe: str | None = None,
) -> StrategyMeta:
  """Extract strategy title, brief, and universe scope from runtime objects."""
  diag = getattr(report, "diagnostics", None) or {}
  strat_mod = getattr(strategy, "_source_module", None) or (
    inspect.getmodule(strategy) if strategy is not None else None
  )
  mod_doc = _clean_doc(getattr(strat_mod, "__doc__", None)) if strat_mod is not None else ""

  resolved_brief = (
    brief
    or diag.get("overview_brief")
    or getattr(strategy, "brief", None)
    or getattr(strategy, "description", None)
    or _clean_doc(getattr(strategy, "__doc__", None))
    or mod_doc
    or _clean_doc(getattr(selector, "__doc__", None))
    or "基于 msd-backtest 量化框架执行的策略回测表现与调仓追踪。"
  )

  brief_headline = _extract_short_title(
    getattr(strategy, "brief", None)
    or getattr(strategy, "description", None)
    or _clean_doc(getattr(strategy, "__doc__", None))
    or mod_doc
    or ""
  )

  resolved_title = (
    title
    or diag.get("overview_title")
    or getattr(strategy, "title", None)
    or getattr(strategy, "name", None)
    or brief_headline
    or (strategy.__class__.__name__ if strategy is not None else None)
    or (f"标的回测 · {report.obj}" if report.obj and report.obj != "PORTFOLIO" else "量化策略回测")
  )

  pool_size = getattr(report, "pool_size", 0) or (len(getattr(req, "objs", [])) if req else 0)
  traded_count = getattr(report, "traded_objs_count", 0)

  if universe or diag.get("overview_universe") or getattr(strategy, "universe", None):
    u_text = str(universe or diag.get("overview_universe") or getattr(strategy, "universe", ""))
    universe_headline = u_text
    universe_detail = f"交易标的 {traded_count} 只" if traded_count > 0 else "自定义选股范围"
  elif is_portfolio:
    sel_name = selector.__class__.__name__ if selector is not None else ""
    if sel_name and sel_name != "StaticSelector":
      universe_headline = f"动态选股 · 池 {pool_size} 只" if pool_size > 0 else "动态选股"
      universe_detail = f"实际交易 {traded_count} 只" if traded_count > 0 else "动态条件筛选"
    else:
      universe_headline = f"指定组合池 · {pool_size} 只" if pool_size > 0 else "组合资产池"
      universe_detail = f"实际交易 {traded_count} 只" if traded_count > 0 else "静态标的池"
  elif multi_count > 1:
    universe_headline = f"多标的独立 · {multi_count} 只"
    universe_detail = f"等权聚合视图 ({report.obj})"
  else:
    universe_headline = f"单标的 · {report.obj}"
    universe_detail = "独立标的回测"

  if is_portfolio:
    mode_label = "组合回测"
  elif multi_count > 1:
    mode_label = "多标的聚合"
  else:
    mode_label = "单标的回测"

  _, date_labels = _parse_dates(report.date) if len(report.date) > 0 else ([], [])
  date_range = f"{date_labels[0]} ~ {date_labels[-1]}" if len(date_labels) >= 2 else (date_labels[0] if date_labels else "-")

  bench = getattr(report, "benchmark_obj", "") or getattr(req, "benchmark", "") or "基准"
  init_cash = getattr(account, "initial_cash", None) or getattr(req, "cash", None) or (1000000.0 if is_portfolio else 10000.0)

  raw_author = (
    getattr(strategy, "M_AUTHOR", None)
    or (getattr(strat_mod, "M_AUTHOR", None) if strat_mod is not None else None)
    or diag.get("M_AUTHOR")
    or "AI Agent"
  )
  resolved_author = str(raw_author).strip() or "AI Agent"

  return StrategyMeta(
    title=str(resolved_title),
    brief=str(resolved_brief),
    universe_headline=universe_headline,
    universe_detail=universe_detail,
    mode_label=mode_label,
    date_range=date_range,
    benchmark_label=str(bench),
    initial_cash_label=_fmt_money(float(init_cash)),
    author=resolved_author,
  )


def extract_holding_counts(
  report: Report,
  account: Any = None,
  dp: Any = None,
) -> np.ndarray:
  """Compute daily holding stock count aligned with report.ror."""
  n_bars = len(report.ror)
  if n_bars == 0:
    return np.zeros(0, dtype=np.int32)

  w = max(0, int(getattr(dp, "warmup_bars", 0))) if dp is not None else 0
  records = getattr(account, "records", None) if account is not None else None

  if records:
    total_bars = n_bars + w
    by_bar: dict[int, list[Any]] = {}
    for rec in records:
      b = int(getattr(rec, "bar", 0))
      if 0 <= b < total_bars:
        by_bar.setdefault(b, []).append(rec)

    shares_by_obj: dict[str, float] = {}
    counts = np.zeros(total_bars, dtype=np.int32)
    for b in range(total_bars):
      for rec in by_bar.get(b, ()):
        obj = str(getattr(rec, "obj", ""))
        action = str(getattr(rec, "action", "")).upper()
        sh = float(getattr(rec, "shares", 0.0))
        if action == "BUY":
          shares_by_obj[obj] = shares_by_obj.get(obj, 0.0) + sh
        elif action == "SELL":
          rem = shares_by_obj.get(obj, 0.0) - sh
          shares_by_obj[obj] = rem if rem > 1e-6 else 0.0
      counts[b] = sum(1 for v in shares_by_obj.values() if v > 1e-6)
    return counts[w: w + n_bars]

  # Fallback to report.trades holding intervals
  trades = getattr(report, "trades", None)
  if trades:
    counts = np.zeros(n_bars, dtype=np.int32)
    for t in trades:
      eb = max(0, int(t.get("entry_bar", 0)) - w)
      xb = int(t.get("exit_bar", n_bars - 1)) - w
      xb = min(n_bars - 1, max(eb, xb))
      counts[eb : xb + 1] += 1
    return counts

  # Fallback to report.actions (1 = buy, 2 = sell)
  actions = getattr(report, "actions", None)
  if actions is not None:
    counts = np.zeros(n_bars, dtype=np.int32)
    holding = 0
    for i in range(min(n_bars, len(actions))):
      act = int(actions[i])
      if act == 1:
        holding = 1
      elif act == 2:
        holding = 0
      counts[i] = holding
    return counts

  return np.zeros(n_bars, dtype=np.int32)


def extract_holding_duration_stats(
  report: Report,
  account: Any = None,
  dp: Any = None,
) -> tuple[float, int, int]:
  """Compute (avg, max, min) holding period in trading bars per stock."""
  n_bars = len(report.ror)
  if n_bars == 0:
    return (0.0, 0, 0)

  w = max(0, int(getattr(dp, "warmup_bars", 0))) if dp is not None else 0
  total_bars = n_bars + w
  records = getattr(account, "records", None) if account is not None else None
  durations: list[int] = []

  if records:
    shares_by_obj: dict[str, float] = {}
    entry_by_obj: dict[str, int] = {}
    for rec in records:
      b = int(getattr(rec, "bar", 0))
      obj = str(getattr(rec, "obj", ""))
      action = str(getattr(rec, "action", "")).upper()
      sh = float(getattr(rec, "shares", 0.0))
      prev = shares_by_obj.get(obj, 0.0)
      if action == "BUY":
        if prev <= 1e-6 and sh > 1e-6:
          entry_by_obj[obj] = b
        shares_by_obj[obj] = prev + sh
      elif action == "SELL":
        rem = prev - sh
        if prev > 1e-6 and rem <= 1e-6:
          eb = entry_by_obj.pop(obj, b)
          durations.append(max(1, b - eb))
          shares_by_obj[obj] = 0.0
        else:
          shares_by_obj[obj] = max(0.0, rem)

    last_bar = total_bars - 1
    for obj, eb in entry_by_obj.items():
      if shares_by_obj.get(obj, 0.0) > 1e-6:
        durations.append(max(1, last_bar - eb))
  elif getattr(report, "trades", None):
    for t in report.trades:
      eb = int(t.get("entry_bar", 0))
      xb = int(t.get("exit_bar", total_bars - 1))
      durations.append(max(1, xb - eb))
  elif getattr(report, "actions", None) is not None:
    entry_i: int | None = None
    for i in range(min(n_bars, len(report.actions))):
      act = int(report.actions[i])
      if act == 1 and entry_i is None:
        entry_i = i
      elif act == 2 and entry_i is not None:
        durations.append(max(1, i - entry_i))
        entry_i = None
    if entry_i is not None:
      durations.append(max(1, (n_bars - 1) - entry_i))

  if not durations:
    return (0.0, 0, 0)
  return (float(np.mean(durations)), int(max(durations)), int(min(durations)))


def extract_rebalance_events(
  report: Report,
  account: Any = None,
  dp: Any = None,
) -> list[RebalanceEvent]:
  """Extract per-bar rebalance events from account.records or report.actions."""
  n_bars = len(report.ror)
  if n_bars == 0:
    return []

  _, date_labels = _parse_dates(report.date)
  w = max(0, int(getattr(dp, "warmup_bars", 0))) if dp is not None else 0
  records = getattr(account, "records", None) if account is not None else None

  if records:
    by_idx: dict[int, dict[str, list[tuple[str, float]]]] = {}
    for rec in records:
      idx = int(getattr(rec, "bar", 0)) - w
      if idx < 0 or idx >= n_bars:
        continue
      bucket = by_idx.setdefault(idx, {"BUY": [], "SELL": []})
      action = str(getattr(rec, "action", "")).upper()
      obj = str(getattr(rec, "obj", ""))
      amount = float(getattr(rec, "amount", 0.0))
      if action == "BUY":
        bucket["BUY"].append((obj, amount))
      elif action == "SELL":
        bucket["SELL"].append((obj, amount))

    events: list[RebalanceEvent] = []
    for idx in sorted(by_idx.keys()):
      buys = by_idx[idx]["BUY"]
      sells = by_idx[idx]["SELL"]
      if not buys and not sells:
        continue
      turnover = sum(a for _, a in buys) + sum(a for _, a in sells)
      short_label = _format_rebalance_short(buys, sells)
      date_str = date_labels[idx] if idx < len(date_labels) else ""
      detail = _format_rebalance_detail(date_str, buys, sells, turnover)
      events.append(
        RebalanceEvent(
          bar_idx=idx,
          date=date_str,
          buys=buys,
          sells=sells,
          turnover=turnover,
          short_label=short_label,
          detail_text=detail,
        )
      )
    return events

  # Fallback to report.actions (1 = buy, 2 = sell)
  events = []
  actions = getattr(report, "actions", None)
  if actions is None:
    return events
  for idx in range(min(n_bars, len(actions))):
    act = int(actions[idx])
    if act not in (1, 2):
      continue
    date_str = date_labels[idx] if idx < len(date_labels) else ""
    obj_name = report.obj if report.obj and report.obj != "PORTFOLIO" else "组合"
    if act == 1:
      buys, sells = [(obj_name, 0.0)], []
      short_label = f"+买入 {obj_name}"
    else:
      buys, sells = [], [(obj_name, 0.0)]
      short_label = f"-卖出 {obj_name}"
    events.append(
      RebalanceEvent(
        bar_idx=idx,
        date=date_str,
        buys=buys,
        sells=sells,
        turnover=1.0,
        short_label=short_label,
        detail_text=f"{date_str}  {short_label}",
      )
    )
  return events


def _format_rebalance_short(
  buys: list[tuple[str, float]],
  sells: list[tuple[str, float]],
) -> str:
  if len(buys) == 1 and len(sells) == 0:
    return f"+买 {buys[0][0]}"
  if len(sells) == 1 and len(buys) == 0:
    return f"-卖 {sells[0][0]}"
  parts = []
  if buys:
    parts.append(f"+买{len(buys)}")
  if sells:
    parts.append(f"-卖{len(sells)}")
  return " / ".join(parts)


def _format_rebalance_detail(
  date_str: str,
  buys: list[tuple[str, float]],
  sells: list[tuple[str, float]],
  turnover: float,
) -> str:
  parts = []
  if buys:
    buy_names = ",".join(o for o, _ in buys[:2]) + (f" 等{len(buys)}只" if len(buys) > 2 else "")
    parts.append(f"买入 {buy_names}")
  if sells:
    sell_names = ",".join(o for o, _ in sells[:2]) + (f" 等{len(sells)}只" if len(sells) > 2 else "")
    parts.append(f"卖出 {sell_names}")
  amt_str = f" · {_fmt_money(turnover)}" if turnover > 0 else ""
  return f"{date_str}  {' | '.join(parts)}{amt_str}"


def select_callout_events(
  events: list[RebalanceEvent],
  total_bars: int,
  max_callouts: int = 6,
) -> list[RebalanceEvent]:
  """Select key non-overlapping rebalance events for curve callout bubbles."""
  if not events or total_bars <= 0:
    return []
  if len(events) <= max_callouts:
    min_gap = max(1, int(total_bars * 0.08))
    selected: list[RebalanceEvent] = []
    for ev in events:
      if not selected or (ev.bar_idx - selected[-1].bar_idx) >= min_gap:
        selected.append(ev)
    return selected

  min_gap = max(1, int(total_bars * 0.11))
  # Always seed with first event, then prioritize by turnover
  chosen: list[RebalanceEvent] = [events[0]]
  ranked = sorted(events[1:], key=lambda e: e.turnover, reverse=True)
  for ev in ranked:
    if len(chosen) >= max_callouts:
      break
    if all(abs(ev.bar_idx - c.bar_idx) >= min_gap for c in chosen):
      chosen.append(ev)
  return sorted(chosen, key=lambda e: e.bar_idx)


def aggregate_reports(reports: list[Report]) -> Report:
  """Aggregate multiple single-stock reports into an equal-weight overview report."""
  if len(reports) == 1:
    return reports[0]
  base = reports[0]
  min_len = min(len(r.ror) for r in reports if len(r.ror) > 0)
  rors = np.array([r.ror[:min_len] for r in reports if len(r.ror) >= min_len], dtype=np.float64)
  avg_ror = np.mean(rors, axis=0) if len(rors) > 0 else np.zeros(0, dtype=np.float64)

  agg = copy.copy(base)
  agg.obj = ", ".join(r.obj for r in reports[:3]) + (f" 等{len(reports)}只" if len(reports) > 3 else "")
  agg.date = base.date[:min_len]
  agg.ror = np.round(avg_ror, 4)
  agg.profit = float(agg.ror[-1]) if len(agg.ror) > 0 else 0.0
  agg.sharpe = float(np.round(np.mean([r.sharpe for r in reports]), 4))
  agg.alpha = float(np.round(np.mean([r.alpha for r in reports]), 4))
  agg.sortino = float(np.round(np.mean([r.sortino for r in reports]), 4))
  agg.win_rate = float(np.round(np.mean([r.win_rate for r in reports]), 4))
  agg.profit_factor = float(np.round(np.mean([r.profit_factor for r in reports]), 4))
  agg.trade_count = int(sum(r.trade_count for r in reports))
  agg.rebalance_count = int(sum(getattr(r, "rebalance_count", r.trade_count) for r in reports))

  if len(agg.ror) > 0:
    eq = 1.0 + agg.ror
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    agg.max_drawdown = float(np.round(np.min(dd), 4))
    ret_daily = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.zeros(0)
    agg.calmar = (
      float(np.round((np.mean(ret_daily) * 252) / abs(agg.max_drawdown), 4))
      if agg.max_drawdown != 0.0 and len(ret_daily) >= 2
      else 0.0
    )
  if base.benchmark_ror is not None and len(base.benchmark_ror) >= min_len:
    agg.benchmark_ror = base.benchmark_ror[:min_len]
  return agg


def _draw_card(
  fig: plt.Figure,
  rect: tuple[float, float, float, float],
  facecolor: str = PALETTE["card"],
  edgecolor: str = PALETTE["border"],
  linewidth: float = 1.0,
  pad: float = 0.0,
  rounding: float = 0.012,
  zorder: int = 1,
) -> FancyBboxPatch:
  """Draw a modern rounded rectangle card in figure coordinates."""
  x, y, w, h = rect
  patch = FancyBboxPatch(
    (x, y),
    w,
    h,
    boxstyle=f"round,pad={pad},rounding_size={rounding}",
    transform=fig.transFigure,
    facecolor=facecolor,
    edgecolor=edgecolor,
    linewidth=linewidth,
    zorder=zorder,
  )
  fig.patches.append(patch)
  return patch


def _val_color(val: float, neutral: str = PALETTE["text_primary"]) -> str:
  if val > 1e-6:
    return PALETTE["up"]
  if val < -1e-6:
    return PALETTE["down"]
  return neutral


def render_strategy_overview(
  reports: list[Report],
  output_path: str,
  strategy: Any = None,
  selector: Any = None,
  account: Any = None,
  dp: Any = None,
  req: Any = None,
  is_portfolio: bool = False,
  title: str | None = None,
  brief: str | None = None,
  universe: str | None = None,
  dpi: int = 200,
) -> str:
  """Render a modern, high-DPI strategy overview PNG chart and save to output_path."""
  if not reports:
    raise ValueError("Cannot render strategy overview: reports list is empty.")

  _configure_fonts()
  report = aggregate_reports(reports)
  meta = extract_strategy_meta(
    report=report,
    strategy=strategy,
    selector=selector,
    account=account,
    req=req,
    is_portfolio=is_portfolio,
    multi_count=len(reports),
    title=title,
    brief=brief,
    universe=universe,
  )
  rebal_events = extract_rebalance_events(report, account=account, dp=dp)
  holding_counts = extract_holding_counts(report, account=account, dp=dp)
  avg_dur, max_dur, min_dur = extract_holding_duration_stats(report, account=account, dp=dp)

  fig = plt.figure(figsize=(6.4, 11.4), dpi=dpi, facecolor=PALETTE["bg"])

  # 1. Header Banner Card
  _draw_card(fig, (0.035, 0.888, 0.930, 0.092), rounding=0.014, zorder=0)
  fig.text(
    0.058,
    0.966,
    meta.title,
    fontsize=16.0,
    fontweight="bold",
    color=PALETTE["text_primary"],
    va="top",
  )

  # Mode & metadata pills
  pills = [
    (meta.mode_label, PALETTE["accent_blue_bg"], PALETTE["strategy"]),
    (f"基准 {meta.benchmark_label}", PALETTE["badge_bg"], PALETTE["text_secondary"]),
    (f"资金 {meta.initial_cash_label}", PALETTE["badge_bg"], PALETTE["text_secondary"]),
  ]
  px = 0.058
  for label, bg_c, fg_c in pills:
    fig.text(
      px,
      0.928,
      f" {label} ",
      fontsize=8.8,
      fontweight="medium",
      color=fg_c,
      va="center",
      bbox=dict(
        boxstyle="round,pad=0.25",
        facecolor=bg_c,
        edgecolor="none",
      ),
    )
    ascii_cnt = sum(1 for ch in label if ord(ch) < 128)
    cjk_cnt = len(label) - ascii_cnt
    px += 0.032 + ascii_cnt * 0.0115 + cjk_cnt * 0.0215

  fig.text(
    0.058,
    0.902,
    f"区间 {meta.date_range}",
    fontsize=8.6,
    color=PALETTE["text_muted"],
    va="center",
  )

  # Right highlight pill in header: Excess Return
  excess_ret = float(report.profit) - float(getattr(report, "benchmark_profit", 0.0))
  ex_color = _val_color(excess_ret)
  ex_bg = PALETTE["up_bg"] if excess_ret >= 0 else PALETTE["down_bg"]
  fig.text(
    0.942,
    0.956,
    f"超额 {excess_ret:+.2%}",
    fontsize=11.5,
    fontweight="bold",
    color=ex_color,
    ha="right",
    va="center",
    bbox=dict(boxstyle="round,pad=0.38", facecolor=ex_bg, edgecolor="none"),
  )
  fig.text(
    0.942,
    0.922,
    f"策略 {report.profit:+.2%} vs 基准 {report.benchmark_profit:+.2%}",
    fontsize=8.2,
    color=PALETTE["text_muted"],
    ha="right",
    va="center",
  )

  # 2. KPI Cards Grid (3 rows x 3 columns)
  n_bars = len(report.ror)
  cagr = (1.0 + float(report.profit)) ** (252.0 / max(1, n_bars)) - 1.0 if n_bars > 0 and (1.0 + float(report.profit)) > 0 else float(report.profit)
  rebal_cnt = getattr(report, "rebalance_count", 0) or len(rebal_events) or report.trade_count

  dur_val = f"平均 {round(avg_dur)} 天" if max_dur > 0 else "无持仓"
  dur_sub = f"最长 {max_dur} / 最短 {min_dur} 天" if max_dur > 0 else "-"

  kpis = [
    ("累计收益", f"{report.profit:+.2%}", f"基准 {report.benchmark_profit:+.2%}", _val_color(report.profit)),
    ("年化收益", f"{cagr:+.2%}", f"Alpha {report.alpha:+.2%}", _val_color(cagr)),
    ("最大回撤", f"{report.max_drawdown:.2%}", f"Calmar {report.calmar:.2f}", PALETTE["down"]),
    ("夏普比率", f"{report.sharpe:.2f}", f"Sortino {report.sortino:.2f}", PALETTE["text_primary"]),
    ("胜率 / 盈亏比", f"{report.win_rate:.1%} / {report.profit_factor:.2f}", f"均盈 {report.avg_win:+.1%} / 亏 {report.avg_loss:+.1%}", PALETTE["text_primary"]),
    ("风险敞口", f"Beta {report.beta:.2f}", f"信息比率 {report.ir:.2f}", PALETTE["text_primary"]),
    ("选股范围", meta.universe_headline, meta.universe_detail, PALETTE["strategy"]),
    ("交易统计", f"调仓 {rebal_cnt} 次 · {report.trade_count} 笔", f"费率 {report.fee_rate:.2%} (按千1计)", PALETTE["text_primary"]),
    ("持仓统计", dur_val, dur_sub, PALETTE["text_primary"]),
  ]

  gap_x = 0.018
  gap_y = 0.011
  card_w = (0.930 - 2 * gap_x) / 3.0
  card_h = 0.087
  for i, (k_title, k_val, k_sub, k_col) in enumerate(kpis):
    col_i = i % 3
    row_i = i // 3
    cx = 0.035 + col_i * (card_w + gap_x)
    cy = 0.788 - row_i * (card_h + gap_y)
    _draw_card(fig, (cx, cy, card_w, card_h), rounding=0.012, zorder=0)
    fig.text(cx + 0.016, cy + card_h - 0.011, k_title, fontsize=9.8, fontweight="medium", color=PALETTE["text_secondary"], va="top")
    if i == 6 and len(k_val) > 8:
      val_fs = 10.5
    elif i == 7 and len(k_val) > 9:
      val_fs = 11.2
    else:
      val_fs = 14.5
    fig.text(cx + 0.016, cy + card_h * 0.50, k_val, fontsize=val_fs, fontweight="bold", color=k_col, va="center")
    sub_fs = 7.8 if i == 7 else 8.3
    fig.text(cx + 0.016, cy + 0.011, k_sub, fontsize=sub_fs, color=PALETTE["text_muted"], va="bottom")

  # 3. Unified 3-Panel Chart Card (Equity + Drawdown + Holding Count)
  _draw_card(fig, (0.035, 0.036, 0.930, 0.542), rounding=0.014, zorder=0)

  ax_eq = fig.add_axes([0.135, 0.358, 0.795, 0.190], facecolor=PALETTE["card"], zorder=3)
  ax_dd = fig.add_axes([0.135, 0.222, 0.795, 0.108], sharex=ax_eq, facecolor=PALETTE["card"], zorder=3)
  ax_pos = fig.add_axes([0.135, 0.080, 0.795, 0.112], sharex=ax_eq, facecolor=PALETTE["card"], zorder=3)

  dates, _ = _parse_dates(report.date) if n_bars > 0 else ([], [])
  strat_pct = np.asarray(report.ror, dtype=np.float64) * 100.0
  bench_arr = report.benchmark_ror if report.benchmark_ror is not None and len(report.benchmark_ror) == n_bars else np.zeros(n_bars, dtype=np.float64)
  bench_pct = np.asarray(bench_arr, dtype=np.float64) * 100.0

  if n_bars > 0:
    ax_eq.axhline(0.0, color=PALETTE["border"], linewidth=1.0, linestyle="--", zorder=1)
    ax_eq.plot(dates, bench_pct, color=PALETTE["benchmark"], linewidth=1.6, linestyle="--", label=f"基准 ({meta.benchmark_label})", zorder=2)
    ax_eq.plot(dates, strat_pct, color=PALETTE["strategy"], linewidth=2.3, label="策略累计收益", zorder=4)
    ax_eq.fill_between(dates, 0.0, strat_pct, color=PALETTE["strategy_fill"], alpha=0.35, zorder=2)

    if rebal_events:
      ev_dates = [dates[ev.bar_idx] for ev in rebal_events]
      ev_vals = [strat_pct[ev.bar_idx] for ev in rebal_events]
      ax_eq.scatter(ev_dates, ev_vals, s=24, color=PALETTE["rebal"], edgecolor="#FFFFFF", linewidth=1.0, zorder=5, label="调仓节点")

    # End-of-line annotations
    ax_eq.annotate(
      f" {strat_pct[-1]:+.1f}%",
      xy=(dates[-1], strat_pct[-1]),
      fontsize=8.8,
      fontweight="bold",
      color=PALETTE["strategy"],
      va="center",
    )
    ax_eq.annotate(
      f" {bench_pct[-1]:+.1f}%",
      xy=(dates[-1], bench_pct[-1]),
      fontsize=8.2,
      color=PALETTE["text_muted"],
      va="center",
    )

    # Underwater Drawdown
    equity = 1.0 + np.asarray(report.ror, dtype=np.float64)
    peaks = np.maximum.accumulate(equity)
    dd_pct = ((equity - peaks) / np.where(peaks > 0, peaks, 1.0)) * 100.0
    ax_dd.plot(dates, dd_pct, color=PALETTE["drawdown"], linewidth=1.3, zorder=3)
    ax_dd.fill_between(dates, dd_pct, 0.0, color=PALETTE["drawdown_fill"], alpha=0.42, zorder=2)
    min_dd_idx = int(np.argmin(dd_pct))
    ax_dd.scatter([dates[min_dd_idx]], [dd_pct[min_dd_idx]], s=24, color=PALETTE["up"], zorder=4)
    dd_on_right = min_dd_idx > int(0.72 * n_bars)
    ax_dd.annotate(
      f"最大回撤 {dd_pct[min_dd_idx]:.2f}%",
      xy=(dates[min_dd_idx], dd_pct[min_dd_idx]),
      xytext=(-8 if dd_on_right else 8, -2),
      textcoords="offset points",
      fontsize=8.2,
      fontweight="medium",
      color=PALETTE["up"],
      ha="right" if dd_on_right else "left",
      va="center",
    )

    # Holding Count Step Curve
    ax_pos.step(dates, holding_counts, where="post", color="#7C3AED", linewidth=1.7, zorder=3)
    ax_pos.fill_between(dates, holding_counts, 0, step="post", color="#EDE9FE", alpha=0.70, zorder=2)
    max_pos = int(np.max(holding_counts)) if len(holding_counts) > 0 else 0
    ax_pos.set_ylim(0, max(1, int(max_pos * 1.25) + 1))

  # Styling axes cleanly
  for ax in (ax_eq, ax_dd, ax_pos):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["border"])
    ax.spines["bottom"].set_color(PALETTE["border"])
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.9, zorder=0)
    ax.tick_params(colors=PALETTE["text_muted"], labelsize=9.0)

  ax_eq.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
  ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
  ax_pos.yaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=4))

  ax_eq.set_title("累计收益曲线与调仓节点", loc="left", fontsize=11.5, fontweight="bold", color=PALETTE["text_primary"], pad=10)
  ax_eq.legend(loc="upper left", frameon=True, facecolor="#FFFFFF", edgecolor=PALETTE["border"], fontsize=9.0, ncol=3)
  plt.setp(ax_eq.get_xticklabels(), visible=False)
  plt.setp(ax_dd.get_xticklabels(), visible=False)

  def _fmt_xaxis_date(x, _pos=None):
    dt = mdates.num2date(x)
    if dt.month == 1 and dt.day == 1:
      return f"{dt.year}"
    return f"{dt.month:02d}-{dt.day:02d}"

  ax_dd.set_ylabel("回撤", fontsize=9.0, color=PALETTE["text_secondary"])
  ax_pos.set_ylabel("持仓数", fontsize=9.0, color=PALETTE["text_secondary"])
  ax_pos.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=6))
  ax_pos.xaxis.set_major_formatter(mticker.FuncFormatter(_fmt_xaxis_date))

  fig.text(
    0.965,
    0.012,
    f"本策略由 {meta.author} 基于 MSD 构建",
    fontsize=9.2,
    fontweight="bold",
    color=PALETTE["strategy"],
    ha="right",
    va="bottom",
  )

  fig.savefig(output_path, dpi=dpi, facecolor=PALETTE["bg"])
  plt.close(fig)
  return output_path
