# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""High-DPI factor evaluation tear-sheet PNG renderer (Chinese UI)."""

from __future__ import annotations

import datetime
import logging
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from mbt.factor.report import FactorReport, MultiFactorReport
from mbt.run.overview import PALETTE, _configure_fonts

logger = logging.getLogger("factor")

_ENTRY_LABELS = {
  "next_open": "次日开盘",
  "open_close": "次开至收盘",
  "close": "当日收盘",
}

_WINSORIZE_LABELS = {
  "none": "无",
  "mad": "MAD",
  "sigma": "3σ",
  "percentile": "分位截尾",
}

_STANDARDIZE_LABELS = {
  "none": "无",
  "zscore": "Z-Score",
  "rank": "截面排序",
}

_NEUTRALIZE_LABELS = {
  "industry": "行业",
  "mktcap": "市值",
}

_CAUSALITY_LABELS = {
  "PASS": "通过",
  "FAIL": "失败",
  "SKIPPED": "已跳过",
}


def _parse_dates(dates: np.ndarray) -> list[datetime.datetime]:
  out: list[datetime.datetime] = []
  for i, d in enumerate(dates):
    s = str(d)[:10]
    try:
      out.append(datetime.datetime.fromisoformat(s))
    except ValueError:
      out.append(datetime.datetime(2020, 1, 1) + datetime.timedelta(days=i))
  return out


def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
  n = len(arr)
  out = np.full(n, np.nan, dtype=np.float64)
  w = max(1, min(window, n))
  for i in range(w - 1, n):
    seg = arr[i - w + 1 : i + 1]
    fin = seg[np.isfinite(seg)]
    if len(fin) > 0:
      out[i] = float(np.mean(fin))
  return out


def _style_axis(ax: Any, title: str) -> None:
  ax.set_facecolor(PALETTE["card"])
  ax.set_title(
    title,
    loc="left",
    fontsize=11,
    fontweight="bold",
    color=PALETTE["text_primary"],
    pad=8,
  )
  ax.grid(True, color=PALETTE["grid"], linewidth=0.8, linestyle="-")
  ax.set_axisbelow(True)
  for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)
  for spine in ("left", "bottom"):
    ax.spines[spine].set_color(PALETTE["border"])
  ax.tick_params(colors=PALETTE["text_muted"], labelsize=8.5)


def render_factor_overview(
  report: FactorReport | MultiFactorReport,
  output_path: str,
  *,
  dpi: int = 160,
) -> str:
  """Render a high-DPI 4-panel factor evaluation tear sheet PNG with Chinese labels.

  Args:
    report: Single :class:`FactorReport` or :class:`MultiFactorReport` (renders
      the highest-ranked factor by ``|RankICIR|``).
    output_path: Destination PNG file path.
    dpi: Output image resolution (default 160).

  Returns:
    ``output_path`` after saving.
  """
  _configure_fonts()

  is_multi = isinstance(report, MultiFactorReport)
  if is_multi:
    ranked = report.ranked_reports()
    rep = ranked[0]
    subtitle_extra = f"（多因子榜第 1 / 共 {len(ranked)} 个）"
  else:
    rep = report
    subtitle_extra = ""

  p = rep.primary_period
  ic_sum = rep.primary_ic
  qa = rep.primary_quantile
  dt_list = _parse_dates(rep.dates)
  x_dates = np.asarray(mdates.date2num(dt_list), dtype=np.float64)

  fig = plt.figure(figsize=(15.5, 9.8), dpi=dpi, facecolor=PALETTE["bg"])
  gs = fig.add_gridspec(
    3,
    2,
    height_ratios=[0.96, 2.2, 2.2],
    hspace=0.36,
    wspace=0.19,
    left=0.055,
    right=0.97,
    top=0.958,
    bottom=0.06,
  )

  # ── Header KPI Banner ──────────────────────────────────────────────
  ax_hdr = fig.add_subplot(gs[0, :])
  ax_hdr.set_facecolor(PALETTE["card"])
  ax_hdr.set_xticks([])
  ax_hdr.set_yticks([])
  for spine in ax_hdr.spines.values():
    spine.set_color(PALETTE["border"])
    spine.set_linewidth(1.0)

  neut_str = "+".join(_NEUTRALIZE_LABELS.get(m, m) for m in rep.neutralize) if rep.neutralize else "无"
  win_str = _WINSORIZE_LABELS.get(rep.winsorize, rep.winsorize)
  std_str = _STANDARDIZE_LABELS.get(rep.standardize, rep.standardize)
  entry_str = _ENTRY_LABELS.get(rep.entry, rep.entry)
  author_str = f" ｜ 作者：{rep.author}" if rep.author else ""

  ax_hdr.text(
    0.015,
    0.77,
    f"因子评估全景报告：{rep.name}{subtitle_extra}",
    fontsize=14.0,
    fontweight="bold",
    color=PALETTE["text_primary"],
    transform=ax_hdr.transAxes,
    va="center",
  )
  meta_line1 = (
    f"区间：{rep.start_date} ~ {rep.end_date}（{rep.bars}日 · {rep.n_objs}股）"
    f" ｜ 主周期：{p}日 ｜ 入场：{entry_str} ｜ 分组：{rep.quantiles}组"
  )
  meta_line2 = (
    f"预处理：去极值={win_str} · 中性化={neut_str} · 标准化={std_str}{author_str}"
  )
  ax_hdr.text(
    0.015,
    0.44,
    meta_line1,
    fontsize=8.4,
    color=PALETTE["text_secondary"],
    transform=ax_hdr.transAxes,
    va="center",
  )
  ax_hdr.text(
    0.015,
    0.18,
    meta_line2,
    fontsize=8.2,
    color=PALETTE["text_muted"],
    transform=ax_hdr.transAxes,
    va="center",
  )
  ax_hdr.plot([0.545, 0.545], [0.14, 0.86], color=PALETTE["border"], linewidth=1.0, transform=ax_hdr.transAxes)

  causality_raw = str(rep.diagnostics.get("causality_check", {}).get("status", "N/A"))
  causality_cn = _CAUSALITY_LABELS.get(causality_raw, causality_raw)
  causality_color = (
    PALETTE["down"]
    if causality_raw == "PASS"
    else (PALETTE["up"] if causality_raw == "FAIL" else PALETTE["rebal"])
  )

  kpis = [
    ("RankIC 均值", f"{ic_sum.rank_ic_mean:+.4f}", PALETTE["up"] if ic_sum.rank_ic_mean >= 0 else PALETTE["down"]),
    ("RankICIR", f"{ic_sum.rank_icir:+.3f}", PALETTE["strategy"]),
    ("多头年化超额", f"{qa.long_excess_market.ann_return * 100:+.1f}%", PALETTE["up"] if qa.long_excess_market.ann_return >= 0 else PALETTE["down"]),
    ("多空夏普", f"{qa.long_short.sharpe:+.2f}", PALETTE["text_primary"]),
    ("分层单调性", f"{qa.monotonicity:+.2f}", PALETTE["strategy"]),
    ("因果检验", causality_cn, causality_color),
  ]
  x0 = 0.56
  step_x = (0.985 - x0) / len(kpis)
  for idx, (k_label, k_val, k_color) in enumerate(kpis):
    xc = x0 + idx * step_x + step_x * 0.5
    ax_hdr.text(
      xc,
      0.70,
      k_val,
      fontsize=11.5,
      fontweight="bold",
      color=k_color,
      transform=ax_hdr.transAxes,
      ha="center",
      va="center",
    )
    ax_hdr.text(
      xc,
      0.27,
      k_label,
      fontsize=8.0,
      color=PALETTE["text_muted"],
      transform=ax_hdr.transAxes,
      ha="center",
      va="center",
    )

  # ── Panel 1 (Top-Left): Quantile Cumulative NAV Curves ─────────────
  ax1 = fig.add_subplot(gs[1, 0])
  _style_axis(ax1, f"分层累计净值曲线（{p}日重叠等权组合）")
  cmap = plt.get_cmap("coolwarm")
  n_q = qa.n_quantiles
  for i, qm in enumerate(qa.quantiles):
    nav = qa.quantile_navs[qm.label]
    c = cmap(i / max(1, n_q - 1))
    lw = 2.0 if i in (0, n_q - 1) else 1.2
    ax1.plot(x_dates, nav, label=f"{qm.label} ({qm.ann_return * 100:+.1f}%)", color=c, linewidth=lw)
  ax1.plot(x_dates, qa.market_nav, label="市场等权", color=PALETTE["benchmark"], linewidth=1.3, linestyle="--")
  ax1.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor=PALETTE["card"], edgecolor=PALETTE["border"], ncol=2)
  ax1.set_ylabel("累计净值", fontsize=9, color=PALETTE["text_secondary"])
  ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

  # ── Panel 2 (Top-Right): Long-Excess & Long-Short NAV + Drawdown ───
  ax2 = fig.add_subplot(gs[1, 1])
  _style_axis(ax2, f"多头超额 (Q{n_q}-市场) 与多空组合 (Q{n_q}-Q1) 净值")
  ax2.plot(
    x_dates,
    qa.long_excess_nav,
    label=f"多头超额 Q{n_q}-市场 ({qa.long_excess_market.ann_return * 100:+.1f}%, 夏普 {qa.long_excess_market.sharpe:.2f})",
    color=PALETTE["up"],
    linewidth=1.9,
  )
  ax2.plot(
    x_dates,
    qa.long_short_nav,
    label=f"多空组合 Q{n_q}-Q1 ({qa.long_short.ann_return * 100:+.1f}%, 夏普 {qa.long_short.sharpe:.2f})",
    color=PALETTE["strategy"],
    linewidth=1.7,
  )
  ax2.axhline(1.0, color=PALETTE["border"], linewidth=1.0, linestyle="--")
  peak = np.maximum.accumulate(np.concatenate(([1.0], qa.long_excess_nav)))[1:]
  ax2.fill_between(
    x_dates,
    qa.long_excess_nav,
    peak,
    color=PALETTE["drawdown_fill"],
    alpha=0.22,
    label=f"超额回撤（最大 {qa.long_excess_market.max_drawdown * 100:.1f}%）",
  )
  ax2.legend(loc="upper left", fontsize=8.0, frameon=True, facecolor=PALETTE["card"], edgecolor=PALETTE["border"])
  ax2.set_ylabel("超额 / 多空净值", fontsize=9, color=PALETTE["text_secondary"])
  ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

  # ── Panel 3 (Bottom-Left): Cumulative & Rolling RankIC ─────────────
  ax3 = fig.add_subplot(gs[2, 0])
  _style_axis(ax3, f"RankIC 累计与滚动趋势（{p}日）")
  ric = rep.rank_ic_series[p]
  ric_clean = np.where(np.isfinite(ric), ric, 0.0)
  cum_ric = np.cumsum(ric_clean)
  roll_win = min(60, max(5, len(ric) // 5))
  roll_ric = _rolling_mean(ric, roll_win)

  ax3.plot(x_dates, cum_ric, color=PALETTE["strategy"], linewidth=1.9, label="累计 RankIC")
  ax3.set_ylabel("累计 RankIC", fontsize=9, color=PALETTE["strategy"])
  ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

  ax3_r = ax3.twinx()
  ax3_r.plot(x_dates, roll_ric, color=PALETTE["rebal"], linewidth=1.3, alpha=0.9, label=f"{roll_win}日滚动 RankIC")
  ax3_r.axhline(0.0, color=PALETTE["border"], linewidth=0.9, linestyle="--")
  ax3_r.set_ylabel("滚动 RankIC", fontsize=9, color=PALETTE["rebal"])
  ax3_r.spines["top"].set_visible(False)
  ax3_r.spines["right"].set_color(PALETTE["border"])
  ax3_r.tick_params(colors=PALETTE["text_muted"], labelsize=8.5)

  h1, l1 = ax3.get_legend_handles_labels()
  h2, l2 = ax3_r.get_legend_handles_labels()
  ax3.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=8.0, frameon=True, facecolor=PALETTE["card"], edgecolor=PALETTE["border"])

  # ── Panel 4 (Bottom-Right): Quantile Ann Return & Horizon IC Decay ─
  ax4 = fig.add_subplot(gs[2, 1])
  _style_axis(ax4, f"各分位组年化收益（{p}日）与多周期 IC 衰减")
  q_labels = [qm.label for qm in qa.quantiles]
  q_ann = [qm.ann_return * 100.0 for qm in qa.quantiles]
  bar_colors = [PALETTE["up"] if v >= 0 else PALETTE["down"] for v in q_ann]
  x_pos = np.arange(len(q_labels))
  bars_plot = ax4.bar(x_pos, q_ann, color=bar_colors, alpha=0.82, width=0.58, edgecolor=PALETTE["border"])
  ax4.axhline(0.0, color=PALETTE["text_muted"], linewidth=0.8)
  ax4.set_xticks(x_pos)
  ax4.set_xticklabels(q_labels)
  ax4.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
  ax4.set_ylabel("年化收益率", fontsize=9, color=PALETTE["text_secondary"])

  y_max = max(0.0, max(q_ann)) if q_ann else 1.0
  y_min = min(0.0, min(q_ann)) if q_ann else 0.0
  y_span = max(2.0, y_max - y_min)
  ax4.set_ylim(y_min - 0.12 * y_span if y_min < 0 else 0.0, y_max + 0.34 * y_span)

  for rect, val in zip(bars_plot, q_ann):
    va = "bottom" if val >= 0 else "top"
    offset = 0.025 * y_span if val >= 0 else -0.025 * y_span
    ax4.text(
      rect.get_x() + rect.get_width() * 0.5,
      val + offset,
      f"{val:+.1f}%",
      ha="center",
      va=va,
      fontsize=8.3,
      color=PALETTE["text_primary"],
    )

  # Compact horizontal multi-horizon RankIC strip at top headroom (never overlaps bars)
  horizon_items = [
    f"{hp}日: {rep.ic_summaries[hp].rank_ic_mean:+.4f} (IR {rep.ic_summaries[hp].rank_icir:+.2f})"
    for hp in rep.periods
  ]
  ax4.text(
    0.5,
    0.955,
    "多周期 RankIC： " + " ｜ ".join(horizon_items),
    transform=ax4.transAxes,
    fontsize=7.6,
    va="top",
    ha="center",
    color=PALETTE["text_secondary"],
    bbox=dict(boxstyle="round,pad=0.32", facecolor=PALETTE["accent_blue_bg"], edgecolor=PALETTE["border"], alpha=0.95),
  )

  fig.savefig(output_path, dpi=dpi, facecolor=PALETTE["bg"])
  plt.close(fig)
  return output_path
