# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""High-performance Matplotlib equity curve widget for Tkinter."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Sequence

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.ticker as mticker
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np

from mbt.gui.downsample import minmax_downsample
from mbt.gui.dpi import get_ui_scale, scaled_font

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

_FONT_CONFIGURED = False


def _ensure_cjk_font() -> None:
  global _FONT_CONFIGURED
  if _FONT_CONFIGURED:
    return
  try:
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for cand in _CJK_FONT_CANDIDATES:
      if cand in available:
        chosen = cand
        break
    matplotlib.rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    matplotlib.rcParams["axes.unicode_minus"] = False
  except Exception:
    pass
  _FONT_CONFIGURED = True


def _compute_drawdown(ror: np.ndarray) -> np.ndarray:
  if len(ror) == 0:
    return np.empty(0, dtype=np.float64)
  equity = 1.0 + np.asarray(ror, dtype=np.float64)
  peaks = np.maximum.accumulate(equity)
  with np.errstate(divide="ignore", invalid="ignore"):
    dd = np.where(peaks > 0, (equity - peaks) / peaks, 0.0)
  return dd


class EquityCurvePanel(ttk.Frame):
  """Interactive equity curve + drawdown chart with dynamic viewport downsampling."""

  def __init__(self, parent: tk.Widget, max_points: int = 3000):
    super().__init__(parent)
    _ensure_cjk_font()
    self._max_points = max_points
    self._dates: list[str] = []
    self._ror = np.empty(0, dtype=np.float64)
    self._orig = np.empty(0, dtype=np.float64)
    self._bench = np.empty(0, dtype=np.float64)
    self._dd = np.empty(0, dtype=np.float64)
    self._title = "收益曲线"
    self._bench_label = "基准"
    self._orig_label = "买入持有"

    self._show_orig = tk.BooleanVar(value=True)
    self._show_bench = tk.BooleanVar(value=True)
    self._show_dd = tk.BooleanVar(value=True)

    self._resampling = False
    self._last_range: tuple[int, int] | None = None
    self._hover_var = tk.StringVar(value="悬停图表查看每日数值，使用工具栏可缩放与平移")

    self._build_ui()

  def _build_ui(self) -> None:
    top_bar = ttk.Frame(self)
    top_bar.pack(fill=tk.X, padx=6, pady=(4, 2))

    self._title_lbl = ttk.Label(top_bar, text="收益曲线", font=scaled_font(self, 13, bold=True))
    self._title_lbl.pack(side=tk.LEFT)

    opts_frame = ttk.Frame(top_bar)
    opts_frame.pack(side=tk.RIGHT)
    ttk.Checkbutton(
      opts_frame, text="买入持有", variable=self._show_orig, command=self._rebuild_plot
    ).pack(side=tk.LEFT, padx=4)
    ttk.Checkbutton(
      opts_frame, text="基准", variable=self._show_bench, command=self._rebuild_plot
    ).pack(side=tk.LEFT, padx=4)
    ttk.Checkbutton(
      opts_frame, text="回撤", variable=self._show_dd, command=self._rebuild_plot
    ).pack(side=tk.LEFT, padx=4)

    hover_lbl = ttk.Label(self, textvariable=self._hover_var, foreground="#475569")
    hover_lbl.pack(fill=tk.X, padx=8, pady=(0, 2))

    fig_dpi = max(80, int(round(100 * get_ui_scale(self))))
    self._fig = Figure(figsize=(8, 4.2), dpi=fig_dpi, facecolor="#FFFFFF")
    self._canvas = FigureCanvasTkAgg(self._fig, master=self)
    self._canvas_widget = self._canvas.get_tk_widget()
    self._canvas_widget.pack(fill=tk.BOTH, expand=True)

    self._toolbar = NavigationToolbar2Tk(self._canvas, self, pack_toolbar=False)
    self._toolbar.update()
    self._toolbar.pack(fill=tk.X, padx=4, pady=(0, 2))

    self._ax_main = None
    self._ax_dd = None
    self._line_strat = None
    self._line_orig = None
    self._line_bench = None
    self._line_dd = None
    self._dd_poly = None
    self._vline_main = None
    self._vline_dd = None

    self._canvas.mpl_connect("motion_notify_event", self._on_mouse_move)
    self._canvas.mpl_connect("axes_leave_event", self._on_mouse_leave)

  def set_series(
    self,
    dates: Sequence[str],
    ror: np.ndarray,
    ror_original: np.ndarray | None = None,
    benchmark_ror: np.ndarray | None = None,
    title: str = "收益曲线",
    orig_label: str = "买入持有",
    bench_label: str = "基准",
  ) -> None:
    """Load full curve arrays and render with viewport Min-Max downsampling."""
    self._dates = list(dates)
    n = len(self._dates)
    self._ror = np.asarray(ror, dtype=np.float64)[:n]
    self._orig = (
      np.asarray(ror_original, dtype=np.float64)[:n]
      if ror_original is not None and len(ror_original) > 0
      else np.empty(0, dtype=np.float64)
    )
    self._bench = (
      np.asarray(benchmark_ror, dtype=np.float64)[:n]
      if benchmark_ror is not None and len(benchmark_ror) > 0
      else np.empty(0, dtype=np.float64)
    )
    self._dd = _compute_drawdown(self._ror)
    self._title = title
    self._orig_label = orig_label
    self._bench_label = bench_label
    self._last_range = None
    self._title_lbl.configure(text=f"{title}  (共 {n} 点)")
    self._rebuild_plot()

  def clear(self, message: str = "暂无曲线数据") -> None:
    """Clear chart and show placeholder text."""
    self._dates = []
    self._ror = np.empty(0, dtype=np.float64)
    self._orig = np.empty(0, dtype=np.float64)
    self._bench = np.empty(0, dtype=np.float64)
    self._dd = np.empty(0, dtype=np.float64)
    self._fig.clear()
    ax = self._fig.add_subplot(111)
    ax.text(0.5, 0.5, message, ha="center", va="center", color="#64748B", fontsize=11)
    ax.set_axis_off()
    self._ax_main = None
    self._ax_dd = None
    self._canvas.draw_idle()

  def _format_x_tick(self, val: float, _pos: int) -> str:
    if not self._dates:
      return ""
    idx = int(round(val))
    if 0 <= idx < len(self._dates):
      return self._dates[idx]
    return ""

  def _rebuild_plot(self) -> None:
    self._fig.clear()
    self._ax_main = None
    self._ax_dd = None
    self._line_strat = None
    self._line_orig = None
    self._line_bench = None
    self._line_dd = None
    self._dd_poly = None
    self._vline_main = None
    self._vline_dd = None
    self._last_range = None

    n = len(self._dates)
    if n == 0 or len(self._ror) == 0:
      self.clear()
      return

    show_dd = bool(self._show_dd.get())
    if show_dd:
      gs = self._fig.add_gridspec(2, 1, height_ratios=[3.2, 1.15], hspace=0.14)
      self._ax_main = self._fig.add_subplot(gs[0])
      self._ax_dd = self._fig.add_subplot(gs[1], sharex=self._ax_main)
    else:
      self._ax_main = self._fig.add_subplot(111)

    ax = self._ax_main
    ax.set_facecolor("#FFFFFF")
    ax.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
    ax.axhline(0.0, color="#94A3B8", linewidth=0.9, linestyle=":")

    xs_s, ys_s = minmax_downsample(self._ror, 0, n, self._max_points)
    (self._line_strat,) = ax.plot(
      xs_s, ys_s * 100.0, color="#2563EB", linewidth=1.7, label="策略收益"
    )

    if self._show_orig.get() and len(self._orig) == n:
      xs_o, ys_o = minmax_downsample(self._orig, 0, n, self._max_points)
      (self._line_orig,) = ax.plot(
        xs_o, ys_o * 100.0, color="#D97706", linewidth=1.2, linestyle="--", alpha=0.88, label=self._orig_label
      )

    if self._show_bench.get() and len(self._bench) == n:
      xs_b, ys_b = minmax_downsample(self._bench, 0, n, self._max_points)
      (self._line_bench,) = ax.plot(
        xs_b, ys_b * 100.0, color="#64748B", linewidth=1.2, alpha=0.85, label=self._bench_label
      )

    ax.set_ylabel("收益率 (%)")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
    ax.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=8.5)
    self._vline_main = ax.axvline(0, color="#475569", linewidth=0.8, linestyle="--", visible=False)

    if show_dd and self._ax_dd is not None:
      ax_dd = self._ax_dd
      ax_dd.set_facecolor("#FFFFFF")
      ax_dd.grid(True, linestyle="--", alpha=0.45, color="#CBD5E1")
      xs_d, ys_d = minmax_downsample(self._dd, 0, n, self._max_points)
      ys_d_pct = ys_d * 100.0
      (self._line_dd,) = ax_dd.plot(xs_d, ys_d_pct, color="#DC2626", linewidth=1.1)
      self._dd_poly = ax_dd.fill_between(xs_d, ys_d_pct, 0.0, color="#FCA5A5", alpha=0.45)
      ax_dd.set_ylabel("回撤 (%)")
      ax_dd.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f%%"))
      ax_dd.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_x_tick))
      ax_dd.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))
      self._vline_dd = ax_dd.axvline(0, color="#475569", linewidth=0.8, linestyle="--", visible=False)
      ax.tick_params(labelbottom=False)
    else:
      ax.xaxis.set_major_formatter(mticker.FuncFormatter(self._format_x_tick))
      ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=8, integer=True))

    ax.set_xlim(0, max(1, n - 1))
    self._last_range = (0, n)
    self._fig.subplots_adjust(left=0.075, right=0.975, top=0.95, bottom=0.11)
    ax.callbacks.connect("xlim_changed", self._on_xlim_changed)

    self._toolbar.update()
    self._canvas.draw_idle()
    self._update_readout(n - 1)

  def _on_xlim_changed(self, ax) -> None:
    if self._resampling or self._ax_main is None or len(self._ror) == 0:
      return
    n = len(self._ror)
    x0, x1 = ax.get_xlim()
    s = max(0, int(np.floor(min(x0, x1))))
    e = min(n, int(np.ceil(max(x0, x1))) + 1)
    if e <= s:
      return
    if self._last_range == (s, e):
      return
    self._last_range = (s, e)
    self._resampling = True
    try:
      if self._line_strat is not None:
        xs_s, ys_s = minmax_downsample(self._ror, s, e, self._max_points)
        self._line_strat.set_data(xs_s, ys_s * 100.0)
      if self._line_orig is not None and len(self._orig) == n:
        xs_o, ys_o = minmax_downsample(self._orig, s, e, self._max_points)
        self._line_orig.set_data(xs_o, ys_o * 100.0)
      if self._line_bench is not None and len(self._bench) == n:
        xs_b, ys_b = minmax_downsample(self._bench, s, e, self._max_points)
        self._line_bench.set_data(xs_b, ys_b * 100.0)
      if self._ax_dd is not None and self._line_dd is not None and len(self._dd) == n:
        xs_d, ys_d = minmax_downsample(self._dd, s, e, self._max_points)
        ys_d_pct = ys_d * 100.0
        self._line_dd.set_data(xs_d, ys_d_pct)
        if self._dd_poly is not None:
          try:
            self._dd_poly.remove()
          except Exception:
            pass
        self._dd_poly = self._ax_dd.fill_between(
          xs_d, ys_d_pct, 0.0, color="#FCA5A5", alpha=0.45
        )
    finally:
      self._resampling = False
    self._canvas.draw_idle()

  def _update_readout(self, idx: int) -> None:
    n = len(self._dates)
    if n == 0 or idx < 0 or idx >= n:
      return
    parts = [f"日期: {self._dates[idx]}", f"策略: {self._ror[idx]:+.2%}"]
    if len(self._orig) == n:
      parts.append(f"{self._orig_label}: {self._orig[idx]:+.2%}")
    if len(self._bench) == n:
      parts.append(f"{self._bench_label}: {self._bench[idx]:+.2%}")
    if len(self._dd) == n:
      parts.append(f"回撤: {self._dd[idx]:.2%}")
    self._hover_var.set("  |  ".join(parts))

  def _on_mouse_move(self, event) -> None:
    if self._ax_main is None or len(self._dates) == 0:
      return
    if event.inaxes not in (self._ax_main, self._ax_dd) or event.xdata is None:
      return
    idx = max(0, min(len(self._dates) - 1, int(round(event.xdata))))
    if self._vline_main is not None:
      self._vline_main.set_xdata([idx, idx])
      self._vline_main.set_visible(True)
    if self._vline_dd is not None:
      self._vline_dd.set_xdata([idx, idx])
      self._vline_dd.set_visible(True)
    self._update_readout(idx)
    self._canvas.draw_idle()

  def _on_mouse_leave(self, _event) -> None:
    changed = False
    if self._vline_main is not None and self._vline_main.get_visible():
      self._vline_main.set_visible(False)
      changed = True
    if self._vline_dd is not None and self._vline_dd.get_visible():
      self._vline_dd.set_visible(False)
      changed = True
    if changed:
      self._canvas.draw_idle()
