# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Main Tkinter GUI application for mbt."""

from __future__ import annotations

import dataclasses
import json
import os
import pickle
import queue
import subprocess
import sys
import tempfile
import threading
import time
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from mbt.gui.config import (
  GuiConfig,
  RUN_MODE_LABELS,
  RUN_MODE_PORTFOLIO,
  RUN_MODE_SELECT,
  RUN_MODE_TIMING,
  ShortcutConfig,
  StrategyConfig,
  get_config_file,
  load_config,
  save_config,
)
from mbt.gui.dialogs import RenameDialog, ShortcutDialog
from mbt.gui.dpi import apply_ui_scale, configure_scaled_ttk_style, scaled, scaled_font
from mbt.gui.inspect_strategy import StrategyMetadata, inspect_strategy_file
from mbt.gui.views import PortfolioResultView, SelectResultView, TimingResultView


def _format_shortcut_summary(sc: ShortcutConfig) -> str:
  parts = [f"模式: {RUN_MODE_LABELS.get(sc.mode, sc.mode)}"]
  if sc.mode == RUN_MODE_SELECT:
    parts.append(f"日期: {sc.select_date or '最新'}")
  else:
    parts.append(f"区间: {sc.begin}~{sc.end}")
  if sc.use_internal_selector and sc.mode != RUN_MODE_SELECT:
    pool_desc = f"内置选股器" + (f" ({sc.objs})" if sc.objs else "")
    parts.append(f"股票池: {pool_desc}")
  elif sc.objs:
    parts.append(f"标的: {sc.objs}")
  if sc.strategy_args:
    arg_str = " ".join(f"{k}={v}" for k, v in sc.strategy_args.items())
    parts.append(f"参数: {arg_str}")
  if sc.extra_args:
    parts.append(f"额外: {sc.extra_args}")
  return "  |  ".join(parts)


class MbtGuiApp:
  """Main window controller for the mbt strategy & backtest GUI."""

  def __init__(self, root: tk.Tk, scale: float | None = None):
    self.root = root
    self.root.title("MSD Backtest Studio (mbt)")
    self._scale = apply_ui_scale(self.root, scale)

    sw = max(800, self.root.winfo_screenwidth())
    sh = max(600, self.root.winfo_screenheight())
    win_w = min(scaled(1320, self.root), sw - 80)
    win_h = min(scaled(840, self.root), sh - 80)
    self.root.geometry(f"{win_w}x{win_h}")
    self.root.minsize(min(scaled(980, self.root), win_w), min(scaled(620, self.root), win_h))

    self._setup_style()
    self._cfg: GuiConfig = self._safe_load_config()
    self._meta_cache: dict[str, StrategyMetadata] = {}

    self._selected_strategy_id: str | None = None
    self._selected_shortcut_id: str | None = None

    self._proc: subprocess.Popen[str] | None = None
    self._temp_dir: tempfile.TemporaryDirectory[str] | None = None
    self._msg_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
    self._run_start_ts: float = 0.0
    self._cancelled = False
    self._log_visible = False

    self._build_ui()
    self._populate_tree()
    self._update_selection_state()
    self.root.protocol("WM_DELETE_WINDOW", self._on_close)

  def _setup_style(self) -> None:
    style = ttk.Style(self.root)
    if "clam" in style.theme_names():
      style.theme_use("clam")
    configure_scaled_ttk_style(self.root, style, self._scale)

  def _safe_load_config(self) -> GuiConfig:
    try:
      return load_config()
    except Exception as exc:
      messagebox.showerror(
        "配置读取失败",
        f"读取配置文件 {get_config_file()} 时出错:\n{exc}",
        parent=self.root,
      )
      return GuiConfig()

  def _save_cfg(self) -> None:
    try:
      save_config(self._cfg)
    except Exception as exc:
      messagebox.showerror(
        "配置保存失败",
        f"写入配置文件 {get_config_file()} 时出错:\n{exc}",
        parent=self.root,
      )

  def _build_ui(self) -> None:
    main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
    main_paned.pack(fill=tk.BOTH, expand=True)

    # Left sidebar: strategies & shortcuts
    left = ttk.Frame(main_paned, width=scaled(300, self.root))
    main_paned.add(left, weight=1)
    self._build_sidebar(left)

    # Right workspace: action bar, dynamic views, collapsible logs, status bar
    right = ttk.Frame(main_paned)
    main_paned.add(right, weight=4)
    self._build_workspace(right)

  def _build_sidebar(self, parent: ttk.Frame) -> None:
    top = ttk.Frame(parent, padding=(8, 8, 8, 4))
    top.pack(fill=tk.X)
    ttk.Label(top, text="策略与快捷路径", font=scaled_font(self.root, 13, bold=True)).pack(side=tk.LEFT)
    ttk.Button(top, text="+ 导入策略", command=self._import_strategy).pack(side=tk.RIGHT)

    strat_btns = ttk.Frame(parent, padding=(8, 0, 8, 6))
    strat_btns.pack(fill=tk.X)
    self._btn_rename = ttk.Button(strat_btns, text="改名", width=6, command=self._rename_strategy)
    self._btn_rename.pack(side=tk.LEFT, padx=(0, 4))
    self._btn_remove = ttk.Button(strat_btns, text="移除", width=6, command=self._remove_strategy)
    self._btn_remove.pack(side=tk.LEFT, padx=(0, 4))
    self._btn_add_sc = ttk.Button(strat_btns, text="+ 快捷路径", command=self._add_shortcut)
    self._btn_add_sc.pack(side=tk.RIGHT)

    tree_wrap = ttk.Frame(parent, padding=(8, 0, 4, 8))
    tree_wrap.pack(fill=tk.BOTH, expand=True)
    tree_wrap.rowconfigure(0, weight=1)
    tree_wrap.columnconfigure(0, weight=1)

    self._tree = ttk.Treeview(tree_wrap, show="tree", selectmode="browse")
    ysb = ttk.Scrollbar(tree_wrap, orient=tk.VERTICAL, command=self._tree.yview)
    self._tree.configure(yscrollcommand=ysb.set)
    self._tree.grid(row=0, column=0, sticky=tk.NSEW)
    ysb.grid(row=0, column=1, sticky=tk.NS)

    self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)
    self._tree.bind("<Double-1>", self._on_tree_double_click)
    self._tree.bind("<Button-3>", self._on_tree_right_click)

    self._ctx_menu = tk.Menu(self.root, tearoff=False)

  def _build_workspace(self, parent: ttk.Frame) -> None:
    # Top action & parameter preview bar
    hdr = ttk.Frame(parent, padding=(10, 8, 10, 6))
    hdr.pack(fill=tk.X)

    info_col = ttk.Frame(hdr)
    info_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
    self._title_var = tk.StringVar(value="请选择或导入策略")
    self._detail_var = tk.StringVar(value=f"配置文件: {get_config_file()}")
    ttk.Label(info_col, textvariable=self._title_var, font=scaled_font(self.root, 14, bold=True)).pack(
      anchor=tk.W
    )
    ttk.Label(info_col, textvariable=self._detail_var, foreground="#475569").pack(
      anchor=tk.W, pady=(2, 0)
    )

    btn_col = ttk.Frame(hdr)
    btn_col.pack(side=tk.RIGHT)
    self._btn_run = ttk.Button(btn_col, text="▶ 运行", command=self._run_selected)
    self._btn_run.pack(side=tk.LEFT, padx=(0, 4))
    self._btn_stop = ttk.Button(btn_col, text="⏹ 停止", command=self._stop_run)
    self._btn_stop.pack(side=tk.LEFT, padx=(0, 8))
    self._btn_edit_sc = ttk.Button(btn_col, text="编辑", width=6, command=self._edit_shortcut)
    self._btn_edit_sc.pack(side=tk.LEFT, padx=(0, 4))
    self._btn_dup_sc = ttk.Button(btn_col, text="复制", width=6, command=self._duplicate_shortcut)
    self._btn_dup_sc.pack(side=tk.LEFT, padx=(0, 4))
    self._btn_del_sc = ttk.Button(btn_col, text="删除", width=6, command=self._delete_shortcut)
    self._btn_del_sc.pack(side=tk.LEFT)

    ttk.Separator(parent, orient=tk.HORIZONTAL).pack(fill=tk.X)

    # Dynamic result view container
    self._view_host = ttk.Frame(parent)
    self._view_host.pack(fill=tk.BOTH, expand=True)

    self._welcome_view = ttk.Frame(self._view_host, padding=30)
    ttk.Label(
      self._welcome_view,
      text="点击左侧“+ 导入策略”导入 .py 策略文件，或双击已有快捷路径运行。",
      font=scaled_font(self.root, 14),
      foreground="#64748B",
    ).pack(expand=True)

    self._select_view = SelectResultView(self._view_host)
    self._timing_view = TimingResultView(self._view_host)
    self._portfolio_view = PortfolioResultView(self._view_host)
    self._active_view: tk.Widget | None = None
    self._switch_view(self._welcome_view)

    # Collapsible log drawer
    self._log_container = ttk.Frame(parent)
    log_bar = ttk.Frame(self._log_container, padding=(8, 2))
    log_bar.pack(fill=tk.X)
    ttk.Label(log_bar, text="运行日志", font=scaled_font(self.root, 12, bold=True)).pack(side=tk.LEFT)
    ttk.Button(log_bar, text="清空", width=6, command=self._clear_log).pack(side=tk.RIGHT)
    self._log_text = tk.Text(
      self._log_container, height=8, wrap=tk.WORD, bg="#0F172A", fg="#E2E8F0", state=tk.DISABLED
    )
    log_scroll = ttk.Scrollbar(self._log_container, orient=tk.VERTICAL, command=self._log_text.yview)
    self._log_text.configure(yscrollcommand=log_scroll.set)
    self._log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(8, 0), pady=(0, 4))
    log_scroll.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 8), pady=(0, 4))

    # Bottom status bar
    self._status_sep = ttk.Separator(parent, orient=tk.HORIZONTAL)
    self._status_sep.pack(fill=tk.X)
    status_bar = ttk.Frame(parent, padding=(8, 4))
    status_bar.pack(fill=tk.X)
    self._status_var = tk.StringVar(value="就绪")
    ttk.Label(status_bar, textvariable=self._status_var, foreground="#334155").pack(side=tk.LEFT)
    self._btn_toggle_log = ttk.Button(
      status_bar, text="展开日志 ▲", command=self._toggle_log_drawer
    )
    self._btn_toggle_log.pack(side=tk.RIGHT)

  def _switch_view(self, target: tk.Widget) -> None:
    if self._active_view is target:
      return
    if self._active_view is not None:
      self._active_view.pack_forget()
    target.pack(fill=tk.BOTH, expand=True)
    self._active_view = target

  def _toggle_log_drawer(self, force: bool | None = None) -> None:
    show = (not self._log_visible) if force is None else bool(force)
    if show == self._log_visible:
      return
    self._log_visible = show
    if show:
      self._log_container.pack(fill=tk.X, before=self._status_sep)
      self._btn_toggle_log.configure(text="收起日志 ▼")
    else:
      self._log_container.pack_forget()
      self._btn_toggle_log.configure(text="展开日志 ▲")

  def _clear_log(self) -> None:
    self._log_text.configure(state=tk.NORMAL)
    self._log_text.delete("1.0", tk.END)
    self._log_text.configure(state=tk.DISABLED)

  def _append_log(self, line: str) -> None:
    self._log_text.configure(state=tk.NORMAL)
    self._log_text.insert(tk.END, line)
    if not line.endswith("\n"):
      self._log_text.insert(tk.END, "\n")
    self._log_text.see(tk.END)
    self._log_text.configure(state=tk.DISABLED)

  def _populate_tree(self) -> None:
    for item in self._tree.get_children():
      self._tree.delete(item)

    for strat in self._cfg.strategies:
      s_iid = f"strat:{strat.id}"
      mode_lbl = RUN_MODE_LABELS.get(strat.default_mode, strat.default_mode)
      self._tree.insert("", tk.END, iid=s_iid, text=f"{strat.name} [{mode_lbl}]", open=True)
      for sc in strat.shortcuts:
        sc_iid = f"sc:{strat.id}:{sc.id}"
        sc_lbl = RUN_MODE_LABELS.get(sc.mode, sc.mode)
        self._tree.insert(s_iid, tk.END, iid=sc_iid, text=f"▸ {sc.name} [{sc_lbl}]")

    # Restore tree selection if still valid
    if self._selected_strategy_id and self._selected_shortcut_id:
      iid = f"sc:{self._selected_strategy_id}:{self._selected_shortcut_id}"
      if self._tree.exists(iid):
        self._tree.selection_set(iid)
        self._tree.focus(iid)
        return
    if self._selected_strategy_id:
      iid = f"strat:{self._selected_strategy_id}"
      if self._tree.exists(iid):
        self._tree.selection_set(iid)
        self._tree.focus(iid)

  def _find_strategy(self, strat_id: str | None) -> StrategyConfig | None:
    if not strat_id:
      return None
    for s in self._cfg.strategies:
      if s.id == strat_id:
        return s
    return None

  def _find_shortcut(self, strat_id: str | None, sc_id: str | None) -> ShortcutConfig | None:
    strat = self._find_strategy(strat_id)
    if strat is None or not sc_id:
      return None
    for sc in strat.shortcuts:
      if sc.id == sc_id:
        return sc
    return None

  def _get_metadata(self, strat: StrategyConfig, force_reload: bool = False) -> StrategyMetadata | None:
    if not force_reload and strat.path in self._meta_cache:
      return self._meta_cache[strat.path]
    try:
      meta = inspect_strategy_file(strat.path)
      self._meta_cache[strat.path] = meta
      return meta
    except Exception:
      return None

  def _on_tree_select(self, _event=None) -> None:
    sel = self._tree.selection()
    if not sel:
      self._selected_strategy_id = None
      self._selected_shortcut_id = None
    else:
      iid = sel[0]
      if iid.startswith("strat:"):
        self._selected_strategy_id = iid.split(":", 1)[1]
        self._selected_shortcut_id = None
      elif iid.startswith("sc:"):
        _, sid, scid = iid.split(":", 2)
        self._selected_strategy_id = sid
        self._selected_shortcut_id = scid
    self._update_selection_state()

  def _on_tree_double_click(self, _event=None) -> None:
    if self._selected_shortcut_id is not None:
      self._run_selected()

  def _on_tree_right_click(self, event) -> None:
    iid = self._tree.identify_row(event.y)
    if iid:
      self._tree.selection_set(iid)
      self._on_tree_select()
    self._ctx_menu.delete(0, tk.END)
    if self._selected_shortcut_id:
      self._ctx_menu.add_command(label="▶ 运行快捷路径", command=self._run_selected)
      self._ctx_menu.add_command(label="编辑快捷路径", command=self._edit_shortcut)
      self._ctx_menu.add_command(label="复制快捷路径", command=self._duplicate_shortcut)
      self._ctx_menu.add_command(label="删除快捷路径", command=self._delete_shortcut)
      self._ctx_menu.add_separator()
    if self._selected_strategy_id:
      self._ctx_menu.add_command(label="+ 新建快捷路径", command=self._add_shortcut)
      self._ctx_menu.add_command(label="重命名策略", command=self._rename_strategy)
      self._ctx_menu.add_command(label="移除策略", command=self._remove_strategy)
    else:
      self._ctx_menu.add_command(label="+ 导入策略", command=self._import_strategy)
    self._ctx_menu.tk_popup(event.x_root, event.y_root)

  def _update_selection_state(self) -> None:
    running = self._proc is not None and self._proc.poll() is None
    strat = self._find_strategy(self._selected_strategy_id)
    sc = self._find_shortcut(self._selected_strategy_id, self._selected_shortcut_id)

    has_strat = strat is not None
    has_sc = sc is not None

    self._btn_rename.state(["!disabled"] if has_strat else ["disabled"])
    self._btn_remove.state(["!disabled"] if has_strat else ["disabled"])
    self._btn_add_sc.state(["!disabled"] if has_strat else ["disabled"])
    self._btn_edit_sc.state(["!disabled"] if has_sc else ["disabled"])
    self._btn_dup_sc.state(["!disabled"] if has_sc else ["disabled"])
    self._btn_del_sc.state(["!disabled"] if has_sc else ["disabled"])
    self._btn_run.state(["!disabled"] if (has_sc and not running) else ["disabled"])
    self._btn_stop.state(["!disabled"] if running else ["disabled"])

    if strat is not None and sc is not None:
      self._title_var.set(f"{strat.name}  /  {sc.name}")
      self._detail_var.set(f"{_format_shortcut_summary(sc)}  |  文件: {strat.path}")
    elif strat is not None:
      self._title_var.set(strat.name)
      self._detail_var.set(
        f"文件: {strat.path}  (请选择或新建一个快捷路径以运行)"
      )
    else:
      self._title_var.set("请选择或导入策略")
      self._detail_var.set(f"配置文件: {get_config_file()}")

  def _import_strategy(self) -> None:
    init_dir = self._cfg.last_dir or os.getcwd()
    path = filedialog.askopenfilename(
      parent=self.root,
      title="选择策略 Python 文件",
      initialdir=init_dir,
      filetypes=[("Python 策略文件", "*.py"), ("All Files", "*.*")],
    )
    if not path:
      return

    abs_path = os.path.abspath(path)
    self._cfg.last_dir = os.path.dirname(abs_path)

    try:
      meta = inspect_strategy_file(abs_path)
    except Exception as exc:
      messagebox.showerror("策略导入失败", str(exc), parent=self.root)
      return

    self._meta_cache[abs_path] = meta
    default_name = Path(abs_path).stem

    # Build initial default shortcut from detected metadata
    default_params = {
      p.name: p.default_str
      for p in meta.params_for_mode(meta.default_mode)
      if p.default_str != ""
    }
    default_objs = "A_STOCKS_EXCLUDE_ST" if meta.default_mode == RUN_MODE_PORTFOLIO else ""
    if meta.default_mode == RUN_MODE_TIMING:
      default_objs = "SH600000"

    starter = ShortcutConfig(
      name="默认运行",
      mode=meta.default_mode,
      objs=default_objs,
      strategy_args=default_params,
    )
    strat = StrategyConfig(
      name=default_name,
      path=abs_path,
      default_mode=meta.default_mode,
      shortcuts=[starter],
    )
    self._cfg.strategies.append(strat)
    self._save_cfg()
    self._selected_strategy_id = strat.id
    self._selected_shortcut_id = starter.id
    self._populate_tree()
    self._update_selection_state()

  def _rename_strategy(self) -> None:
    strat = self._find_strategy(self._selected_strategy_id)
    if strat is None:
      return
    dlg = RenameDialog(self.root, strat.name)
    self.root.wait_window(dlg)
    if dlg.result:
      strat.name = dlg.result
      self._save_cfg()
      self._populate_tree()
      self._update_selection_state()

  def _remove_strategy(self) -> None:
    strat = self._find_strategy(self._selected_strategy_id)
    if strat is None:
      return
    if not messagebox.askyesno(
      "确认移除",
      f"确定从列表移除策略“{strat.name}”吗？（不会删除磁盘上的 .py 文件）",
      parent=self.root,
    ):
      return
    self._cfg.strategies = [s for s in self._cfg.strategies if s.id != strat.id]
    self._selected_strategy_id = None
    self._selected_shortcut_id = None
    self._save_cfg()
    self._populate_tree()
    self._update_selection_state()

  def _add_shortcut(self) -> None:
    strat = self._find_strategy(self._selected_strategy_id)
    if strat is None:
      return
    meta = self._get_metadata(strat, force_reload=True)
    mode = meta.default_mode if meta else strat.default_mode
    default_params = (
      {p.name: p.default_str for p in meta.params_for_mode(mode) if p.default_str != ""}
      if meta
      else {}
    )
    draft = ShortcutConfig(
      name=f"运行方案 {len(strat.shortcuts) + 1}",
      mode=mode,
      objs="A_STOCKS_EXCLUDE_ST" if mode == RUN_MODE_PORTFOLIO else ("SH600000" if mode == RUN_MODE_TIMING else ""),
      strategy_args=default_params,
    )
    dlg = ShortcutDialog(self.root, draft, meta, title=f"新建快捷路径 - {strat.name}")
    self.root.wait_window(dlg)
    if dlg.result is not None:
      strat.shortcuts.append(dlg.result)
      self._save_cfg()
      self._selected_shortcut_id = dlg.result.id
      self._populate_tree()
      self._update_selection_state()

  def _edit_shortcut(self) -> None:
    strat = self._find_strategy(self._selected_strategy_id)
    sc = self._find_shortcut(self._selected_strategy_id, self._selected_shortcut_id)
    if strat is None or sc is None:
      return
    meta = self._get_metadata(strat, force_reload=True)
    dlg = ShortcutDialog(
      self.root, dataclasses.replace(sc), meta, title=f"编辑快捷路径 - {strat.name}"
    )
    self.root.wait_window(dlg)
    if dlg.result is not None:
      for idx, item in enumerate(strat.shortcuts):
        if item.id == sc.id:
          strat.shortcuts[idx] = dlg.result
          break
      self._save_cfg()
      self._populate_tree()
      self._update_selection_state()

  def _duplicate_shortcut(self) -> None:
    strat = self._find_strategy(self._selected_strategy_id)
    sc = self._find_shortcut(self._selected_strategy_id, self._selected_shortcut_id)
    if strat is None or sc is None:
      return
    copy_sc = ShortcutConfig.from_dict(sc.to_dict())
    copy_sc.id = uuid.uuid4().hex[:12]
    copy_sc.name = f"{sc.name} (副本)"
    strat.shortcuts.append(copy_sc)
    self._save_cfg()
    self._selected_shortcut_id = copy_sc.id
    self._populate_tree()
    self._update_selection_state()

  def _delete_shortcut(self) -> None:
    strat = self._find_strategy(self._selected_strategy_id)
    sc = self._find_shortcut(self._selected_strategy_id, self._selected_shortcut_id)
    if strat is None or sc is None:
      return
    if not messagebox.askyesno(
      "确认删除", f"确定删除快捷路径“{sc.name}”吗？", parent=self.root
    ):
      return
    strat.shortcuts = [item for item in strat.shortcuts if item.id != sc.id]
    self._selected_shortcut_id = None
    self._save_cfg()
    self._populate_tree()
    self._update_selection_state()

  def _run_selected(self) -> None:
    if self._proc is not None and self._proc.poll() is None:
      messagebox.showinfo("提示", "已有任务正在运行，请等待完成或先点击停止。", parent=self.root)
      return

    strat = self._find_strategy(self._selected_strategy_id)
    sc = self._find_shortcut(self._selected_strategy_id, self._selected_shortcut_id)
    if strat is None or sc is None:
      return
    if not os.path.isfile(strat.path):
      messagebox.showerror("文件不存在", f"策略文件不存在:\n{strat.path}", parent=self.root)
      return

    self._clear_log()
    self._cancelled = False
    self._run_start_ts = time.perf_counter()
    self._temp_dir = tempfile.TemporaryDirectory(prefix="mbt_gui_")
    spec_path = os.path.join(self._temp_dir.name, "spec.json")
    out_path = os.path.join(self._temp_dir.name, "result.pkl")

    spec = {
      "strategy_path": strat.path,
      "mode": sc.mode,
      "objs": sc.objs,
      "use_internal_selector": sc.use_internal_selector,
      "begin": sc.begin,
      "end": sc.end,
      "select_date": sc.select_date,
      "benchmark": sc.benchmark,
      "cash": sc.cash,
      "exec_mode": sc.exec_mode,
      "warmup_bars": sc.warmup_bars,
      "msd_host": sc.msd_host,
      "strategy_args": sc.strategy_args,
      "extra_args": sc.extra_args,
    }
    with open(spec_path, "w", encoding="utf-8") as f:
      json.dump(spec, f, ensure_ascii=False, indent=2)

    cmd = [sys.executable, "-u", "-m", "mbt.gui.worker", "--spec", spec_path, "--out", out_path]
    self._append_log(f"$ {' '.join(cmd)}")
    self._status_var.set(f"运行中: {strat.name} / {sc.name} ...")

    self._proc = subprocess.Popen(
      cmd,
      stdout=subprocess.PIPE,
      stderr=subprocess.STDOUT,
      text=True,
      encoding="utf-8",
      errors="replace",
      bufsize=1,
    )
    self._update_selection_state()

    def _reader_thread(proc: subprocess.Popen[str], result_file: str) -> None:
      assert proc.stdout is not None
      for line in proc.stdout:
        self._msg_queue.put(("log", line))
      rc = proc.wait()
      self._msg_queue.put(("done", (rc, result_file)))

    threading.Thread(
      target=_reader_thread, args=(self._proc, out_path), daemon=True
    ).start()
    self.root.after(80, self._poll_worker)

  def _stop_run(self) -> None:
    if self._proc is None or self._proc.poll() is not None:
      return
    self._cancelled = True
    self._append_log("[GUI] 正在终止任务...")
    try:
      self._proc.terminate()
      try:
        self._proc.wait(timeout=2.0)
      except subprocess.TimeoutExpired:
        self._proc.kill()
    except Exception as exc:
      self._append_log(f"[GUI] 停止异常: {exc}")

  def _poll_worker(self) -> None:
    finished = False
    done_payload = None
    while True:
      try:
        kind, payload = self._msg_queue.get_nowait()
      except queue.Empty:
        break
      if kind == "log":
        self._append_log(payload)
      elif kind == "done":
        finished = True
        done_payload = payload

    if not finished or done_payload is None:
      elapsed = time.perf_counter() - self._run_start_ts
      self._status_var.set(f"运行中... ({elapsed:.1f}s)")
      self.root.after(80, self._poll_worker)
      return

    rc, out_path = done_payload
    elapsed = time.perf_counter() - self._run_start_ts
    self._proc = None
    self._update_selection_state()

    if self._cancelled:
      self._status_var.set(f"已手动停止 ({elapsed:.1f}s)")
      self._cleanup_temp()
      return

    if rc != 0 or not os.path.isfile(out_path):
      self._status_var.set(f"运行失败 (退出码 {rc}, 耗时 {elapsed:.1f}s)")
      self._toggle_log_drawer(force=True)
      self._cleanup_temp()
      messagebox.showerror(
        "运行失败",
        f"策略执行失败 (退出码 {rc})，请查看底部运行日志定位原因。",
        parent=self.root,
      )
      return

    try:
      with open(out_path, "rb") as f:
        res = pickle.load(f)
    except Exception as exc:
      self._status_var.set("结果解析失败")
      self._cleanup_temp()
      messagebox.showerror("结果解析失败", str(exc), parent=self.root)
      return
    finally:
      self._cleanup_temp()

    mode = res.get("mode")
    if mode == RUN_MODE_SELECT:
      self._switch_view(self._select_view)
      self._select_view.show_result(res)
    elif mode == RUN_MODE_TIMING:
      self._switch_view(self._timing_view)
      self._timing_view.show_result(res)
    elif mode == RUN_MODE_PORTFOLIO:
      self._switch_view(self._portfolio_view)
      self._portfolio_view.show_result(res)
    else:
      messagebox.showerror("未知结果类型", f"不支持的结果模式: {mode}", parent=self.root)
      return

    self._status_var.set(f"运行完成 ({elapsed:.2f}s)")

  def _cleanup_temp(self) -> None:
    if self._temp_dir is not None:
      try:
        self._temp_dir.cleanup()
      except Exception:
        pass
      self._temp_dir = None

  def _on_close(self) -> None:
    if self._proc is not None and self._proc.poll() is None:
      self._stop_run()
    self._cleanup_temp()
    self.root.destroy()
