# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""HiDPI / UI scaling and system font detection helpers for Tkinter."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tkinter as tk
import tkinter.font as tkfont
from tkinter import ttk


def _query_xft_dpi() -> float | None:
  if not sys.platform.startswith("linux"):
    return None
  try:
    proc = subprocess.run(
      ["xrdb", "-query"],
      capture_output=True,
      text=True,
      timeout=1.0,
      check=False,
    )
    if proc.returncode != 0:
      return None
    m = re.search(r"^\s*Xft\.dpi:\s*([0-9]+(?:\.[0-9]+)?)\s*$", proc.stdout, re.MULTILINE)
    if m:
      return float(m.group(1))
  except Exception:
    return None
  return None


def detect_system_sans_font(root: tk.Misc | None = None) -> str | None:
  """Detect the system's default CJK sans-serif font if Tk supports it."""
  if not sys.platform.startswith("linux"):
    return None
  try:
    proc = subprocess.run(
      ["fc-match", "-f", "%{family[0]}\n", "sans-serif:lang=zh-cn"],
      capture_output=True,
      text=True,
      timeout=1.0,
      check=False,
    )
    if proc.returncode != 0:
      return None
    family = proc.stdout.strip().splitlines()[0].strip() if proc.stdout.strip() else ""
    if not family:
      return None
    if root is not None:
      available = set(tkfont.families(root))
      if family in available:
        return family
      return None
    return family
  except Exception:
    return None


def tk_supports_xft_cjk(root: tk.Misc) -> bool:
  """Return True if the active Tk runtime can render system TrueType/CJK fonts."""
  f_sys = detect_system_sans_font(root)
  return f_sys is not None


def detect_ui_scale(root: tk.Misc | None = None) -> float:
  """Detect the appropriate UI scale factor (e.g. 2.0 on 2x HiDPI displays)."""
  env_scale = os.environ.get("MBT_GUI_SCALE", "").strip()
  if env_scale:
    try:
      val = float(env_scale)
      if val > 0.25:
        return val
    except ValueError:
      pass

  for var in ("QT_SCALE_FACTOR", "GDK_SCALE"):
    raw = os.environ.get(var, "").strip()
    if raw:
      try:
        val = float(raw)
        if val >= 1.15:
          return val
      except ValueError:
        pass

  xft_dpi = _query_xft_dpi()
  if xft_dpi is not None and xft_dpi >= 110.0:
    return round(xft_dpi / 96.0, 2)

  if root is not None:
    try:
      fpix = float(root.winfo_fpixels("1i"))
      if fpix >= 110.0:
        return round(fpix / 96.0, 2)
    except Exception:
      pass

  return 1.0


def apply_ui_scale(root: tk.Tk, scale: float | None = None) -> float:
  """Apply system font and UI scaling to Tk named fonts before building widgets."""
  factor = float(scale) if (scale is not None and scale > 0.25) else detect_ui_scale(root)
  setattr(root, "_mbt_ui_scale", factor)

  # Apply system default CJK font when supported by this Tk build (Xft-enabled)
  sys_family = detect_system_sans_font(root)
  if sys_family:
    for name in (
      "TkDefaultFont",
      "TkTextFont",
      "TkMenuFont",
      "TkHeadingFont",
      "TkCaptionFont",
      "TkSmallCaptionFont",
      "TkIconFont",
      "TkTooltipFont",
    ):
      try:
        f = tkfont.nametofont(name, root=root)
        f.configure(family=sys_family)
      except Exception:
        pass

  if not hasattr(root, "_mbt_orig_font_sizes"):
    orig_sizes: dict[str, int] = {}
    for name in tkfont.names(root):
      try:
        f = tkfont.nametofont(name, root=root)
        orig_sizes[name] = int(f.cget("size"))
      except Exception:
        pass
    setattr(root, "_mbt_orig_font_sizes", orig_sizes)

  # Check if Xft already applied Xft.dpi natively (positive point size + already large linespace)
  f_def = tkfont.nametofont("TkDefaultFont", root=root)
  base_linespace = int(f_def.metrics("linespace"))
  orig_def_size = int(getattr(root, "_mbt_orig_font_sizes").get("TkDefaultFont", -12))
  xft_dpi = _query_xft_dpi()
  xft_native_scale = 1.0
  if (
    orig_def_size > 0
    and xft_dpi is not None
    and xft_dpi >= 110.0
    and base_linespace >= int(round(15 * (xft_dpi / 96.0) * 0.85))
  ):
    xft_native_scale = round(xft_dpi / 96.0, 2)

  font_factor = factor / xft_native_scale
  setattr(root, "_mbt_xft_native_scale", xft_native_scale)

  if abs(font_factor - 1.0) >= 0.05:
    orig_sizes = getattr(root, "_mbt_orig_font_sizes")
    for name, orig in orig_sizes.items():
      try:
        f = tkfont.nametofont(name, root=root)
        if orig < 0:
          new_size = -max(8, int(round(abs(orig) * font_factor)))
        else:
          new_size = max(7, int(round(orig * font_factor)))
        f.configure(size=new_size)
      except Exception:
        pass

  return factor


def get_ui_scale(widget: tk.Misc | None = None) -> float:
  """Retrieve the active UI scale factor from a widget's toplevel window."""
  if widget is None:
    return 1.0
  try:
    top = widget.winfo_toplevel()
    return float(getattr(top, "_mbt_ui_scale", 1.0))
  except Exception:
    return 1.0


def scaled(px: int | float, widget: tk.Misc | None = None) -> int:
  """Scale a logical pixel dimension by the active UI scale factor."""
  return max(1, int(round(px * get_ui_scale(widget))))


def scaled_font(
  widget: tk.Misc | None,
  base_px: int = 12,
  bold: bool = False,
) -> tuple[str, int, str] | tuple[str, int]:
  """Return a Tkinter font tuple matching TkDefaultFont's family and active scale."""
  scale = get_ui_scale(widget)
  family = "sans-serif"
  size = -max(8, int(round(base_px * scale)))

  if widget is not None:
    try:
      top = widget.winfo_toplevel()
      f_def = tkfont.nametofont("TkDefaultFont", root=top)
      family = str(f_def.actual("family"))
      cur_size = int(f_def.cget("size"))
      ratio = base_px / 12.0
      if cur_size > 0:
        size = max(7, int(round(cur_size * ratio)))
      else:
        size = -max(8, int(round(abs(cur_size) * ratio)))
    except Exception:
      pass

  if bold:
    return (family, size, "bold")
  return (family, size)


def configure_scaled_ttk_style(root: tk.Tk, style: ttk.Style, scale: float) -> None:
  """Apply scaled paddings and row heights to ttk styles."""
  f_def = tkfont.nametofont("TkDefaultFont", root=root)
  line_h = int(f_def.metrics("linespace"))
  row_h = max(int(round(24 * scale)), int(round(line_h * 1.28)))
  style.configure("Treeview", rowheight=row_h)
  style.configure("Heading", font=scaled_font(root, 12, bold=True))
  style.configure("Treeview.Heading", font=scaled_font(root, 12, bold=True))
  pad_x = max(4, int(round(6 * scale)))
  pad_y = max(2, int(round(3 * scale)))
  style.configure("TButton", padding=(pad_x, pad_y))
  style.configure("TNotebook.Tab", padding=(max(6, int(round(8 * scale))), pad_y))
