# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Entry point for running ``python -m mbt.gui``."""

from __future__ import annotations

import argparse
import tkinter as tk

from mbt.gui.app import MbtGuiApp


def main() -> None:
  """Launch the mbt Tkinter GUI application."""
  parser = argparse.ArgumentParser(description="Launch MSD Backtest Studio GUI")
  parser.add_argument(
    "--scale",
    type=float,
    default=None,
    help="Override UI scale factor (e.g. 1.5 or 2.0); defaults to auto-detection",
  )
  args = parser.parse_args()

  root = tk.Tk()
  MbtGuiApp(root, scale=args.scale)
  root.mainloop()


if __name__ == "__main__":
  main()
