# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""Allow running factor CLI via ``python -m mbt.factor``."""

import sys

from mbt.factor.cli import main

if __name__ == "__main__":
  sys.exit(main())
