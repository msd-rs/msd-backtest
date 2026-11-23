# Copyright 2026 MSD-RS Project LiJia
# SPDX-License-Identifier: BSD-2-Clause

"""High-performance Min-Max envelope downsampling for large financial curves."""

from __future__ import annotations

import numpy as np


def minmax_downsample(
  y: np.ndarray,
  start_idx: int = 0,
  end_idx: int | None = None,
  max_points: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
  """Downsample ``y[start_idx:end_idx]`` preserving local peaks and troughs.

  Divides the visible slice into uniform buckets and emits the ``(min, max)``
  pair in chronological index order per bucket, plus exact slice endpoints.
  Runs in vectorized NumPy O(N) time without Python loops.

  Args:
    y: 1D float array of curve values.
    start_idx: Inclusive start bar index in ``y``.
    end_idx: Exclusive end bar index in ``y`` (defaults to ``len(y)``).
    max_points: Target maximum number of output points (>= 4).

  Returns:
    Tuple of ``(x_indices, y_values)`` where ``x_indices`` are strictly
    increasing bar indices in ``[start_idx, end_idx)``.
  """
  n_total = len(y)
  if n_total == 0:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

  s = max(0, int(start_idx))
  e = n_total if end_idx is None else min(n_total, int(end_idx))
  if s >= e:
    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

  max_pts = max(4, int(max_points))
  span = e - s
  if span <= max_pts:
    xs = np.arange(s, e, dtype=np.int64)
    return xs, np.asarray(y[s:e], dtype=np.float64)

  sub = np.asarray(y[s:e], dtype=np.float64)
  n_buckets = max(1, (max_pts - 2) // 2)

  # Vectorized bucket reduction across uniform chunks plus tail remainder
  bucket_size = span // n_buckets
  usable = bucket_size * n_buckets
  reshaped = sub[:usable].reshape(n_buckets, bucket_size)

  # Replace NaNs with +inf/-inf temporarily so argmin/argmax stay well-defined
  nan_mask = ~np.isfinite(reshaped)
  if np.any(nan_mask):
    for_min = np.where(nan_mask, np.inf, reshaped)
    for_max = np.where(nan_mask, -np.inf, reshaped)
    all_nan = np.all(nan_mask, axis=1)
    rel_min = np.argmin(for_min, axis=1)
    rel_max = np.argmax(for_max, axis=1)
  else:
    all_nan = None
    rel_min = np.argmin(reshaped, axis=1)
    rel_max = np.argmax(reshaped, axis=1)

  offsets = np.arange(n_buckets, dtype=np.int64) * bucket_size
  idx_min = s + offsets + rel_min
  idx_max = s + offsets + rel_max

  # Order (min, max) chronologically within each bucket
  first_idx = np.minimum(idx_min, idx_max)
  second_idx = np.maximum(idx_min, idx_max)
  interleaved = np.empty(n_buckets * 2, dtype=np.int64)
  interleaved[0::2] = first_idx
  interleaved[1::2] = second_idx

  if all_nan is not None and np.any(all_nan):
    valid_pair = np.repeat(~all_nan, 2)
    interleaved = interleaved[valid_pair]

  # Include exact start, tail remainder, and end index
  tail_indices: list[int] = [s]
  parts = [np.array(tail_indices, dtype=np.int64), interleaved]
  if usable < span:
    rem = sub[usable:]
    rem_finite = np.isfinite(rem)
    if np.any(rem_finite):
      r_min = usable + int(np.argmin(np.where(rem_finite, rem, np.inf)))
      r_max = usable + int(np.argmax(np.where(rem_finite, rem, -np.inf)))
      rem_pair = sorted({s + r_min, s + r_max})
      parts.append(np.array(rem_pair, dtype=np.int64))
  parts.append(np.array([e - 1], dtype=np.int64))

  combined = np.concatenate(parts)
  # Deduplicate adjacent identical indices while keeping strict order
  if len(combined) > 1:
    keep = np.empty(len(combined), dtype=bool)
    keep[0] = True
    keep[1:] = combined[1:] != combined[:-1]
    combined = combined[keep]

  return combined, np.asarray(y[combined], dtype=np.float64)
