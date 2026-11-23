import type * as echarts from "echarts";
import type { CurveQueryResult } from "@/types/api";

export interface MergedCurveData {
  totalPoints: number;
  indices: number[];
  dates: string[];
  series: Record<string, (number | null)[]>;
  downsampled: boolean;
}

export interface BarZoomWindow {
  start: number;
  end: number;
}

/**
 * Merge a full-range overview curve with an optional zoomed-in high-resolution
 * detail slice while keeping bar indices sorted and covering the full timeline.
 */
export function mergeCurveData(
  overview: CurveQueryResult,
  detail: CurveQueryResult | null,
): MergedCurveData {
  if (!detail || !overview.downsampled || overview.total_points === 0) {
    return {
      totalPoints: overview.total_points,
      indices: overview.indices,
      dates: overview.dates,
      series: overview.series,
      downsampled: overview.downsampled,
    };
  }

  const allKeys = Array.from(
    new Set([...Object.keys(overview.series), ...Object.keys(detail.series)]),
  );
  const indices: number[] = [];
  const dates: string[] = [];
  const series: Record<string, (number | null)[]> = {};
  for (const key of allKeys) {
    series[key] = [];
  }

  let lastIdx = -1;
  const pushFrom = (src: CurveQueryResult, i: number) => {
    const barIdx = src.indices[i];
    if (barIdx <= lastIdx) return;
    lastIdx = barIdx;
    indices.push(barIdx);
    dates.push(src.dates[i] ?? "");
    for (const key of allKeys) {
      series[key].push(src.series[key]?.[i] ?? null);
    }
  };

  for (let i = 0; i < overview.indices.length; i++) {
    if (overview.indices[i] < detail.start_idx) {
      pushFrom(overview, i);
    }
  }
  for (let i = 0; i < detail.indices.length; i++) {
    pushFrom(detail, i);
  }
  for (let i = 0; i < overview.indices.length; i++) {
    if (overview.indices[i] >= detail.end_idx) {
      pushFrom(overview, i);
    }
  }

  return {
    totalPoints: overview.total_points,
    indices,
    dates,
    series,
    downsampled: overview.downsampled || detail.downsampled,
  };
}

/**
 * Binary-search the closest date label for a given integer `bar_idx`.
 */
export function lookupDateByBarIdx(
  indices: number[],
  dates: string[],
  targetIdx: number,
): string {
  const n = indices.length;
  if (n === 0 || dates.length === 0) return "";
  if (targetIdx <= indices[0]) return dates[0] ?? "";
  if (targetIdx >= indices[n - 1]) return dates[n - 1] ?? "";

  let lo = 0;
  let hi = n - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    const midVal = indices[mid];
    if (midVal === targetIdx) return dates[mid] ?? "";
    if (midVal < targetIdx) lo = mid + 1;
    else hi = mid - 1;
  }

  if (hi < 0) return dates[0] ?? "";
  if (lo >= n) return dates[n - 1] ?? "";
  const dHi = Math.abs(targetIdx - indices[hi]);
  const dLo = Math.abs(indices[lo] - targetIdx);
  return (dHi <= dLo ? dates[hi] : dates[lo]) ?? "";
}

interface EChartsZoomOptionEntry {
  startValue?: unknown;
  endValue?: unknown;
  start?: number;
  end?: number;
}

/**
 * Read the current clamped `[start, end]` bar index window from an ECharts instance.
 */
export function readDataZoomWindow(
  chart: echarts.ECharts,
  totalPoints: number,
): BarZoomWindow {
  const maxIdx = Math.max(0, totalPoints - 1);
  if (maxIdx === 0) return { start: 0, end: 0 };

  const opt = chart.getOption() as { dataZoom?: EChartsZoomOptionEntry[] };
  const dzList = opt.dataZoom ?? [];
  const dz =
    dzList.find(
      (z) =>
        (typeof z.startValue === "number" && Number.isFinite(z.startValue)) ||
        (typeof z.start === "number" && Number.isFinite(z.start)),
    ) ?? dzList[0];

  if (!dz) return { start: 0, end: maxIdx };

  if (
    typeof dz.startValue === "number" &&
    Number.isFinite(dz.startValue) &&
    typeof dz.endValue === "number" &&
    Number.isFinite(dz.endValue)
  ) {
    const start = Math.max(0, Math.min(maxIdx, Math.floor(dz.startValue)));
    const end = Math.max(start, Math.min(maxIdx, Math.ceil(dz.endValue)));
    return { start, end };
  }

  const startPct = typeof dz.start === "number" && Number.isFinite(dz.start) ? dz.start : 0;
  const endPct = typeof dz.end === "number" && Number.isFinite(dz.end) ? dz.end : 100;
  const start = Math.max(0, Math.min(maxIdx, Math.round((startPct / 100) * maxIdx)));
  const end = Math.max(start, Math.min(maxIdx, Math.round((endPct / 100) * maxIdx)));
  return { start, end };
}

/**
 * Count how many points fall inside `[start, end]`.
 */
export function countVisiblePoints(
  indices: number[],
  start: number,
  end: number,
): number {
  let count = 0;
  for (let i = 0; i < indices.length; i++) {
    const v = indices[i];
    if (v >= start && v <= end) count++;
  }
  return count;
}

/**
 * Escape HTML special characters for ECharts tooltip strings.
 */
export function escapeHtml(text: string): string {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}
