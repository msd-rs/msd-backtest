import { useEffect, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import { RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import {
  type BarZoomWindow,
  countVisiblePoints,
  escapeHtml,
  lookupDateByBarIdx,
  mergeCurveData,
  readDataZoomWindow,
} from "@/lib/curveZoom";
import type { CurveQueryResult } from "@/types/api";

interface EquityChartProps {
  runId: string;
  obj?: string;
  title: string;
  origLabel?: string;
  benchLabel?: string;
  darkMode: boolean;
  fillHeight?: boolean;
}

interface TooltipSeriesItem {
  seriesName?: string;
  marker?: string;
  axisValue?: number;
  value?: unknown;
}

export function EquityChart({
  runId,
  obj,
  title,
  origLabel = "买入持有",
  benchLabel = "基准",
  darkMode,
  fillHeight = false,
}: EquityChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<echarts.ECharts | null>(null);
  const zoomRef = useRef<BarZoomWindow | null>(null);

  const [overview, setOverview] = useState<CurveQueryResult | null>(null);
  const [detail, setDetail] = useState<CurveQueryResult | null>(null);
  const [detailReq, setDetailReq] = useState<BarZoomWindow | null>(null);
  const [visibleWindow, setVisibleWindow] = useState<BarZoomWindow | null>(null);

  const [showOrig, setShowOrig] = useState(true);
  const [showBench, setShowBench] = useState(true);
  const [showDd, setShowDd] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    zoomRef.current = null;
    setVisibleWindow(null);
    setDetailReq(null);
    setDetail(null);
    setOverview(null);
    setLoading(true);
    setError("");
    api
      .queryCurve(runId, {
        obj,
        start_idx: 0,
        max_points: 10000,
      })
      .then((res) => {
        if (!cancelled) setOverview(res);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, obj]);

  useEffect(() => {
    if (!overview?.downsampled || !detailReq) return;
    let cancelled = false;
    setLoading(true);
    api
      .queryCurve(runId, {
        obj,
        start_idx: detailReq.start,
        end_idx: detailReq.end,
        max_points: 10000,
      })
      .then((res) => {
        if (!cancelled) setDetail(res);
      })
      .catch((err) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, obj, overview, detailReq]);

  const merged = useMemo(
    () => (overview ? mergeCurveData(overview, detail) : null),
    [overview, detail],
  );

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const chart = echarts.init(el);
    chartRef.current = chart;
    const obs = new ResizeObserver(() => chart.resize());
    obs.observe(el);
    return () => {
      obs.disconnect();
      chart.dispose();
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    const chart = chartRef.current;
    if (!chart || !merged) return;

    const textColor = darkMode ? "#cbd5e1" : "#334155";
    const gridLine = darkMode ? "rgba(148,163,184,0.16)" : "rgba(148,163,184,0.28)";
    const indices = merged.indices;
    const dates = merged.dates;
    const maxIdx = Math.max(0, merged.totalPoints - 1);
    const axisMax = Math.max(1, maxIdx);

    const toPairs = (raw: (number | null)[] | undefined): [number, number | null][] =>
      (raw || []).map((v, i) => [
        indices[i] ?? i,
        v === null || v === undefined ? null : Number((v * 100).toFixed(3)),
      ]);

    const rorPairs = toPairs(merged.series.ror);
    const origPairs = toPairs(merged.series.ror_original);
    const benchPairs = toPairs(merged.series.benchmark_ror);
    const ddPairs = toPairs(merged.series.drawdown);

    const series: echarts.SeriesOption[] = [
      {
        name: "策略收益",
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: rorPairs,
        showSymbol: false,
        lineStyle: { width: 2, color: "#2563eb" },
        itemStyle: { color: "#2563eb" },
      },
    ];
    if (showOrig && origPairs.length === indices.length) {
      series.push({
        name: origLabel,
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: origPairs,
        showSymbol: false,
        lineStyle: { width: 1.4, type: "dashed", color: "#d97706" },
        itemStyle: { color: "#d97706" },
      });
    }
    if (showBench && benchPairs.length === indices.length) {
      series.push({
        name: benchLabel,
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data: benchPairs,
        showSymbol: false,
        lineStyle: { width: 1.4, color: "#64748b" },
        itemStyle: { color: "#64748b" },
      });
    }
    if (showDd && ddPairs.length === indices.length) {
      series.push({
        name: "回撤",
        type: "line",
        xAxisIndex: 1,
        yAxisIndex: 1,
        data: ddPairs,
        showSymbol: false,
        lineStyle: { width: 1.2, color: "#dc2626" },
        itemStyle: { color: "#dc2626" },
        areaStyle: { color: "rgba(220,38,38,0.22)" },
      });
    }

    const formatXLabel = (val: number) => {
      if (val < -0.5 || val > maxIdx + 0.5) return "";
      return lookupDateByBarIdx(indices, dates, Math.round(val));
    };
    const formatXPointer = (p: { value: unknown }) =>
      lookupDateByBarIdx(indices, dates, Math.round(Number(p.value)));
    const formatYPointer = (p: { value: unknown }) => {
      const n = Number(p.value);
      return Number.isFinite(n) ? `${n.toFixed(2)}%` : String(p.value ?? "");
    };

    const grids = showDd
      ? fillHeight
        ? [
          { left: 58, right: 20, top: 28, bottom: "39%" },
          { left: 58, right: 20, top: "65%", bottom: 46 },
        ]
        : [
          { left: 58, right: 20, top: 34, height: "49%" },
          { left: 58, right: 20, top: "63%", height: "18%" },
        ]
      : fillHeight
        ? [{ left: 58, right: 20, top: 28, bottom: 46 }]
        : [{ left: 58, right: 20, top: 34, bottom: 58 }];

    const xAxes: echarts.XAXisComponentOption[] = showDd
      ? [
        {
          type: "value",
          gridIndex: 0,
          min: 0,
          max: axisMax,
          minInterval: 1,
          axisLabel: { show: false },
          axisTick: { show: false },
          axisLine: { lineStyle: { color: gridLine } },
          splitLine: { show: false },
          axisPointer: { label: { show: false } },
        },
        {
          type: "value",
          gridIndex: 1,
          min: 0,
          max: axisMax,
          minInterval: 1,
          axisLabel: {
            color: textColor,
            fontSize: 10,
            hideOverlap: true,
            formatter: formatXLabel,
          },
          axisLine: { lineStyle: { color: gridLine } },
          splitLine: { show: false },
          axisPointer: { label: { formatter: formatXPointer } },
        },
      ]
      : [
        {
          type: "value",
          gridIndex: 0,
          min: 0,
          max: axisMax,
          minInterval: 1,
          axisLabel: {
            color: textColor,
            fontSize: 10,
            hideOverlap: true,
            formatter: formatXLabel,
          },
          axisLine: { lineStyle: { color: gridLine } },
          splitLine: { show: false },
          axisPointer: { label: { formatter: formatXPointer } },
        },
      ];

    const yAxes: echarts.YAXisComponentOption[] = showDd
      ? [
        {
          type: "value",
          gridIndex: 0,
          scale: true,
          axisLabel: { color: textColor, fontSize: 10, formatter: "{value}%" },
          splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
          axisPointer: { label: { formatter: formatYPointer } },
        },
        {
          type: "value",
          gridIndex: 1,
          scale: true,
          axisLabel: { color: textColor, fontSize: 10, formatter: "{value}%" },
          splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
          axisPointer: { label: { formatter: formatYPointer } },
        },
      ]
      : [
        {
          type: "value",
          gridIndex: 0,
          scale: true,
          axisLabel: { color: textColor, fontSize: 10, formatter: "{value}%" },
          splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
          axisPointer: { label: { formatter: formatYPointer } },
        },
      ];

    const dzStart = zoomRef.current
      ? Math.max(0, Math.min(maxIdx, zoomRef.current.start))
      : 0;
    const dzEnd = zoomRef.current
      ? Math.max(dzStart, Math.min(maxIdx, zoomRef.current.end))
      : maxIdx;
    const minSpan = Math.min(5, maxIdx);
    const zoomAxes = showDd ? [0, 1] : [0];

    chart.setOption(
      {
        animation: false,
        tooltip: {
          trigger: "axis",
          confine: true,
          axisPointer: { type: "cross" },
          formatter: (rawParams: unknown) => {
            const items = (
              Array.isArray(rawParams) ? rawParams : [rawParams]
            ) as TooltipSeriesItem[];
            if (!items.length) return "";
            const first = items[0];
            const rawX =
              typeof first.axisValue === "number"
                ? first.axisValue
                : Array.isArray(first.value) && typeof first.value[0] === "number"
                  ? first.value[0]
                  : 0;
            const dateStr = lookupDateByBarIdx(indices, dates, Math.round(rawX));
            const order = ["策略收益", origLabel, benchLabel, "回撤"];
            const byName = new Map<string, TooltipSeriesItem>();
            for (const it of items) {
              if (it.seriesName && !byName.has(it.seriesName)) {
                byName.set(it.seriesName, it);
              }
            }
            const sorted = Array.from(byName.values()).sort((a, b) => {
              const ia = order.indexOf(a.seriesName ?? "");
              const ib = order.indexOf(b.seriesName ?? "");
              return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
            });
            const rows = sorted
              .map((it) => {
                const y = Array.isArray(it.value) ? it.value[1] : it.value;
                const valText =
                  typeof y === "number" && Number.isFinite(y)
                    ? `${y.toFixed(2)}%`
                    : "-";
                const name = escapeHtml(it.seriesName ?? "");
                const marker = it.marker ?? "";
                return `<div style="display:flex;justify-content:space-between;gap:14px;"><span>${marker}${name}</span><span style="font-weight:600;font-variant-numeric:tabular-nums;">${valText}</span></div>`;
              })
              .join("");
            return `<div style="font-size:12px;line-height:1.5;"><div style="margin-bottom:4px;font-weight:600;">${escapeHtml(dateStr)}</div>${rows}</div>`;
          },
        },
        axisPointer: showDd ? { link: [{ xAxisIndex: "all" }] } : undefined,
        legend: {
          top: fillHeight ? 2 : 4,
          textStyle: { color: textColor, fontSize: 11 },
        },
        grid: grids,
        xAxis: xAxes,
        yAxis: yAxes,
        dataZoom: [
          {
            type: "inside",
            xAxisIndex: zoomAxes,
            filterMode: "filter",
            startValue: dzStart,
            endValue: dzEnd,
            minValueSpan: minSpan,
          },
          {
            type: "slider",
            xAxisIndex: zoomAxes,
            filterMode: "filter",
            height: fillHeight ? 18 : 20,
            bottom: fillHeight ? 4 : 6,
            startValue: dzStart,
            endValue: dzEnd,
            minValueSpan: minSpan,
            labelFormatter: (val: number) =>
              lookupDateByBarIdx(indices, dates, Math.round(val)),
          },
        ],
        series,
      },
      true,
    );

    let zoomTimer: ReturnType<typeof setTimeout> | null = null;
    const onZoom = () => {
      const win = readDataZoomWindow(chart, merged.totalPoints);
      const prev = zoomRef.current;
      if (prev && prev.start === win.start && prev.end === win.end) return;
      zoomRef.current = win;
      setVisibleWindow(win);
      if (!overview?.downsampled) return;
      if (zoomTimer) clearTimeout(zoomTimer);
      zoomTimer = setTimeout(() => {
        if (win.start <= 0 && win.end >= maxIdx) {
          setDetailReq(null);
          setDetail(null);
        } else if (win.end - win.start >= 5) {
          setDetailReq({ start: win.start, end: win.end + 1 });
        }
      }, 240);
    };
    chart.off("datazoom");
    chart.on("datazoom", onZoom);
    return () => {
      if (zoomTimer) clearTimeout(zoomTimer);
      chart.off("datazoom", onZoom);
    };
  }, [merged, overview, showOrig, showBench, showDd, origLabel, benchLabel, darkMode, fillHeight]);

  const maxIdx = merged ? Math.max(0, merged.totalPoints - 1) : 0;
  const visStart = visibleWindow ? visibleWindow.start : 0;
  const visEnd = visibleWindow ? visibleWindow.end : maxIdx;
  const isZoomed =
    merged !== null &&
    merged.totalPoints > 1 &&
    visibleWindow !== null &&
    (visibleWindow.start > 0 || visibleWindow.end < maxIdx);

  const visibleStartDate = merged
    ? lookupDateByBarIdx(merged.indices, merged.dates, visStart)
    : "";
  const visibleEndDate = merged
    ? lookupDateByBarIdx(merged.indices, merged.dates, visEnd)
    : "";
  const visibleCount = merged
    ? countVisiblePoints(merged.indices, visStart, visEnd)
    : 0;

  const handleResetZoom = () => {
    zoomRef.current = null;
    setVisibleWindow(null);
    setDetailReq(null);
    setDetail(null);
    chartRef.current?.dispatchAction({
      type: "dataZoom",
      startValue: 0,
      endValue: maxIdx,
    });
  };

  return (
    <div
      className={
        fillHeight
          ? "flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-2.5"
          : "rounded-lg border bg-card p-3"
      }
    >
      <div
        className={
          fillHeight
            ? "mb-1.5 flex shrink-0 flex-wrap items-center justify-between gap-2"
            : "mb-2 flex flex-wrap items-center justify-between gap-2"
        }
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">{title}</span>
          {merged && (
            <span className="text-[11px] text-muted-foreground">
              {visibleStartDate} ~ {visibleEndDate} · 显示{" "}
              {visibleCount.toLocaleString()} / {merged.totalPoints.toLocaleString()} 点
              {merged.downsampled ? " (Min-Max 降采样)" : ""}
            </span>
          )}
          {loading && <span className="text-[11px] text-primary">加载中...</span>}
        </div>

        <div className="flex flex-wrap items-center gap-3 text-xs">
          <label className="inline-flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showOrig}
              onChange={(e) => setShowOrig(e.target.checked)}
            />
            <span>{origLabel}</span>
          </label>
          <label className="inline-flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showBench}
              onChange={(e) => setShowBench(e.target.checked)}
            />
            <span>{benchLabel}</span>
          </label>
          <label className="inline-flex items-center gap-1 cursor-pointer">
            <input
              type="checkbox"
              checked={showDd}
              onChange={(e) => setShowDd(e.target.checked)}
            />
            <span>回撤</span>
          </label>
          <Button
            variant="outline"
            size="sm"
            className="h-6 px-2 text-[11px]"
            onClick={handleResetZoom}
            disabled={!isZoomed}
          >
            <RotateCcw className="h-3 w-3" aria-hidden="true" />
            重置缩放
          </Button>
        </div>
      </div>

      {error && (
        <div role="alert" className="py-10 text-center text-xs text-destructive">
          {error}
        </div>
      )}
      <div
        ref={containerRef}
        className={
          error ? "hidden" : fillHeight ? "min-h-0 w-full flex-1" : "h-87.5 w-full"
        }
      />
    </div>
  );
}
