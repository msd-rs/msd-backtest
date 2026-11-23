import { useEffect, useId, useMemo, useRef, useState } from "react";
import * as echarts from "echarts";
import { AlertTriangle, Download, RotateCcw } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { SelectNative } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { api } from "@/lib/api";
import {
  type BarZoomWindow,
  countVisiblePoints,
  escapeHtml,
  lookupDateByBarIdx,
  mergeCurveData,
  readDataZoomWindow,
} from "@/lib/curveZoom";
import { fmtNum, fmtPct } from "@/lib/format";
import type { CurveQueryResult, RunManifest, RunSummaryRecord } from "@/types/api";
import { RunLogsView } from "./RunLogsView";
import { ServerDataTable, type ColumnSpec } from "./ServerDataTable";

interface FactorResultViewProps {
  run: RunSummaryRecord;
  manifest: RunManifest;
  darkMode: boolean;
  onRefreshRun: () => void;
}

interface FactorMetaEntry {
  name: string;
  direction: number;
  expression: string;
  author: string;
  start_date: string;
  end_date: string;
  bars: number;
  n_objs: number;
  quantiles: number;
  entry: string;
  winsorize: string;
  standardize: string;
  neutralize: string[];
  cost_bps: number;
  benchmark: string;
  summary: Record<string, unknown>;
  turnover: {
    top_daily_turnover?: number;
    bottom_daily_turnover?: number;
    half_life_days?: number | null;
    autocorrelation?: Record<string, number>;
  };
  holdout: Record<string, unknown> | null;
  diagnostics: {
    coverage_mean?: number;
    coverage_min?: number;
    constant_cs_days?: number;
    suspended_ratio?: number;
    limit_filtered_ratio?: number;
    mktcap_corr?: number | null;
    causality_check?: { status?: string; message?: string };
    warnings?: string[];
  };
  monotonicities?: Record<string, number>;
}

interface TooltipSeriesItem {
  seriesName?: string;
  marker?: string;
  axisValue?: number;
  value?: unknown;
}

const LEADERBOARD_COLS: ColumnSpec[] = [
  { key: "factor", label: "因子", align: "left" },
  { key: "rank_ic_mean", label: "RankIC", fmt: "f4", align: "right", colorSign: true },
  { key: "rank_icir", label: "RankICIR", fmt: "f4", align: "right", colorSign: true },
  { key: "nw_t_stat", label: "NW t-stat", fmt: "f2", align: "right" },
  { key: "top_ann_excess", label: "Top超额年化", fmt: "pct", align: "right", colorSign: true },
  { key: "ls_ann_return", label: "多空年化", fmt: "pct", align: "right", colorSign: true },
  { key: "ls_sharpe", label: "多空Sharpe", fmt: "f2", align: "right" },
  { key: "monotonicity", label: "单调性", fmt: "f4", align: "right" },
  { key: "top_daily_turnover", label: "日换手", fmt: "pct_plain", align: "right" },
  { key: "half_life_days", label: "半衰期(天)", fmt: "f2", align: "right" },
  { key: "avg_coverage", label: "平均覆盖率", fmt: "pct_plain", align: "right" },
  { key: "causality", label: "因果探针", align: "center" },
];

const IC_COLS: ColumnSpec[] = [
  { key: "period", label: "周期", fmt: "int", align: "right" },
  { key: "valid_bars", label: "有效交易日", fmt: "int", align: "right" },
  { key: "rank_ic_mean", label: "RankIC Mean", fmt: "f4", align: "right", colorSign: true },
  { key: "rank_ic_std", label: "RankIC Std", fmt: "f4", align: "right" },
  { key: "rank_icir", label: "RankICIR", fmt: "f4", align: "right" },
  { key: "ann_rank_icir", label: "年化RankICIR", fmt: "f2", align: "right" },
  { key: "nw_t_stat", label: "NW t-stat", fmt: "f2", align: "right" },
  { key: "ic_mean", label: "IC Mean", fmt: "f4", align: "right", colorSign: true },
  { key: "rank_ic_positive_ratio", label: "RankIC>0占比", fmt: "pct_plain", align: "right" },
  { key: "rank_ic_sig_ratio", label: "|RankIC|>2%", fmt: "pct_plain", align: "right" },
];

const QUANTILE_COLS: ColumnSpec[] = [
  { key: "label", label: "分组", align: "left" },
  { key: "kind", label: "类型", align: "center" },
  { key: "mean_period_ret", label: "期均收益", fmt: "bps", align: "right", colorSign: true },
  { key: "ann_return", label: "年化收益", fmt: "pct", align: "right", colorSign: true },
  { key: "ann_vol", label: "年化波动", fmt: "pct_plain", align: "right" },
  { key: "sharpe", label: "Sharpe", fmt: "f2", align: "right" },
  { key: "max_drawdown", label: "最大回撤", fmt: "pct", align: "right" },
  { key: "total_return", label: "累计收益", fmt: "pct", align: "right", colorSign: true },
  { key: "daily_turnover", label: "日换手", fmt: "pct_plain", align: "right" },
];

const ANNUAL_COLS: ColumnSpec[] = [
  { key: "year", label: "年份", align: "left" },
  { key: "bars", label: "交易日", fmt: "int", align: "right" },
  { key: "rank_ic_mean", label: "RankIC", fmt: "f4", align: "right", colorSign: true },
  { key: "rank_icir", label: "RankICIR", fmt: "f4", align: "right" },
  { key: "rank_ic_positive_ratio", label: "RankIC>0", fmt: "pct_plain", align: "right" },
  { key: "top_excess_return", label: "Top超额收益", fmt: "pct", align: "right", colorSign: true },
];

const INDUSTRY_COLS: ColumnSpec[] = [
  { key: "industry", label: "行业", align: "left" },
  { key: "n_stocks", label: "股票数", fmt: "int", align: "right" },
  { key: "avg_valid_stocks", label: "平均有效数", fmt: "f2", align: "right" },
  { key: "rank_ic_mean", label: "RankIC", fmt: "f4", align: "right", colorSign: true },
  { key: "rank_icir", label: "RankICIR", fmt: "f4", align: "right" },
  { key: "ic_mean", label: "IC", fmt: "f4", align: "right" },
];

interface FactorCurvesPanelProps {
  runId: string;
  factor: string;
  period: number;
  darkMode: boolean;
}

function FactorCurvesPanel({
  runId,
  factor,
  period,
  darkMode,
}: FactorCurvesPanelProps) {
  const groupId = useId();
  const navChartEl = useRef<HTMLDivElement | null>(null);
  const navChartRef = useRef<echarts.ECharts | null>(null);
  const icChartEl = useRef<HTMLDivElement | null>(null);
  const icChartRef = useRef<echarts.ECharts | null>(null);
  const zoomRef = useRef<BarZoomWindow | null>(null);

  const [overview, setOverview] = useState<CurveQueryResult | null>(null);
  const [detail, setDetail] = useState<CurveQueryResult | null>(null);
  const [detailReq, setDetailReq] = useState<BarZoomWindow | null>(null);
  const [visibleWindow, setVisibleWindow] = useState<BarZoomWindow | null>(null);
  const [curveLoading, setCurveLoading] = useState(false);
  const [curveError, setCurveError] = useState("");

  useEffect(() => {
    if (!factor) return;
    let cancelled = false;
    zoomRef.current = null;
    setVisibleWindow(null);
    setDetailReq(null);
    setDetail(null);
    setOverview(null);
    setCurveLoading(true);
    setCurveError("");
    api
      .queryCurve(runId, {
        factor,
        period,
        start_idx: 0,
        max_points: 2000,
      })
      .then((res) => {
        if (!cancelled) setOverview(res);
      })
      .catch((err) => {
        if (!cancelled) {
          setCurveError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setCurveLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, factor, period]);

  useEffect(() => {
    if (!factor || !overview?.downsampled || !detailReq) return;
    let cancelled = false;
    setCurveLoading(true);
    api
      .queryCurve(runId, {
        factor,
        period,
        start_idx: detailReq.start,
        end_idx: detailReq.end,
        max_points: 2000,
      })
      .then((res) => {
        if (!cancelled) setDetail(res);
      })
      .catch((err) => {
        if (!cancelled) {
          setCurveError(err instanceof Error ? err.message : String(err));
        }
      })
      .finally(() => {
        if (!cancelled) setCurveLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId, factor, period, overview, detailReq]);

  const merged = useMemo(
    () => (overview ? mergeCurveData(overview, detail) : null),
    [overview, detail],
  );

  useEffect(() => {
    const el1 = navChartEl.current;
    const el2 = icChartEl.current;
    if (!el1 || !el2) return;
    const c1 = echarts.init(el1);
    const c2 = echarts.init(el2);
    c1.group = groupId;
    c2.group = groupId;
    echarts.connect(groupId);
    navChartRef.current = c1;
    icChartRef.current = c2;
    const obs = new ResizeObserver(() => {
      c1.resize();
      c2.resize();
    });
    obs.observe(el1);
    obs.observe(el2);
    return () => {
      obs.disconnect();
      echarts.disconnect(groupId);
      c1.dispose();
      c2.dispose();
      navChartRef.current = null;
      icChartRef.current = null;
    };
  }, [groupId]);

  useEffect(() => {
    const c1 = navChartRef.current;
    const c2 = icChartRef.current;
    if (!c1 || !c2 || !merged) return;

    const textColor = darkMode ? "#cbd5e1" : "#334155";
    const gridLine = darkMode ? "rgba(148,163,184,0.16)" : "rgba(148,163,184,0.28)";
    const indices = merged.indices;
    const dates = merged.dates;
    const maxIdx = Math.max(0, merged.totalPoints - 1);
    const axisMax = Math.max(1, maxIdx);

    const toPairs = (raw: (number | null)[] | undefined): [number, number | null][] =>
      (raw || []).map((v, i) => [indices[i] ?? i, v ?? null]);

    const formatXLabel = (val: number) => {
      if (val < -0.5 || val > maxIdx + 0.5) return "";
      return lookupDateByBarIdx(indices, dates, Math.round(val));
    };
    const formatXPointer = (p: { value: unknown }) =>
      lookupDateByBarIdx(indices, dates, Math.round(Number(p.value)));

    const formatTooltip = (rawParams: unknown) => {
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
      const rows = items
        .map((it) => {
          const y = Array.isArray(it.value) ? it.value[1] : it.value;
          const valText =
            typeof y === "number" && Number.isFinite(y) ? y.toFixed(4) : "-";
          const name = escapeHtml(it.seriesName ?? "");
          const marker = it.marker ?? "";
          return `<div style="display:flex;justify-content:space-between;gap:14px;"><span>${marker}${name}</span><span style="font-weight:600;font-variant-numeric:tabular-nums;">${valText}</span></div>`;
        })
        .join("");
      return `<div style="font-size:12px;line-height:1.5;"><div style="margin-bottom:4px;font-weight:600;">${escapeHtml(dateStr)}</div>${rows}</div>`;
    };

    const dzStart = zoomRef.current
      ? Math.max(0, Math.min(maxIdx, zoomRef.current.start))
      : 0;
    const dzEnd = zoomRef.current
      ? Math.max(dzStart, Math.min(maxIdx, zoomRef.current.end))
      : maxIdx;
    const minSpan = Math.min(5, maxIdx);

    const makeDataZoom = (): echarts.DataZoomComponentOption[] => [
      {
        type: "inside",
        xAxisIndex: 0,
        filterMode: "filter",
        startValue: dzStart,
        endValue: dzEnd,
        minValueSpan: minSpan,
      },
      {
        type: "slider",
        xAxisIndex: 0,
        filterMode: "filter",
        height: 16,
        bottom: 4,
        startValue: dzStart,
        endValue: dzEnd,
        minValueSpan: minSpan,
        labelFormatter: (val: number) =>
          lookupDateByBarIdx(indices, dates, Math.round(val)),
      },
    ];

    const qKeys = Object.keys(merged.series)
      .filter((k) => /^Q\d+_nav$/.test(k))
      .sort((a, b) => Number(a.slice(1, -4)) - Number(b.slice(1, -4)));
    const palette = [
      "#94a3b8",
      "#64748b",
      "#38bdf8",
      "#818cf8",
      "#2563eb",
      "#7c3aed",
      "#db2777",
      "#ea580c",
      "#16a34a",
      "#0d9488",
    ];

    const navSeries: echarts.SeriesOption[] = qKeys.map((k, idx) => ({
      name: k.replace("_nav", ""),
      type: "line",
      data: toPairs(merged.series[k]),
      showSymbol: false,
      lineStyle: {
        width: idx === qKeys.length - 1 ? 2.2 : 1.3,
        color: palette[idx % palette.length],
      },
      itemStyle: { color: palette[idx % palette.length] },
    }));
    if (merged.series.long_excess_nav) {
      navSeries.push({
        name: "Top超额净值",
        type: "line",
        data: toPairs(merged.series.long_excess_nav),
        showSymbol: false,
        lineStyle: { width: 2, type: "dashed", color: "#dc2626" },
        itemStyle: { color: "#dc2626" },
      });
    }
    if (merged.series.market_nav) {
      navSeries.push({
        name: "等权市场",
        type: "line",
        data: toPairs(merged.series.market_nav),
        showSymbol: false,
        lineStyle: { width: 1.2, type: "dotted", color: "#f59e0b" },
        itemStyle: { color: "#f59e0b" },
      });
    }

    c1.setOption(
      {
        animation: false,
        tooltip: {
          trigger: "axis",
          confine: true,
          axisPointer: { type: "cross" },
          formatter: formatTooltip,
        },
        legend: { top: 0, textStyle: { color: textColor, fontSize: 11 } },
        grid: { left: 50, right: 16, top: 26, bottom: 44 },
        xAxis: {
          type: "value",
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
        yAxis: {
          type: "value",
          scale: true,
          axisLabel: { color: textColor, fontSize: 10 },
          splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
        },
        dataZoom: makeDataZoom(),
        series: navSeries,
      },
      true,
    );

    c2.setOption(
      {
        animation: false,
        tooltip: {
          trigger: "axis",
          confine: true,
          axisPointer: { type: "cross" },
          formatter: formatTooltip,
        },
        legend: { top: 0, textStyle: { color: textColor, fontSize: 11 } },
        grid: { left: 50, right: 52, top: 26, bottom: 44 },
        xAxis: {
          type: "value",
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
        yAxis: [
          {
            type: "value",
            name: "日RankIC",
            nameTextStyle: { color: textColor, fontSize: 10 },
            axisLabel: { color: textColor, fontSize: 10 },
            splitLine: { lineStyle: { color: gridLine, type: "dashed" } },
          },
          {
            type: "value",
            name: "累计RankIC",
            scale: true,
            nameTextStyle: { color: textColor, fontSize: 10 },
            axisLabel: { color: textColor, fontSize: 10 },
            splitLine: { show: false },
          },
        ],
        dataZoom: makeDataZoom(),
        series: [
          {
            name: "日 RankIC",
            type: "bar",
            yAxisIndex: 0,
            barMaxWidth: 8,
            data: toPairs(merged.series.rank_ic),
            itemStyle: {
              color: (p: { value: unknown }) => {
                const y = Array.isArray(p.value) ? p.value[1] : p.value;
                return typeof y === "number" && y >= 0 ? "#ef4444" : "#10b981";
              },
            },
          },
          {
            name: "累计 RankIC",
            type: "line",
            yAxisIndex: 1,
            data: toPairs(merged.series.cum_rank_ic),
            showSymbol: false,
            lineStyle: { width: 2, color: "#2563eb" },
            itemStyle: { color: "#2563eb" },
          },
        ],
      },
      true,
    );

    let zoomTimer: ReturnType<typeof setTimeout> | null = null;
    const handleZoom = (sourceChart: echarts.ECharts) => {
      const win = readDataZoomWindow(sourceChart, merged.totalPoints);
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

    const onZoomNav = () => handleZoom(c1);
    const onZoomIc = () => handleZoom(c2);
    c1.off("datazoom");
    c2.off("datazoom");
    c1.on("datazoom", onZoomNav);
    c2.on("datazoom", onZoomIc);

    return () => {
      if (zoomTimer) clearTimeout(zoomTimer);
      c1.off("datazoom", onZoomNav);
      c2.off("datazoom", onZoomIc);
    };
  }, [merged, overview, darkMode]);

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
    navChartRef.current?.dispatchAction({
      type: "dataZoom",
      startValue: 0,
      endValue: maxIdx,
    });
    icChartRef.current?.dispatchAction({
      type: "dataZoom",
      startValue: 0,
      endValue: maxIdx,
    });
  };

  return (
    <div className="grid h-full min-h-0 flex-1 grid-rows-[minmax(130px,54fr)_minmax(120px,46fr)] gap-2 overflow-hidden">
      <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-2.5">
        <div className="mb-1 flex shrink-0 items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold">
              {factor} 分层净值曲线 ({period}d)
            </span>
            {merged && (
              <span className="text-[11px] text-muted-foreground">
                {visibleStartDate} ~ {visibleEndDate} · 显示{" "}
                {visibleCount.toLocaleString()} / {merged.totalPoints.toLocaleString()} 点
                {merged.downsampled ? " (Min-Max 降采样)" : ""}
              </span>
            )}
            {curveLoading && <span className="text-[11px] text-primary">加载中...</span>}
          </div>
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
        {curveError && (
          <div role="alert" className="py-8 text-center text-xs text-destructive">
            {curveError}
          </div>
        )}
        <div ref={navChartEl} className={curveError ? "hidden" : "min-h-0 w-full flex-1"} />
      </div>

      <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-2.5">
        <div className="mb-1 flex shrink-0 items-center justify-between gap-2">
          <span className="text-xs font-semibold">
            {factor} 日 RankIC 与累计 RankIC ({period}d)
          </span>
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
        <div ref={icChartEl} className={curveError ? "hidden" : "min-h-0 w-full flex-1"} />
      </div>
    </div>
  );
}

interface FactorDecayChartProps {
  autocorrelation: Record<string, number> | undefined;
  darkMode: boolean;
}

function FactorDecayChart({ autocorrelation, darkMode }: FactorDecayChartProps) {
  const decayChartEl = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = decayChartEl.current;
    if (!el || !autocorrelation) return;
    const chart = echarts.init(el);
    const obs = new ResizeObserver(() => chart.resize());
    obs.observe(el);

    const textColor = darkMode ? "#cbd5e1" : "#334155";
    const lags = Object.keys(autocorrelation).sort((a, b) => Number(a) - Number(b));
    chart.setOption({
      animation: false,
      tooltip: { trigger: "axis" },
      grid: { left: 42, right: 16, top: 18, bottom: 24 },
      xAxis: {
        type: "category",
        data: lags.map((l) => `Lag ${l}d`),
        axisLabel: { color: textColor, fontSize: 11 },
      },
      yAxis: {
        type: "value",
        min: 0,
        max: 1,
        axisLabel: { color: textColor, fontSize: 10 },
      },
      series: [
        {
          name: "秩自相关",
          type: "bar",
          barWidth: "40%",
          data: lags.map((l) => autocorrelation[l]),
          itemStyle: { color: "#3b82f6" },
        },
      ],
    });

    return () => {
      obs.disconnect();
      chart.dispose();
    };
  }, [autocorrelation, darkMode]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-2.5">
      <div className="mb-1 shrink-0 text-xs font-semibold">信号秩自相关衰减</div>
      <div ref={decayChartEl} className="min-h-0 w-full flex-1" />
    </div>
  );
}

interface FactorCorrHeatmapProps {
  corrMatrix: Record<string, Record<string, number>>;
  darkMode: boolean;
}

function FactorCorrHeatmap({ corrMatrix, darkMode }: FactorCorrHeatmapProps) {
  const corrChartEl = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const el = corrChartEl.current;
    const names = Object.keys(corrMatrix);
    if (!el || names.length < 2) return;
    const chart = echarts.init(el);
    const obs = new ResizeObserver(() => chart.resize());
    obs.observe(el);

    const textColor = darkMode ? "#cbd5e1" : "#334155";
    const points: [number, number, number][] = [];
    names.forEach((ni, i) => {
      names.forEach((nj, j) => {
        const val = corrMatrix[ni]?.[nj];
        points.push([j, i, typeof val === "number" ? Number(val.toFixed(3)) : 0]);
      });
    });

    chart.setOption({
      animation: false,
      tooltip: { position: "top" },
      grid: { left: 96, right: 32, top: 12, bottom: 56 },
      xAxis: {
        type: "category",
        data: names,
        axisLabel: { color: textColor, fontSize: 10, rotate: 30 },
      },
      yAxis: {
        type: "category",
        data: names,
        axisLabel: { color: textColor, fontSize: 10 },
      },
      visualMap: {
        min: -1,
        max: 1,
        calculable: true,
        orient: "horizontal",
        left: "center",
        bottom: 4,
        inRange: { color: ["#2563eb", "#f8fafc", "#dc2626"] },
        textStyle: { color: textColor, fontSize: 10 },
      },
      series: [
        {
          type: "heatmap",
          data: points,
          label: { show: names.length <= 8, fontSize: 10 },
        },
      ],
    });

    return () => {
      obs.disconnect();
      chart.dispose();
    };
  }, [corrMatrix, darkMode]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card p-2.5">
      <div className="mb-1 shrink-0 text-xs font-semibold">因子截面秩相关性热力图</div>
      <div ref={corrChartEl} className="min-h-0 w-full flex-1" />
    </div>
  );
}

export function FactorResultView({
  run,
  manifest,
  darkMode,
  onRefreshRun,
}: FactorResultViewProps) {
  const s = manifest.summary || {};
  const factorList = (s.factors as FactorMetaEntry[] | undefined) || [];
  const periods = (s.periods as number[] | undefined) || [5];
  const defaultPeriod = Number(s.primary_period ?? periods[0] ?? 5);
  const isMulti = Boolean(s.is_multi_factor);
  const corrMatrix = (s.correlation_matrix as Record<string, Record<string, number>>) || {};

  const [selectedFactor, setSelectedFactor] = useState<string>(factorList[0]?.name || "");
  const [selectedPeriod, setSelectedPeriod] = useState<number>(defaultPeriod);
  const [warningsOpen, setWarningsOpen] = useState(false);

  useEffect(() => {
    setWarningsOpen(false);
  }, [run.id, selectedFactor]);

  useEffect(() => {
    if (factorList.length > 0 && !factorList.some((f) => f.name === selectedFactor)) {
      setSelectedFactor(factorList[0].name);
    }
  }, [factorList, selectedFactor]);

  const activeFactor = useMemo(
    () => factorList.find((f) => f.name === selectedFactor) || factorList[0],
    [factorList, selectedFactor],
  );

  const fSum = (activeFactor?.summary || {}) as Record<string, unknown>;
  const warnings = activeFactor?.diagnostics?.warnings || [];
  const causalityStatus = activeFactor?.diagnostics?.causality_check?.status || "UNKNOWN";
  const hasCorrMatrix = Object.keys(corrMatrix).length > 1;

  return (
    <div className="flex h-full min-h-0 flex-1 flex-col gap-2 overflow-hidden">
      <div className="flex shrink-0 flex-wrap items-center justify-between gap-2 rounded-lg border bg-card px-2.5 py-1.5">
        <div className="flex flex-wrap items-center gap-3">
          {factorList.length > 1 && (
            <label className="flex items-center gap-1.5 text-xs">
              <span className="text-muted-foreground">因子:</span>
              <SelectNative
                value={selectedFactor}
                onChange={(e) => setSelectedFactor(e.target.value)}
                className="h-7 w-44"
              >
                {factorList.map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.name}
                  </option>
                ))}
              </SelectNative>
            </label>
          )}
          <label className="flex items-center gap-1.5 text-xs">
            <span className="text-muted-foreground">评估周期:</span>
            <SelectNative
              value={String(selectedPeriod)}
              onChange={(e) => setSelectedPeriod(Number(e.target.value))}
              className="h-7 w-24"
            >
              {periods.map((p) => (
                <option key={p} value={p}>
                  {p}d {p === defaultPeriod ? "(主)" : ""}
                </option>
              ))}
            </SelectNative>
          </label>
          <Badge variant={causalityStatus === "PASS" ? "select" : "destructive"}>
            因果探针: {causalityStatus}
          </Badge>
          {activeFactor?.expression && (
            <code className="max-w-md truncate rounded bg-muted px-2 py-0.5 text-[11px]">
              {activeFactor.expression}
            </code>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <a
            href={api.exportUrl(run.id, "xlsx")}
            className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs hover:bg-accent"
          >
            <Download className="h-3.5 w-3.5" aria-hidden="true" />
            导出 XLSX
          </a>
          <a
            href={api.exportUrl(run.id, "parquet")}
            className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-xs hover:bg-accent"
          >
            Parquet ZIP
          </a>
        </div>
      </div>

      {warnings.length > 0 && (
        <div className="relative shrink-0">
          <div className="flex items-center justify-between gap-2 rounded-md border border-amber-500/40 bg-amber-500/10 px-2.5 py-1 text-[11px] text-amber-800 dark:text-amber-200">
            <div className="flex min-w-0 flex-1 items-center gap-1.5">
              <AlertTriangle className="h-3.5 w-3.5 shrink-0" aria-hidden="true" />
              <span className="shrink-0 font-semibold">告警 ({warnings.length}):</span>
              <span className="truncate" title={warnings.join("；")}>
                {warnings[0]}
              </span>
            </div>
            {(warnings.length > 1 || warnings[0].length > 60) && (
              <Button
                variant="ghost"
                size="sm"
                className="h-5 shrink-0 px-1.5 text-[11px] text-amber-800 hover:bg-amber-500/20 dark:text-amber-200"
                onClick={() => setWarningsOpen((v) => !v)}
                aria-expanded={warningsOpen}
              >
                {warningsOpen ? "收起" : "查看全部"}
              </Button>
            )}
          </div>
          {warningsOpen && (
            <div className="absolute inset-x-0 top-full z-30 mt-1 max-h-52 overflow-auto overscroll-contain rounded-lg border border-amber-500/40 bg-popover p-2.5 text-xs text-amber-800 shadow-lg dark:text-amber-200">
              <ul className="list-disc space-y-1 pl-4">
                {warnings.map((w, i) => (
                  <li key={i}>{w}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      <div className="grid shrink-0 grid-cols-3 gap-2 lg:grid-cols-6">
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">RankIC Mean / IR</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtNum(fSum.rank_ic_mean, 4)} / {fmtNum(fSum.rank_icir, 3)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">NW t-stat</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtNum(fSum.nw_t_stat, 2)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">Top 超额年化</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtPct(fSum.top_ann_excess)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">多空年化 / Sharpe</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtPct(fSum.ls_ann_return)} / {fmtNum(fSum.ls_sharpe, 2)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">单调性 / 覆盖率</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtNum(activeFactor?.monotonicities?.[String(selectedPeriod)] ?? fSum.monotonicity, 3)} /{" "}
            {fmtPct(fSum.avg_coverage, false, 1)}
          </div>
        </div>
        <div className="rounded-lg border bg-card px-2.5 py-1.5">
          <div className="text-[11px] leading-tight text-muted-foreground">日换手 / 半衰期</div>
          <div className="mt-0.5 truncate text-xs font-semibold leading-tight">
            {fmtPct(fSum.top_daily_turnover, false, 1)} /{" "}
            {fSum.half_life_days !== null && fSum.half_life_days !== undefined
              ? `${fmtNum(fSum.half_life_days, 1)}d`
              : "N/A"}
          </div>
        </div>
      </div>

      <Tabs
        defaultValue={isMulti ? "leaderboard" : "curves"}
        className="flex min-h-0 flex-1 flex-col overflow-hidden"
      >
        <TabsList className="w-fit shrink-0">
          {isMulti && <TabsTrigger value="leaderboard">多因子排行与相关性</TabsTrigger>}
          <TabsTrigger value="curves">分层净值与 IC 时序</TabsTrigger>
          <TabsTrigger value="stats">分层收益与 IC 统计</TabsTrigger>
          <TabsTrigger value="breakdown">年度 / 样本外 / 行业</TabsTrigger>
          <TabsTrigger value="logs">运行日志</TabsTrigger>
        </TabsList>

        {isMulti && (
          <TabsContent
            value="leaderboard"
            className={
              hasCorrMatrix
                ? "mt-2 grid min-h-0 flex-1 grid-rows-[minmax(120px,46fr)_minmax(140px,54fr)] gap-2 overflow-hidden data-[state=inactive]:hidden"
                : "mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
            }
          >
            <ServerDataTable
              runId={run.id}
              tableName="leaderboard"
              title="多因子评价排行榜"
              columns={LEADERBOARD_COLS}
              defaultSortBy="rank_icir"
              defaultSortDir="desc"
              rowKeyField="factor"
              selectedRowValue={selectedFactor}
              onRowClick={(row) => {
                if (row.factor) setSelectedFactor(String(row.factor));
              }}
              fillHeight
            />
            {hasCorrMatrix && (
              <FactorCorrHeatmap corrMatrix={corrMatrix} darkMode={darkMode} />
            )}
          </TabsContent>
        )}

        <TabsContent
          value="curves"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <FactorCurvesPanel
            runId={run.id}
            factor={selectedFactor}
            period={selectedPeriod}
            darkMode={darkMode}
          />
        </TabsContent>

        <TabsContent
          value="stats"
          className="mt-2 grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(110px,36fr)_minmax(100px,34fr)_minmax(90px,30fr)] gap-2 overflow-hidden data-[state=inactive]:hidden lg:grid-cols-12 lg:grid-rows-[minmax(130px,52fr)_minmax(120px,48fr)]"
        >
          <div className="min-h-0 overflow-hidden lg:col-span-12">
            <ServerDataTable
              runId={run.id}
              tableName="quantiles"
              title={`${selectedFactor} 分层组合绩效 (${selectedPeriod}d)`}
              columns={QUANTILE_COLS}
              factor={selectedFactor}
              period={selectedPeriod}
              fillHeight
            />
          </div>
          <div className="min-h-0 overflow-hidden lg:col-span-8">
            <ServerDataTable
              runId={run.id}
              tableName="ic_summary"
              title={`${selectedFactor} 各周期 IC / RankIC 汇总`}
              columns={IC_COLS}
              factor={selectedFactor}
              defaultSortBy="period"
              defaultSortDir="asc"
              fillHeight
            />
          </div>
          <div className="min-h-0 overflow-hidden lg:col-span-4">
            <FactorDecayChart
              autocorrelation={activeFactor?.turnover?.autocorrelation}
              darkMode={darkMode}
            />
          </div>
        </TabsContent>

        <TabsContent
          value="breakdown"
          className="mt-2 flex min-h-0 flex-1 flex-col gap-2 overflow-hidden data-[state=inactive]:hidden"
        >
          {activeFactor?.holdout && (
            <div className="shrink-0 space-y-1.5 rounded-lg border bg-card px-3 py-2 text-xs">
              <div className="font-semibold">
                Holdout 样本外检验 (Split: {String(activeFactor.holdout.split_date)} | Retention:{" "}
                {fmtNum(activeFactor.holdout.rank_ic_retention, 2)}x)
              </div>
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                {(["in_sample", "out_of_sample"] as const).map((segKey) => {
                  const seg = activeFactor.holdout?.[segKey] as Record<string, unknown> | undefined;
                  if (!seg) return null;
                  return (
                    <div key={segKey} className="space-y-0.5 rounded border p-2">
                      <div className="font-medium">
                        {segKey === "in_sample" ? "样本内 (IS)" : "样本外 (OOS)"}: {String(seg.start)} ~{" "}
                        {String(seg.end)} ({String(seg.bars)} bars)
                      </div>
                      <div>
                        RankIC: {fmtNum(seg.rank_ic_mean, 4)} | RankICIR: {fmtNum(seg.rank_icir, 3)} |
                        单调性: {fmtNum(seg.monotonicity, 3)}
                      </div>
                      <div>
                        Top超额年化: {fmtPct(seg.top_ann_excess)} | 多空年化: {fmtPct(seg.ls_ann_return)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-2 gap-2 overflow-hidden lg:grid-cols-2 lg:grid-rows-1">
            <ServerDataTable
              runId={run.id}
              tableName="annual"
              title={`${selectedFactor} 分年度稳定性`}
              columns={ANNUAL_COLS}
              factor={selectedFactor}
              defaultSortBy="year"
              defaultSortDir="asc"
              fillHeight
            />

            <ServerDataTable
              runId={run.id}
              tableName="industry"
              title={`${selectedFactor} 分行业 RankIC`}
              columns={INDUSTRY_COLS}
              factor={selectedFactor}
              defaultSortBy="rank_ic_mean"
              defaultSortDir="desc"
              fillHeight
            />
          </div>
        </TabsContent>

        <TabsContent
          value="logs"
          className="mt-2 flex min-h-0 flex-1 flex-col overflow-hidden data-[state=inactive]:hidden"
        >
          <RunLogsView runId={run.id} status={run.status} onFinished={onRefreshRun} fillHeight />
        </TabsContent>
      </Tabs>
    </div>
  );
}
