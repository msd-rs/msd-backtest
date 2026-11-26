import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, ArrowUpDown, Download, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { api } from "@/lib/api";
import { fmtBps, fmtInt, fmtMoney, fmtNum, fmtPct } from "@/lib/format";
import { cn } from "@/lib/utils";
import type { TableQueryResult } from "@/types/api";

export type CellFormat =
  | "pct"
  | "pct_plain"
  | "money"
  | "money_signed"
  | "price"
  | "f4"
  | "f2"
  | "int"
  | "bps"
  | "text";

export interface ColumnSpec {
  key: string;
  label?: string;
  fmt?: CellFormat;
  align?: "left" | "right" | "center";
  colorSign?: boolean;
}

interface ServerDataTableProps {
  runId: string;
  tableName: string;
  title: string;
  columns?: ColumnSpec[];
  factor?: string;
  period?: number;
  defaultSortBy?: string;
  defaultSortDir?: "asc" | "desc";
  pageSize?: number;
  selectedRowValue?: string;
  rowKeyField?: string;
  onRowClick?: (row: Record<string, unknown>) => void;
  showExport?: boolean;
  fillHeight?: boolean;
}

function formatCellValue(val: unknown, fmt?: CellFormat): string {
  if (val === null || val === undefined) return "-";
  switch (fmt) {
    case "pct":
      return fmtPct(val, true, 2);
    case "pct_plain":
      return fmtPct(val, false, 2);
    case "money":
      return fmtMoney(val, false);
    case "money_signed":
      return fmtMoney(val, true);
    case "price":
      return fmtNum(val, 3);
    case "f4":
      return fmtNum(val, 4);
    case "f2":
      return fmtNum(val, 2);
    case "int":
      return fmtInt(val);
    case "bps":
      return fmtBps(val, 1);
    default:
      if (typeof val === "number") {
        if (Number.isInteger(val)) return String(val);
        return fmtNum(val, 4);
      }
      return String(val);
  }
}

export function ServerDataTable({
  runId,
  tableName,
  title,
  columns,
  factor,
  period,
  defaultSortBy,
  defaultSortDir = "desc",
  pageSize = 50,
  selectedRowValue,
  rowKeyField,
  onRowClick,
  showExport = true,
  fillHeight = false,
}: ServerDataTableProps) {
  const [data, setData] = useState<TableQueryResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState<string | undefined>(defaultSortBy);
  const [sortDir, setSortDir] = useState<"asc" | "desc">(defaultSortDir);
  const [searchInput, setSearchInput] = useState("");
  const [debouncedQ, setDebouncedQ] = useState("");

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedQ(searchInput);
      setPage(1);
    }, 220);
    return () => clearTimeout(timer);
  }, [searchInput]);

  useEffect(() => {
    setPage(1);
  }, [runId, tableName, factor, period]);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError("");
    api
      .queryTable(runId, tableName, {
        page,
        page_size: pageSize,
        sort_by: sortBy,
        sort_dir: sortDir,
        q: debouncedQ || undefined,
        factor,
        period,
      })
      .then((res) => {
        if (!cancelled) setData(res);
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
  }, [runId, tableName, page, pageSize, sortBy, sortDir, debouncedQ, factor, period]);

  const DEFAULT_COLUMN_LABELS: Record<string, string> = {
    obj: "标的",
    name: "名称",
  };

  const effectiveCols: ColumnSpec[] = (() => {
    if (columns && columns.length > 0) return columns;
    if (!data) return [];
    return data.columns.map((c) => ({
      key: c.name,
      label: DEFAULT_COLUMN_LABELS[c.name] ?? c.name,
      align: c.dtype.includes("Float") || c.dtype.includes("Int") ? "right" : "left",
    }));
  })();

  function handleSort(colKey: string) {
    if (sortBy === colKey) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortBy(colKey);
      setSortDir("desc");
    }
    setPage(1);
  }

  return (
    <div
      className={
        fillHeight
          ? "flex h-full min-h-0 flex-col overflow-hidden rounded-lg border bg-card"
          : "flex flex-col rounded-lg border bg-card"
      }
    >
      <div
        className={
          fillHeight
            ? "flex shrink-0 flex-wrap items-center justify-between gap-2 border-b px-3 py-1.5"
            : "flex flex-wrap items-center justify-between gap-2 border-b px-3 py-2"
        }
      >
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold">{title}</span>
          <span className="text-[11px] text-muted-foreground">
            {data ? `${data.total_rows.toLocaleString()} 条` : ""}
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className={fillHeight ? "relative w-32" : "relative w-44"}>
            <Search
              className="pointer-events-none absolute left-2 top-2 h-3.5 w-3.5 text-muted-foreground"
              aria-hidden="true"
            />
            <Input
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              placeholder="搜索过滤..."
              className="h-7 pl-7 text-xs"
              aria-label={`过滤 ${title}`}
            />
          </div>

          {showExport && (
            <div className="flex items-center gap-1">
              <a
                href={api.exportUrl(runId, "csv", tableName)}
                className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] hover:bg-accent"
              >
                <Download className="h-3 w-3" aria-hidden="true" />
                CSV
              </a>
              <a
                href={api.exportUrl(runId, "xlsx", tableName)}
                className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] hover:bg-accent"
              >
                XLSX
              </a>
              <a
                href={api.exportUrl(runId, "parquet", tableName)}
                className="inline-flex h-7 items-center gap-1 rounded-md border px-2 text-[11px] hover:bg-accent"
              >
                Parquet
              </a>
            </div>
          )}
        </div>
      </div>

      {error && (
        <div role="alert" className="px-3 py-2 text-xs text-destructive">
          {error}
        </div>
      )}

      <div
        tabIndex={0}
        role="region"
        aria-label={title}
        className={
          fillHeight
            ? "min-h-0 flex-1 overflow-auto overscroll-contain [scrollbar-gutter:stable] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            : "max-h-105 overflow-auto focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
        }
      >
        <table className="w-full border-collapse text-xs">
          <caption className="sr-only">{title}</caption>
          <thead className="sticky top-0 z-10 bg-muted/95 backdrop-blur-xs">
            <tr className="border-b">
              {effectiveCols.map((col) => {
                const isSorted = sortBy === col.key;
                const alignClass =
                  col.align === "right"
                    ? "justify-end text-right"
                    : col.align === "center"
                      ? "justify-center text-center"
                      : "justify-start text-left";
                return (
                  <th
                    key={col.key}
                    scope="col"
                    aria-sort={
                      isSorted ? (sortDir === "asc" ? "ascending" : "descending") : undefined
                    }
                    className="whitespace-nowrap border-b font-medium text-muted-foreground"
                  >
                    <button
                      type="button"
                      onClick={() => handleSort(col.key)}
                      className={cn(
                        "flex w-full items-center gap-1 hover:bg-accent/60 hover:text-foreground cursor-pointer",
                        fillHeight ? "min-h-7 px-2.5 py-1" : "min-h-7.5 px-2.5 py-1.5",
                        alignClass,
                      )}
                    >
                      <span>{col.label || col.key}</span>
                      {isSorted ? (
                        sortDir === "asc" ? (
                          <ArrowUp className="h-3 w-3 shrink-0" aria-hidden="true" />
                        ) : (
                          <ArrowDown className="h-3 w-3 shrink-0" aria-hidden="true" />
                        )
                      ) : (
                        <ArrowUpDown className="h-3 w-3 shrink-0 opacity-40" aria-hidden="true" />
                      )}
                    </button>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            {loading && !data ? (
              <tr>
                <td colSpan={Math.max(1, effectiveCols.length)} className="py-8 text-center text-muted-foreground">
                  加载中...
                </td>
              </tr>
            ) : !data || data.rows.length === 0 ? (
              <tr>
                <td colSpan={Math.max(1, effectiveCols.length)} className="py-8 text-center text-muted-foreground">
                  暂无数据
                </td>
              </tr>
            ) : (
              data.rows.map((row, rIdx) => {
                const rowKey = rowKeyField ? String(row[rowKeyField] ?? rIdx) : String(rIdx);
                const isSelected =
                  selectedRowValue !== undefined &&
                  rowKeyField !== undefined &&
                  String(row[rowKeyField]) === selectedRowValue;
                return (
                  <tr
                    key={`${rowKey}-${rIdx}`}
                    onClick={onRowClick ? () => onRowClick(row) : undefined}
                    className={cn(
                      "border-b transition-colors last:border-b-0",
                      !isSelected && "even:bg-muted/30",
                      onRowClick ? "cursor-pointer hover:bg-accent/60" : "hover:bg-muted/55",
                      isSelected && "bg-primary/12",
                    )}
                  >
                    {effectiveCols.map((col) => {
                      const raw = row[col.key];
                      const align =
                        col.align === "right"
                          ? "text-right font-mono tabular-nums"
                          : col.align === "center"
                            ? "text-center"
                            : "text-left";
                      let colorCls = "";
                      if (col.colorSign && typeof raw === "number" && Number.isFinite(raw)) {
                        if (raw > 0) colorCls = "text-red-600 dark:text-red-400";
                        else if (raw < 0) colorCls = "text-emerald-600 dark:text-emerald-400";
                      } else if (col.key === "action" && typeof raw === "string") {
                        if (raw === "BUY") colorCls = "text-red-600 dark:text-red-400 font-semibold";
                        else if (raw === "SELL") colorCls = "text-emerald-600 dark:text-emerald-400 font-semibold";
                      }
                      return (
                        <td
                          key={col.key}
                          className={cn(
                            "whitespace-nowrap px-2.5",
                            fillHeight ? "py-1" : "py-1.5",
                            align,
                            colorCls,
                          )}
                        >
                          {formatCellValue(raw, col.fmt)}
                        </td>
                      );
                    })}
                  </tr>
                );
              })
            )}
          </tbody>
        </table>
      </div>

      {data && data.total_pages > 1 && (
        <div
          className={cn(
            "flex shrink-0 items-center justify-between border-t px-3 text-[11px] text-muted-foreground",
            fillHeight ? "py-1" : "py-1.5",
          )}
        >
          <span>
            第 {data.page} / {data.total_pages} 页（共 {data.total_rows.toLocaleString()} 条）
          </span>
          <div className="flex items-center gap-1.5">
            <Button
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[11px]"
              disabled={page <= 1}
              onClick={() => setPage((p) => Math.max(1, p - 1))}
            >
              上一页
            </Button>
            <Button
              variant="outline"
              size="sm"
              className="h-6 px-2 text-[11px]"
              disabled={page >= data.total_pages}
              onClick={() => setPage((p) => Math.min(data.total_pages, p + 1))}
            >
              下一页
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
