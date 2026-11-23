export function fmtPct(v: unknown, signed = true, digits = 2): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  const pct = (v * 100).toFixed(digits);
  if (signed && v > 0) return `+${pct}%`;
  return `${pct}%`;
}

export function fmtNum(v: unknown, digits = 4): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  return v.toFixed(digits);
}

export function fmtMoney(v: unknown, signed = false): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  const formatted = Math.abs(v).toLocaleString("zh-CN", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  if (v < 0) return `-${formatted}`;
  if (signed && v > 0) return `+${formatted}`;
  return formatted;
}

export function fmtInt(v: unknown): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  return Math.round(v).toLocaleString("zh-CN");
}

export function fmtBps(v: unknown, digits = 1): string {
  if (typeof v !== "number" || !Number.isFinite(v)) return "-";
  return `${(v * 10000).toFixed(digits)}bp`;
}
