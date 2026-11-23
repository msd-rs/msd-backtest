export type RunMode = "select" | "timing" | "portfolio" | "factor";

export const RUN_MODE_LABELS: Record<RunMode, string> = {
  select: "选股",
  timing: "择时",
  portfolio: "组合",
  factor: "因子",
};

export interface ParamSpec {
  name: string;
  default_str: string;
  type_label: string;
}

export interface StrategyMetadata {
  path: string;
  has_selector: boolean;
  has_strategy: boolean;
  has_portfolio_strategy: boolean;
  has_factor: boolean;
  default_mode: RunMode;
  available_modes: RunMode[];
  selector_name: string | null;
  strategy_name: string | null;
  factor_name: string | null;
  selector_params: ParamSpec[];
  strategy_params: ParamSpec[];
  factor_params: ParamSpec[];
  is_expression_file: boolean;
  expression_names: string[];
  docstring: string;
  author: string;
}

export interface SystemDefaults {
  msd_host: string;
  begin: string;
  end: string;
  select_date: string;
  benchmark: string;
  cash: number;
  fee: string;
  exec_mode: "close" | "next_open";
  warmup_bars: number;
  holdout: string;
  fin_fields: string;
  fin_only_year: boolean;
  fin_lookback_days: number | null;
  factor_periods: string;
  factor_primary_period: number | null;
  factor_quantiles: number;
  factor_entry: "next_open" | "open_close" | "close";
  factor_exclude_limit: boolean;
  factor_min_listed_days: number;
  factor_winsorize: "none" | "mad" | "sigma" | "percentile";
  factor_standardize: "none" | "zscore" | "rank";
  factor_neutralize: string;
  factor_cost_bps: number;
  factor_min_obs: number;
  factor_by_industry: boolean;
  factor_industry_category: string;
  factor_causality_check: boolean;
}

export interface ServerConfigData {
  config_file: string;
  data_dir: string;
  server: {
    host: string;
    port: number;
  };
  defaults: SystemDefaults;
}

export interface RunSummaryRecord {
  id: string;
  target_type: "strategy" | "shortcut";
  target_id: string;
  strategy_id: string;
  shortcut_id: string | null;
  mode: RunMode;
  status: "running" | "completed" | "failed" | "cancelled";
  pid: number | null;
  resolved_params: Record<string, unknown>;
  summary: Record<string, unknown> | null;
  error: string;
  started_at: string;
  finished_at: string | null;
  elapsed: number | null;
  is_latest: boolean;
}

export interface ShortcutRecord {
  id: string;
  strategy_id: string;
  name: string;
  mode: RunMode;
  params: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  latest_run: RunSummaryRecord | null;
  active_run: RunSummaryRecord | null;
}

export interface StrategyRecord {
  id: string;
  name: string;
  path: string;
  default_mode: RunMode;
  metadata: StrategyMetadata;
  notes: string;
  file_exists: boolean;
  created_at: string;
  updated_at: string;
  created?: boolean;
  shortcuts: ShortcutRecord[];
  latest_run: RunSummaryRecord | null;
  active_run: RunSummaryRecord | null;
}

export interface TableQueryResult {
  table: string;
  columns: { name: string; dtype: string }[];
  total_rows: number;
  page: number;
  page_size: number;
  total_pages: number;
  rows: Record<string, unknown>[];
}

export interface CurveQueryResult {
  total_points: number;
  start_idx: number;
  end_idx: number;
  downsampled: boolean;
  indices: number[];
  dates: string[];
  series: Record<string, (number | null)[]>;
  meta: Record<string, unknown>;
}

export interface RunManifest {
  version: number;
  mode: RunMode;
  strategy_name: string;
  elapsed: number;
  summary: Record<string, unknown>;
  tables: Record<string, { file: string; rows: number; columns: { name: string; dtype: string }[] }>;
  curves: Record<string, Record<string, unknown>>;
}

export interface DraftParams {
  mode: RunMode;
  objs: string;
  use_internal_selector: boolean;
  begin: string;
  end: string;
  select_date: string;
  benchmark: string;
  cash: string;
  fee: string;
  exec_mode: string;
  warmup_bars: string;
  holdout: string;
  msd_host: string;
  fin_fields: string;
  fin_only_year: boolean;
  fin_lookback_days: string;
  strategy_args: Record<string, string>;
  extra_args: string;
  factor_periods: string;
  factor_primary_period: string;
  factor_quantiles: string;
  factor_entry: string;
  factor_exclude_limit: boolean;
  factor_min_listed_days: string;
  factor_winsorize: string;
  factor_standardize: string;
  factor_neutralize: string;
  factor_cost_bps: string;
  factor_min_obs: string;
  factor_by_industry: boolean;
  factor_industry_category: string;
  factor_causality_check: boolean;
  factor_direction: string;
}
