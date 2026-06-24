export interface StockInfo {
  ticker: string
  short_name: string
  sector: string | null
  industry: string | null
  market_cap: number | null
  currency: string
  fifty_two_week_low: number | null
  fifty_two_week_high: number | null
  previous_close: number | null
  logo_url: string | null
  last_close: number
  change: number
  change_pct: number
  last_date: string
  volume: number | null
}

export interface OhlcvBar {
  Date: string
  Open: number | null
  High: number | null
  Low: number | null
  Close: number | null
  ['Adj Close']?: number | null
  Volume?: number | null
}

export interface IndicatorBar {
  Date: string
  Close: number | null
  SMA_10?: number | null
  SMA_20?: number | null
  SMA_50?: number | null
  SMA_200?: number | null
  EMA_10?: number | null
  EMA_20?: number | null
  EMA_50?: number | null
  RSI_14?: number | null
  MACD?: number | null
  MACD_signal?: number | null
  MACD_hist?: number | null
  BB_upper?: number | null
  BB_mid?: number | null
  BB_lower?: number | null
}

export interface ModelMetric {
  rmse: number
  mae: number
  mape: number
  r2: number
  dir_acc?: number
}

export interface ModelComparisonResponse {
  best_model: string
  metrics: Record<string, ModelMetric>
}

export interface TrainRequest {
  ticker?: string
  years?: number
  target?: 'log_return' | 'price'
  models?: string[] | null
  epochs?: number
  lookback?: number
  tune?: boolean
}

export interface TrainJobResponse {
  job_id: string
}

export interface TrainStatusResponse {
  job_id: string
  status: 'pending' | 'running' | 'completed' | 'failed'
  message: string
  error: string | null
}

export interface ForecastPoint {
  date: string
  pred_close: number
  lower: number
  upper: number
}

export interface ForecastHorizonSummary extends ForecastPoint {
  pct_change: number
}

export interface ForecastResponse {
  model_name: string
  last_close: number
  path: ForecastPoint[]
  summary: Record<string, ForecastHorizonSummary>
}

export interface InsightsResponse {
  why_selected: string
  trend_summary: string
  confidence: string
  confidence_note?: string
}

export interface EdaResponse {
  rows: number
  start: string
  end: string
  price_min: number
  price_max: number
  price_last: number
  total_return_pct: number
  cagr_pct: number
  daily_vol_pct: number
  annual_vol_pct: number
  ann_sharpe_naive: number
  skew: number
  kurtosis: number
  max_daily_gain_pct: number
  max_daily_loss_pct: number
}
