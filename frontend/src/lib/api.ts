import { 
  DatasetSummary, 
  DatasetColumn, 
  DatasetMetadata, 
  DatasetValidation, 
  HealthStatus 
} from "@/types/dataset";
import { 
  VariableStatistics, 
  FrequencyTableResponse 
} from "@/types/statistics";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchJson<T>(endpoint: string): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;
  try {
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) {
      let errMsg = `HTTP error ${res.status}: ${res.statusText}`;
      try {
        const body = await res.json();
        if (body && body.detail) {
          errMsg = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
        }
      } catch (_) {}
      throw new Error(errMsg);
    }
    return (await res.json()) as T;
  } catch (error: any) {
    console.error(`API Fetch Error [${endpoint}]:`, error);
    throw new Error(error.message || `Failed to fetch from ${endpoint}`);
  }
}

// Dataset APIs
export async function fetchHealth(): Promise<HealthStatus> {
  return fetchJson<HealthStatus>("/api/health");
}

export async function fetchDatasetSummary(): Promise<DatasetSummary> {
  return fetchJson<DatasetSummary>("/api/dataset/summary");
}

export async function fetchDatasetColumns(dataset: "train" | "test" = "train"): Promise<DatasetColumn[]> {
  return fetchJson<DatasetColumn[]>(`/api/dataset/columns?dataset=${dataset}`);
}

export async function fetchDatasetSample(
  dataset: "train" | "test" = "train",
  limit: number = 5,
  includeEngineeredFeatures: boolean = false
): Promise<Record<string, any>[]> {
  return fetchJson<Record<string, any>[]>(
    `/api/dataset/sample?dataset=${dataset}&limit=${limit}&include_engineered_features=${includeEngineeredFeatures}`
  );
}

export async function fetchDatasetMetadata(): Promise<DatasetMetadata> {
  return fetchJson<DatasetMetadata>("/api/dataset/metadata");
}

export async function fetchDatasetValidation(): Promise<DatasetValidation> {
  return fetchJson<DatasetValidation>("/api/dataset/validation");
}

// Statistics APIs
export async function fetchVariableStatistics(variable: string): Promise<VariableStatistics> {
  return fetchJson<VariableStatistics>(`/api/statistics/variable/${variable}`);
}

export async function fetchCategoricalFrequency(variable: string): Promise<FrequencyTableResponse> {
  return fetchJson<FrequencyTableResponse>(`/api/statistics/frequency/${variable}`);
}

export async function fetchAllStatisticsSummary(): Promise<Record<string, any>> {
  return fetchJson<Record<string, any>>("/api/statistics/summary");
}

// Visualization APIs (Phase 6)
import {
  HistogramResponse,
  BoxplotResponse,
  DistributionResponse,
  ScatterResponse,
  ScatterMatrixResponse,
  BubbleResponse,
  CategoricalResponse,
} from "@/types/visualization";

export async function fetchHistogram(variable: string, bins?: number): Promise<HistogramResponse> {
  const query = bins ? `?bins=${bins}` : "";
  return fetchJson<HistogramResponse>(`/api/visualization/histogram/${variable}${query}`);
}

export async function fetchBoxplot(variable: string): Promise<BoxplotResponse> {
  return fetchJson<BoxplotResponse>(`/api/visualization/boxplot/${variable}`);
}

export async function fetchDistribution(variable: string): Promise<DistributionResponse> {
  return fetchJson<DistributionResponse>(`/api/visualization/distribution/${variable}`);
}

export async function fetchScatter(x: string, y: string, sampleSize: number = 1000): Promise<ScatterResponse> {
  return fetchJson<ScatterResponse>(`/api/visualization/scatter?x=${x}&y=${y}&sample_size=${sampleSize}`);
}

export async function fetchScatterMatrix(sampleSize: number = 600): Promise<ScatterMatrixResponse> {
  return fetchJson<ScatterMatrixResponse>(`/api/visualization/scatter-matrix?sample_size=${sampleSize}`);
}

export async function fetchBubble(
  x: string = "temp",
  y: string = "count",
  size: string = "humidity",
  sampleSize: number = 600
): Promise<BubbleResponse> {
  return fetchJson<BubbleResponse>(
    `/api/visualization/bubble?x=${x}&y=${y}&size=${size}&sample_size=${sampleSize}`
  );
}

export async function fetchCategoricalVisualization(variable: string): Promise<CategoricalResponse> {
  return fetchJson<CategoricalResponse>(`/api/visualization/categorical/${variable}`);
}

// Correlation APIs (Phase 7)
import {
  CombinedMatrixResponse,
  PearsonMatrixResponse,
  SpearmanMatrixResponse,
  PairwiseCorrelationResponse,
} from "@/types/correlation";

export async function fetchCorrelationMatrix(): Promise<CombinedMatrixResponse> {
  return fetchJson<CombinedMatrixResponse>("/api/correlation/matrix");
}

export async function fetchPearsonCorrelation(): Promise<PearsonMatrixResponse> {
  return fetchJson<PearsonMatrixResponse>("/api/correlation/pearson");
}

export async function fetchSpearmanCorrelation(): Promise<SpearmanMatrixResponse> {
  return fetchJson<SpearmanMatrixResponse>("/api/correlation/spearman");
}

export async function fetchPairwiseCorrelation(x: string, y: string): Promise<PairwiseCorrelationResponse> {
  return fetchJson<PairwiseCorrelationResponse>(`/api/correlation/pair?x=${x}&y=${y}`);
}

// Inferential Statistics APIs (Phase 8)
import {
  InferenceSummaryResponse,
  NormalDistributionResponse,
  PoissonDistributionResponse,
  CltSimulationResponse,
  ConfidenceIntervalResponse,
  ZTestResponse,
  TTestResponse,
  AnovaResponse,
  ErrorAnalysisResponse,
} from "@/types/inference";

export async function fetchInferenceSummary(): Promise<InferenceSummaryResponse> {
  return fetchJson<InferenceSummaryResponse>("/api/inference/summary");
}

export async function fetchNormalDistribution(
  variable: string = "temp",
  xVal?: number
): Promise<NormalDistributionResponse> {
  const query = xVal !== undefined ? `?variable=${variable}&x_val=${xVal}` : `?variable=${variable}`;
  return fetchJson<NormalDistributionResponse>(`/api/inference/distribution/normal${query}`);
}

export async function fetchPoissonDistribution(
  variable: string = "count"
): Promise<PoissonDistributionResponse> {
  return fetchJson<PoissonDistributionResponse>(`/api/inference/distribution/poisson?variable=${variable}`);
}

export async function fetchCltSimulation(
  variable: string = "count",
  sampleSize: number = 30,
  numSamples: number = 500,
  seed: number = 42
): Promise<CltSimulationResponse> {
  return fetchJson<CltSimulationResponse>(
    `/api/inference/clt?variable=${variable}&sample_size=${sampleSize}&num_samples=${numSamples}&seed=${seed}`
  );
}

export async function fetchConfidenceInterval(
  variable: string = "count",
  confidenceLevel: number = 0.95
): Promise<ConfidenceIntervalResponse> {
  return fetchJson<ConfidenceIntervalResponse>(
    `/api/inference/confidence-interval?variable=${variable}&confidence_level=${confidenceLevel}`
  );
}

export async function fetchZTest(
  variable: string = "temp",
  hypothesizedMean: number = 20.0,
  knownStd: number = 7.5,
  alternative: string = "two-sided",
  alpha: number = 0.05
): Promise<ZTestResponse> {
  return fetchJson<ZTestResponse>(
    `/api/inference/z-test?variable=${variable}&hypothesized_mean=${hypothesizedMean}&known_std=${knownStd}&alternative=${alternative}&alpha=${alpha}`
  );
}

export async function fetchTTest(params: {
  testType: "one-sample" | "two-sample";
  variable: string;
  hypothesizedMean?: number;
  groupVariable?: string;
  group1?: string;
  group2?: string;
  alternative?: string;
  alpha?: number;
}): Promise<TTestResponse> {
  const q = new URLSearchParams();
  q.append("test_type", params.testType);
  q.append("variable", params.variable);
  if (params.hypothesizedMean !== undefined) q.append("hypothesized_mean", params.hypothesizedMean.toString());
  if (params.groupVariable) q.append("group_variable", params.groupVariable);
  if (params.group1 !== undefined) q.append("group1", params.group1);
  if (params.group2 !== undefined) q.append("group2", params.group2);
  if (params.alternative) q.append("alternative", params.alternative);
  if (params.alpha !== undefined) q.append("alpha", params.alpha.toString());
  return fetchJson<TTestResponse>(`/api/inference/t-test?${q.toString()}`);
}

export async function fetchAnova(
  variable: string = "count",
  groupVariable: string = "season",
  alpha: number = 0.05
): Promise<AnovaResponse> {
  return fetchJson<AnovaResponse>(
    `/api/inference/anova?variable=${variable}&group_variable=${groupVariable}&alpha=${alpha}`
  );
}

export async function fetchErrorAnalysis(
  alpha: number = 0.05,
  effectSize: number = 0.2,
  sampleSize: number = 100
): Promise<ErrorAnalysisResponse> {
  return fetchJson<ErrorAnalysisResponse>(
    `/api/inference/error-analysis?alpha=${alpha}&effect_size=${effectSize}&sample_size=${sampleSize}`
  );
}

// Regression APIs (Phase 9)
import {
  RegressionSummaryResponse,
  SimpleRegressionResponse,
  MultipleRegressionResponse,
  RegressionMetricsResponse,
  CoefficientsAnalysisResponse,
  ResidualDiagnosticsResponse,
} from "@/types/regression";

export async function fetchRegressionSummary(): Promise<RegressionSummaryResponse> {
  return fetchJson<RegressionSummaryResponse>("/api/regression/summary");
}

export async function fetchSimpleRegression(predictor: string = "temp"): Promise<SimpleRegressionResponse> {
  return fetchJson<SimpleRegressionResponse>(`/api/regression/simple?predictor=${predictor}`);
}

export async function fetchMultipleRegression(predictors?: string[]): Promise<MultipleRegressionResponse> {
  const query = predictors && predictors.length > 0 ? `?predictors=${predictors.join(",")}` : "";
  return fetchJson<MultipleRegressionResponse>(`/api/regression/multiple${query}`);
}

export async function fetchRegressionMetrics(
  modelType: string = "multiple",
  predictor?: string,
  predictors?: string[]
): Promise<RegressionMetricsResponse> {
  const q = new URLSearchParams();
  q.append("model_type", modelType);
  if (predictor) q.append("predictor", predictor);
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  return fetchJson<RegressionMetricsResponse>(`/api/regression/metrics?${q.toString()}`);
}

export async function fetchRegressionCoefficients(
  modelType: string = "multiple",
  predictor?: string,
  predictors?: string[],
  alpha: number = 0.05
): Promise<CoefficientsAnalysisResponse> {
  const q = new URLSearchParams();
  q.append("model_type", modelType);
  if (predictor) q.append("predictor", predictor);
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  q.append("alpha", alpha.toString());
  return fetchJson<CoefficientsAnalysisResponse>(`/api/regression/coefficients?${q.toString()}`);
}

export async function fetchRegressionResiduals(
  modelType: string = "multiple",
  predictor?: string,
  predictors?: string[],
  sampleSize: number = 1000
): Promise<ResidualDiagnosticsResponse> {
  const q = new URLSearchParams();
  q.append("model_type", modelType);
  if (predictor) q.append("predictor", predictor);
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  q.append("sample_size", sampleSize.toString());
  return fetchJson<ResidualDiagnosticsResponse>(`/api/regression/residuals?${q.toString()}`);
}

// Validation APIs (Phase 10)
import {
  ValidationSummaryResponse,
  HoldoutResponse,
  KFoldResponse,
  StratifiedKFoldResponse,
  LoocvResponse,
  BootstrapResponse,
  ValidationCompareResponse,
} from "@/types/validation";

export async function fetchValidationSummary(): Promise<ValidationSummaryResponse> {
  return fetchJson<ValidationSummaryResponse>("/api/validation/summary");
}

export async function fetchHoldoutValidation(
  testSize: number = 0.2,
  randomState: number = 42,
  predictors?: string[]
): Promise<HoldoutResponse> {
  const q = new URLSearchParams();
  q.append("test_size", testSize.toString());
  q.append("random_state", randomState.toString());
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  return fetchJson<HoldoutResponse>(`/api/validation/holdout?${q.toString()}`);
}

export async function fetchKFoldValidation(
  nSplits: number = 5,
  shuffle: boolean = true,
  randomState: number = 42,
  predictors?: string[]
): Promise<KFoldResponse> {
  const q = new URLSearchParams();
  q.append("n_splits", nSplits.toString());
  q.append("shuffle", shuffle.toString());
  q.append("random_state", randomState.toString());
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  return fetchJson<KFoldResponse>(`/api/validation/kfold?${q.toString()}`);
}

export async function fetchStratifiedKFoldValidation(
  nSplits: number = 5,
  nBins: number = 5,
  randomState: number = 42,
  predictors?: string[]
): Promise<StratifiedKFoldResponse> {
  const q = new URLSearchParams();
  q.append("n_splits", nSplits.toString());
  q.append("n_bins", nBins.toString());
  q.append("random_state", randomState.toString());
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  return fetchJson<StratifiedKFoldResponse>(`/api/validation/stratified-kfold?${q.toString()}`);
}

export async function fetchLoocvValidation(
  mode: string = "exact_leverage",
  subsetSize: number = 500,
  predictors?: string[]
): Promise<LoocvResponse> {
  const q = new URLSearchParams();
  q.append("mode", mode);
  q.append("subset_size", subsetSize.toString());
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  return fetchJson<LoocvResponse>(`/api/validation/loocv?${q.toString()}`);
}

export async function fetchBootstrapValidation(
  nIterations: number = 50,
  randomState: number = 42,
  predictors?: string[]
): Promise<BootstrapResponse> {
  const q = new URLSearchParams();
  q.append("n_iterations", nIterations.toString());
  q.append("random_state", randomState.toString());
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  return fetchJson<BootstrapResponse>(`/api/validation/bootstrap?${q.toString()}`);
}

export async function fetchCompareValidation(
  predictors?: string[]
): Promise<ValidationCompareResponse> {
  const q = new URLSearchParams();
  if (predictors && predictors.length > 0) q.append("predictors", predictors.join(","));
  const queryStr = q.toString() ? `?${q.toString()}` : "";
  return fetchJson<ValidationCompareResponse>(`/api/validation/compare${queryStr}`);
}

// Anomaly Detection APIs (Phase 11)
import {
  AnomalySummaryResponse,
  StatisticalResponse,
  DistanceResponse,
  DensityResponse,
  OutlierExplorerResponse,
  EvaluationResponse,
  ImbalanceResponse,
} from "@/types/anomaly";

export async function fetchAnomalySummary(): Promise<AnomalySummaryResponse> {
  return fetchJson<AnomalySummaryResponse>("/api/anomaly/summary");
}

export async function fetchStatisticalDetection(
  variable: string = "count",
  iqrMultiplier: number = 1.5,
  zscoreThreshold: number = 3.0
): Promise<StatisticalResponse> {
  const q = new URLSearchParams();
  q.append("variable", variable);
  q.append("iqr_multiplier", iqrMultiplier.toString());
  q.append("zscore_threshold", zscoreThreshold.toString());
  return fetchJson<StatisticalResponse>(`/api/anomaly/statistical?${q.toString()}`);
}

export async function fetchDistanceDetection(
  features?: string[],
  nNeighbors: number = 5,
  topN: number = 50
): Promise<DistanceResponse> {
  const q = new URLSearchParams();
  if (features && features.length > 0) q.append("features", features.join(","));
  q.append("n_neighbors", nNeighbors.toString());
  q.append("top_n", topN.toString());
  return fetchJson<DistanceResponse>(`/api/anomaly/distance?${q.toString()}`);
}

export async function fetchDensityDetection(
  features?: string[],
  nNeighbors: number = 20,
  contamination: number = 0.05
): Promise<DensityResponse> {
  const q = new URLSearchParams();
  if (features && features.length > 0) q.append("features", features.join(","));
  q.append("n_neighbors", nNeighbors.toString());
  q.append("contamination", contamination.toString());
  return fetchJson<DensityResponse>(`/api/anomaly/density?${q.toString()}`);
}

export async function fetchOutlierExplorer(params: {
  variable?: string;
  method?: string;
  iqrMultiplier?: number;
  zscoreThreshold?: number;
  features?: string[];
  nNeighbors?: number;
  contamination?: number;
  scatterX?: string;
  scatterY?: string;
  scatterSample?: number;
}): Promise<OutlierExplorerResponse> {
  const q = new URLSearchParams();
  if (params.variable) q.append("variable", params.variable);
  if (params.method) q.append("method", params.method);
  if (params.iqrMultiplier !== undefined) q.append("iqr_multiplier", params.iqrMultiplier.toString());
  if (params.zscoreThreshold !== undefined) q.append("zscore_threshold", params.zscoreThreshold.toString());
  if (params.features && params.features.length > 0) q.append("features", params.features.join(","));
  if (params.nNeighbors !== undefined) q.append("n_neighbors", params.nNeighbors.toString());
  if (params.contamination !== undefined) q.append("contamination", params.contamination.toString());
  if (params.scatterX) q.append("scatter_x", params.scatterX);
  if (params.scatterY) q.append("scatter_y", params.scatterY);
  if (params.scatterSample !== undefined) q.append("scatter_sample", params.scatterSample.toString());
  return fetchJson<OutlierExplorerResponse>(`/api/anomaly/outliers?${q.toString()}`);
}

export async function fetchAnomalyEvaluation(
  targetPercentile: number = 95.0,
  variable: string = "count",
  features?: string[],
  testSize: number = 0.2,
  randomState: number = 42
): Promise<EvaluationResponse> {
  const q = new URLSearchParams();
  q.append("target_percentile", targetPercentile.toString());
  q.append("variable", variable);
  if (features && features.length > 0) q.append("features", features.join(","));
  q.append("test_size", testSize.toString());
  q.append("random_state", randomState.toString());
  return fetchJson<EvaluationResponse>(`/api/anomaly/evaluation?${q.toString()}`);
}

export async function fetchAnomalyImbalance(
  targetPercentile: number = 95.0,
  variable: string = "count",
  features?: string[],
  testSize: number = 0.2,
  randomState: number = 42,
  applySmote: boolean = true
): Promise<ImbalanceResponse> {
  const q = new URLSearchParams();
  q.append("target_percentile", targetPercentile.toString());
  q.append("variable", variable);
  if (features && features.length > 0) q.append("features", features.join(","));
  q.append("test_size", testSize.toString());
  q.append("random_state", randomState.toString());
  q.append("apply_smote", applySmote.toString());
  return fetchJson<ImbalanceResponse>(`/api/anomaly/imbalance?${q.toString()}`);
}

// ===========================================================================
// Time Series APIs (Phase 12)
// ===========================================================================
import {
  TimeSeriesSummaryResponse,
  TimeSeriesComponentsResponse,
  MovingAverageResponse,
  ExponentialSmoothingResponse,
  TrendAnalysisResponse,
  DecompositionResponse,
  StationarityResponse,
  ArimaAnalysisResponse,
} from "@/types/timeseries";

export async function fetchTimeSeriesSummary(): Promise<TimeSeriesSummaryResponse> {
  return fetchJson<TimeSeriesSummaryResponse>("/api/timeseries/summary");
}

export async function fetchTimeSeriesComponents(params?: {
  targetVariable?: string;
  startDate?: string;
  endDate?: string;
  sampleLimit?: number;
}): Promise<TimeSeriesComponentsResponse> {
  const q = new URLSearchParams();
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.startDate) q.append("start_date", params.startDate);
  if (params?.endDate) q.append("end_date", params.endDate);
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  return fetchJson<TimeSeriesComponentsResponse>(`/api/timeseries/components?${q.toString()}`);
}

export async function fetchMovingAverage(params?: {
  targetVariable?: string;
  windowSize?: number;
  centered?: boolean;
  sampleLimit?: number;
}): Promise<MovingAverageResponse> {
  const q = new URLSearchParams();
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.windowSize !== undefined) q.append("window_size", params.windowSize.toString());
  if (params?.centered !== undefined) q.append("centered", params.centered.toString());
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  return fetchJson<MovingAverageResponse>(`/api/timeseries/moving-average?${q.toString()}`);
}

export async function fetchExponentialSmoothing(params?: {
  targetVariable?: string;
  alpha?: number;
  sampleLimit?: number;
}): Promise<ExponentialSmoothingResponse> {
  const q = new URLSearchParams();
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.alpha !== undefined) q.append("alpha", params.alpha.toString());
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  return fetchJson<ExponentialSmoothingResponse>(`/api/timeseries/exponential-smoothing?${q.toString()}`);
}

export async function fetchTrendAnalysis(
  targetVariable: string = "count",
  sampleLimit: number = 500
): Promise<TrendAnalysisResponse> {
  const q = new URLSearchParams();
  q.append("target_variable", targetVariable);
  q.append("sample_limit", sampleLimit.toString());
  return fetchJson<TrendAnalysisResponse>(`/api/timeseries/trend?${q.toString()}`);
}

export async function fetchSeasonalDecomposition(params?: {
  targetVariable?: string;
  model?: string;
  period?: number;
  sampleLimit?: number;
}): Promise<DecompositionResponse> {
  const q = new URLSearchParams();
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.model) q.append("model", params.model);
  if (params?.period !== undefined) q.append("period", params.period.toString());
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  return fetchJson<DecompositionResponse>(`/api/timeseries/decomposition?${q.toString()}`);
}

export async function fetchStationarity(params?: {
  targetVariable?: string;
  difference?: boolean;
  significanceLevel?: number;
  sampleLimit?: number;
}): Promise<StationarityResponse> {
  const q = new URLSearchParams();
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.difference !== undefined) q.append("difference", params.difference.toString());
  if (params?.significanceLevel !== undefined) q.append("significance_level", params.significanceLevel.toString());
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  return fetchJson<StationarityResponse>(`/api/timeseries/stationarity?${q.toString()}`);
}

export async function fetchArimaAnalysis(params?: {
  targetVariable?: string;
  p?: number;
  d?: number;
  q?: number;
  maxObs?: number;
}): Promise<ArimaAnalysisResponse> {
  const q = new URLSearchParams();
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.p !== undefined) q.append("p", params.p.toString());
  if (params?.d !== undefined) q.append("d", params.d.toString());
  if (params?.q !== undefined) q.append("q", params.q.toString());
  if (params?.maxObs !== undefined) q.append("max_obs", params.maxObs.toString());
  return fetchJson<ArimaAnalysisResponse>(`/api/timeseries/arima-analysis?${q.toString()}`);
}

// ===========================================================================
// Phase 13 — Time Series Forecasting APIs
// ===========================================================================
import {
  ForecastingSummaryResponse,
  ForecastingSplitResponse,
  BaselineForecastsResponse,
  SmoothingForecastResponse,
  ArimaForecastResponse,
  ModelEvaluationResponse,
  ModelComparisonResponse,
} from "@/types/forecasting";

export async function fetchForecastingSummary(): Promise<ForecastingSummaryResponse> {
  return fetchJson<ForecastingSummaryResponse>("/api/forecasting/summary");
}

export async function fetchForecastingSplit(params?: {
  validationFraction?: number;
  horizon?: number;
  targetVariable?: string;
}): Promise<ForecastingSplitResponse> {
  const q = new URLSearchParams();
  if (params?.validationFraction !== undefined) q.append("validation_fraction", params.validationFraction.toString());
  if (params?.horizon !== undefined) q.append("horizon", params.horizon.toString());
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  const qs = q.toString();
  return fetchJson<ForecastingSplitResponse>(`/api/forecasting/split${qs ? `?${qs}` : ""}`);
}

export async function fetchBaselineForecasts(params?: {
  validationFraction?: number;
  horizon?: number;
  windowSize?: number;
  seasonalPeriod?: number;
  targetVariable?: string;
  sampleLimit?: number;
}): Promise<BaselineForecastsResponse> {
  const q = new URLSearchParams();
  if (params?.validationFraction !== undefined) q.append("validation_fraction", params.validationFraction.toString());
  if (params?.horizon !== undefined) q.append("horizon", params.horizon.toString());
  if (params?.windowSize !== undefined) q.append("window_size", params.windowSize.toString());
  if (params?.seasonalPeriod !== undefined) q.append("seasonal_period", params.seasonalPeriod.toString());
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  const qs = q.toString();
  return fetchJson<BaselineForecastsResponse>(`/api/forecasting/naive${qs ? `?${qs}` : ""}`);
}

export async function fetchSmoothingForecasts(params?: {
  modelType?: "ses" | "holt" | "holt_winters";
  validationFraction?: number;
  horizon?: number;
  alpha?: number;
  seasonalPeriod?: number;
  targetVariable?: string;
  sampleLimit?: number;
}): Promise<SmoothingForecastResponse> {
  const q = new URLSearchParams();
  if (params?.modelType) q.append("model_type", params.modelType);
  if (params?.validationFraction !== undefined) q.append("validation_fraction", params.validationFraction.toString());
  if (params?.horizon !== undefined) q.append("horizon", params.horizon.toString());
  if (params?.alpha !== undefined) q.append("alpha", params.alpha.toString());
  if (params?.seasonalPeriod !== undefined) q.append("seasonal_period", params.seasonalPeriod.toString());
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  const qs = q.toString();
  return fetchJson<SmoothingForecastResponse>(`/api/forecasting/smoothing${qs ? `?${qs}` : ""}`);
}

export async function fetchArimaForecast(params?: {
  p?: number;
  d?: number;
  q?: number;
  validationFraction?: number;
  horizon?: number;
  targetVariable?: string;
  sampleLimit?: number;
}): Promise<ArimaForecastResponse> {
  const q = new URLSearchParams();
  if (params?.p !== undefined) q.append("p", params.p.toString());
  if (params?.d !== undefined) q.append("d", params.d.toString());
  if (params?.q !== undefined) q.append("q", params.q.toString());
  if (params?.validationFraction !== undefined) q.append("validation_fraction", params.validationFraction.toString());
  if (params?.horizon !== undefined) q.append("horizon", params.horizon.toString());
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  if (params?.sampleLimit !== undefined) q.append("sample_limit", params.sampleLimit.toString());
  const qs = q.toString();
  return fetchJson<ArimaForecastResponse>(`/api/forecasting/arima${qs ? `?${qs}` : ""}`);
}

export async function fetchModelEvaluation(params?: {
  modelName?: string;
  validationFraction?: number;
  horizon?: number;
  seasonalPeriod?: number;
  windowSize?: number;
  alpha?: number;
  p?: number;
  d?: number;
  q?: number;
  targetVariable?: string;
}): Promise<ModelEvaluationResponse> {
  const q = new URLSearchParams();
  if (params?.modelName) q.append("model_name", params.modelName);
  if (params?.validationFraction !== undefined) q.append("validation_fraction", params.validationFraction.toString());
  if (params?.horizon !== undefined) q.append("horizon", params.horizon.toString());
  if (params?.seasonalPeriod !== undefined) q.append("seasonal_period", params.seasonalPeriod.toString());
  if (params?.windowSize !== undefined) q.append("window_size", params.windowSize.toString());
  if (params?.alpha !== undefined) q.append("alpha", params.alpha.toString());
  if (params?.p !== undefined) q.append("p", params.p.toString());
  if (params?.d !== undefined) q.append("d", params.d.toString());
  if (params?.q !== undefined) q.append("q", params.q.toString());
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  const qs = q.toString();
  return fetchJson<ModelEvaluationResponse>(`/api/forecasting/evaluate${qs ? `?${qs}` : ""}`);
}

export async function fetchModelComparison(params?: {
  validationFraction?: number;
  horizon?: number;
  seasonalPeriod?: number;
  arimaP?: number;
  arimaD?: number;
  arimaQ?: number;
  targetVariable?: string;
}): Promise<ModelComparisonResponse> {
  const q = new URLSearchParams();
  if (params?.validationFraction !== undefined) q.append("validation_fraction", params.validationFraction.toString());
  if (params?.horizon !== undefined) q.append("horizon", params.horizon.toString());
  if (params?.seasonalPeriod !== undefined) q.append("seasonal_period", params.seasonalPeriod.toString());
  if (params?.arimaP !== undefined) q.append("arima_p", params.arimaP.toString());
  if (params?.arimaD !== undefined) q.append("arima_d", params.arimaD.toString());
  if (params?.arimaQ !== undefined) q.append("arima_q", params.arimaQ.toString());
  if (params?.targetVariable) q.append("target_variable", params.targetVariable);
  const qs = q.toString();
  return fetchJson<ModelComparisonResponse>(`/api/forecasting/compare${qs ? `?${qs}` : ""}`);
}

// ===========================================================================
// Phase 14 — Insights & Analytical Conclusions APIs
// ===========================================================================
import {
  InsightsSummaryResponse,
  KeyFindingsResponse,
  UnifiedModelComparisonResponse,
  EvidenceRecommendationsResponse,
  DataQualityAuditResponse,
} from "@/types/insights";

export async function fetchInsightsSummary(): Promise<InsightsSummaryResponse> {
  return fetchJson<InsightsSummaryResponse>("/api/insights/summary");
}

export async function fetchKeyFindings(): Promise<KeyFindingsResponse> {
  return fetchJson<KeyFindingsResponse>("/api/insights/key-findings");
}

export async function fetchUnifiedModelComparison(): Promise<UnifiedModelComparisonResponse> {
  return fetchJson<UnifiedModelComparisonResponse>("/api/insights/model-comparison");
}

export async function fetchEvidenceRecommendations(): Promise<EvidenceRecommendationsResponse> {
  return fetchJson<EvidenceRecommendationsResponse>("/api/insights/recommendations");
}

export async function fetchDataQualityAudit(): Promise<DataQualityAuditResponse> {
  return fetchJson<DataQualityAuditResponse>("/api/insights/data-quality");
}




