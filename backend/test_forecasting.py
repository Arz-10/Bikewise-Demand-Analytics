"""Forecasting Test Suite — Phase 13.

Verifies:
1. Chronological train-validation splitting without shuffling.
2. Training and validation timestamps have strictly no overlap.
3. Forecast values align with validation timestamps.
4. Naive forecast calculation: flat projection of origin y_T.
5. Seasonal naive calculation: exact periodic projection (m=24).
6. Moving-average forecast: uses only training window at forecast origin.
7. Exponential smoothing (SES, Holt, Holt-Winters) parameter bounds & forecasts.
8. ARIMA(p,d,q) fitting, parameter validation, and 95% confidence intervals.
9. MAE and RMSE matching independent numpy/scipy calculations.
10. MAPE zero-protection policy.
11. MASE scaling denominator computed strictly from training data.
12. Fair model comparison tournament on identical holdout partition.
13. Parameter boundary rejection (HTTP 400 / 422).
14. No NaN or Infinity in JSON responses.
15. Existing module endpoints remain operational (regression check).
"""

import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import forecasting_service

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. SUMMARY ENDPOINT
# ---------------------------------------------------------------------------

def test_forecasting_summary_endpoint():
    resp = client.get("/api/forecasting/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["n_total_observations"] == 10886
    assert data["target_variable"] == "count"
    assert "Kaggle Split Architecture" in data["kaggle_structure_notice"]
    assert "casual and registered may never be included" in data["data_leakage_rule"]
    assert len(data["supported_models"]) == 7
    assert "MAE" in data["metrics_definition"]
    assert "MASE" in data["metrics_definition"]


# ---------------------------------------------------------------------------
# 2. CHRONOLOGICAL SPLIT VERIFICATION
# ---------------------------------------------------------------------------

def test_chronological_split_integrity():
    resp = client.get("/api/forecasting/split?validation_fraction=0.20")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    train_info = data["train"]
    val_info = data["validation"]
    integrity = data["chronological_integrity"]

    # Check counts sum to total
    assert train_info["n_observations"] + val_info["n_observations"] == data["n_total"]
    assert data["n_total"] == 10886

    # Verify strictly no timestamp overlap
    train_end = pd.to_datetime(train_info["end_timestamp"])
    val_start = pd.to_datetime(val_info["start_timestamp"])
    assert train_end < val_start, "Train end timestamp must precede validation start timestamp"
    assert integrity["no_timestamp_overlap"] is True
    assert integrity["shuffled"] is False


def test_chronological_split_bounds():
    # Below min fraction (0.05)
    r1 = client.get("/api/forecasting/split?validation_fraction=0.01")
    assert r1.status_code == 422 or r1.status_code == 400

    # Above max fraction (0.40)
    r2 = client.get("/api/forecasting/split?validation_fraction=0.50")
    assert r2.status_code == 422 or r2.status_code == 400


# ---------------------------------------------------------------------------
# 3. BASELINE MODELS (NAIVE, SEASONAL NAIVE, MOVING AVERAGE)
# ---------------------------------------------------------------------------

def test_baseline_models_endpoint():
    resp = client.get("/api/forecasting/naive?validation_fraction=0.10&window_size=24&seasonal_period=24")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    models = data["models"]
    assert "naive" in models
    assert "seasonal_naive" in models
    assert "moving_average" in models

    # Check metrics existence
    for m in ["naive", "seasonal_naive", "moving_average"]:
        met = models[m]["metrics"]
        assert met["mae"] > 0
        assert met["rmse"] > 0
        assert met["mape"] > 0
        assert met["mase"] > 0

    # Check calculation card details
    assert "calculation_details" in data
    assert "naive_formula" in data["calculation_details"]
    assert "s_naive_formula" in data["calculation_details"]

    # Verify Seasonal Naive drastically beats flat Naive due to diurnal commuter peaks
    assert models["seasonal_naive"]["metrics"]["mae"] < models["naive"]["metrics"]["mae"]


def test_independent_naive_forecast_calculation():
    """Verify Naive forecast equals the origin training value y_T."""
    df = forecasting_service._get_sorted_df()
    train_df, val_df, n_train, n_val = forecasting_service._compute_split(df, validation_fraction=0.15)
    expected_y_T = float(train_df["count"].iloc[-1])

    res = forecasting_service.get_baseline_forecasts(validation_fraction=0.15)
    naive_val = res["models"]["naive"]["origin_value"]
    assert math.isclose(naive_val, round(expected_y_T, 4), rel_tol=1e-3)


def test_moving_average_does_not_leak_validation():
    """Verify MA forecast uses only the final w training points."""
    df = forecasting_service._get_sorted_df()
    train_df, val_df, n_train, n_val = forecasting_service._compute_split(df, validation_fraction=0.10)
    w = 24
    expected_ma = float(train_df["count"].iloc[-w:].mean())

    res = forecasting_service.get_baseline_forecasts(validation_fraction=0.10, window_size=w)
    reported_ma = res["models"]["moving_average"]["origin_average"]
    assert math.isclose(reported_ma, round(expected_ma, 4), rel_tol=1e-3)


# ---------------------------------------------------------------------------
# 4. EXPONENTIAL SMOOTHING MODELS
# ---------------------------------------------------------------------------

def test_smoothing_ses_endpoint():
    resp = client.get("/api/forecasting/smoothing?model_type=ses&validation_fraction=0.10")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_type"] == "ses"
    assert "alpha" in data["parameters"]
    assert data["metrics"]["mae"] > 0
    assert len(data["series_data"]) > 0


def test_smoothing_holt_endpoint():
    resp = client.get("/api/forecasting/smoothing?model_type=holt&validation_fraction=0.10")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_type"] == "holt"
    assert "alpha" in data["parameters"]
    assert "beta" in data["parameters"]
    assert data["metrics"]["rmse"] > 0


def test_smoothing_holt_winters_endpoint():
    resp = client.get("/api/forecasting/smoothing?model_type=holt_winters&validation_fraction=0.10&seasonal_period=24")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["model_type"] == "holt_winters"
    assert "alpha" in data["parameters"]
    assert "beta" in data["parameters"]
    assert "gamma" in data["parameters"]
    # Holt-Winters seasonal captures diurnal peaks, beating SES
    ses_resp = client.get("/api/forecasting/smoothing?model_type=ses&validation_fraction=0.10")
    assert data["metrics"]["mae"] < ses_resp.json()["metrics"]["mae"]


def test_smoothing_invalid_model():
    resp = client.get("/api/forecasting/smoothing?model_type=invalid_model")
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# 5. ARIMA FORECASTING
# ---------------------------------------------------------------------------

def test_arima_forecasting_endpoint():
    resp = client.get("/api/forecasting/arima?p=1&d=1&q=1&validation_fraction=0.08")
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["order"] == {"p": 1, "d": 1, "q": 1}
    assert data["metrics"]["mae"] > 0
    assert data["aic"] is not None
    assert data["bic"] is not None
    assert len(data["coefficients"]) >= 2
    # Verify 95% confidence intervals exist in series_data
    first_pt = data["series_data"][0]
    assert "lower_95" in first_pt
    assert "upper_95" in first_pt
    assert first_pt["upper_95"] >= first_pt["lower_95"]


def test_arima_invalid_orders():
    # p > 5
    r1 = client.get("/api/forecasting/arima?p=9")
    assert r1.status_code == 422 or r1.status_code == 400
    # d > 2
    r2 = client.get("/api/forecasting/arima?d=4")
    assert r2.status_code == 422 or r2.status_code == 400


# ---------------------------------------------------------------------------
# 6. EVALUATION METRICS INDEPENDENT VERIFICATION
# ---------------------------------------------------------------------------

def test_independent_metric_calculations():
    actuals = np.array([100.0, 150.0, 200.0, 80.0, 0.0])
    forecasts = np.array([110.0, 140.0, 180.0, 90.0, 10.0])
    train_history = np.array([50.0, 60.0, 80.0, 90.0, 100.0])

    # Expected MAE = (|10| + |10| + |20| + |10| + |10|) / 5 = 60 / 5 = 12.0
    expected_mae = 12.0
    # Expected RMSE = sqrt((100 + 100 + 400 + 100 + 100) / 5) = sqrt(800 / 5) = sqrt(160) = 12.6491
    expected_rmse = math.sqrt(160.0)

    # MASE denominator on train: diffs = [10, 20, 10, 10], mean = 50 / 4 = 12.5
    # MASE = 12.0 / 12.5 = 0.96
    expected_mase = 12.0 / 12.5

    metrics = forecasting_service._compute_metrics(actuals, forecasts, train_history, mase_m=1)
    assert math.isclose(metrics["mae"], expected_mae, abs_tol=1e-3)
    assert math.isclose(metrics["rmse"], round(expected_rmse, 4), abs_tol=1e-3)
    assert math.isclose(metrics["mase"], round(expected_mase, 4), abs_tol=1e-3)
    assert metrics["n_eval_observations"] == 5


def test_mape_zero_handling():
    """Verify MAPE calculates safely without division-by-zero errors when target is zero."""
    actuals = np.array([0.0, 0.0, 100.0])
    forecasts = np.array([5.0, 2.0, 105.0])
    train_history = np.array([50.0, 60.0, 70.0])

    metrics = forecasting_service._compute_metrics(actuals, forecasts, train_history, mase_m=1)
    assert metrics["mape"] is not None
    assert not math.isnan(metrics["mape"])
    assert not math.isinf(metrics["mape"])


def test_mase_denominator_uses_train_only():
    """Verify MASE denominator is computed exclusively from training data and never validation."""
    train_1 = np.array([10.0, 20.0, 30.0, 40.0])  # diffs are all 10 -> denom = 10
    train_2 = np.array([10.0, 50.0, 90.0, 130.0])  # diffs are all 40 -> denom = 40
    val = np.array([100.0, 100.0])
    pred = np.array([110.0, 110.0])  # mae = 10

    m1 = forecasting_service._compute_metrics(val, pred, train_1, mase_m=1)
    m2 = forecasting_service._compute_metrics(val, pred, train_2, mase_m=1)

    assert math.isclose(m1["mase"], 1.0, abs_tol=1e-3)   # 10 / 10 = 1.0
    assert math.isclose(m2["mase"], 0.25, abs_tol=1e-3)  # 10 / 40 = 0.25


# ---------------------------------------------------------------------------
# 7. UNIFIED EVALUATE ENDPOINT
# ---------------------------------------------------------------------------

def test_unified_evaluate_endpoint():
    for mod in ["naive", "seasonal_naive", "moving_average", "ses", "holt", "holt_winters", "arima"]:
        resp = client.get(f"/api/forecasting/evaluate?model_name={mod}&validation_fraction=0.08")
        assert resp.status_code == 200, f"Model {mod} failed: {resp.text}"
        data = resp.json()
        assert "metrics" in data
        assert data["metrics"]["mae"] > 0
        assert len(data["series_data"]) > 0


# ---------------------------------------------------------------------------
# 8. MODEL COMPARISON TOURNAMENT
# ---------------------------------------------------------------------------

def test_model_comparison_tournament():
    resp = client.get("/api/forecasting/compare?validation_fraction=0.10&seasonal_period=24")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert len(data["models"]) == 7
    # Verify all models ran on the identical partition
    assert data["n_val"] > 0
    assert "start" in data["evaluation_partition"]
    assert "end" in data["evaluation_partition"]

    # Verify best model is determined dynamically
    best = data["best_model"]
    assert best["model_id"] is not None
    assert best["mae"] > 0

    # Ensure ranking is ordered by MAE ascending
    valid_ranks = [m for m in data["models"] if "rank" in m]
    for i in range(len(valid_ranks) - 1):
        assert valid_ranks[i]["metrics"]["mae"] <= valid_ranks[i+1]["metrics"]["mae"]


# ---------------------------------------------------------------------------
# 9. JSON INTEGRITY (NO NAN / INFINITY)
# ---------------------------------------------------------------------------

def test_json_serialization_no_nan_or_inf():
    resp = client.get("/api/forecasting/compare?validation_fraction=0.10")
    text = resp.text
    assert "NaN" not in text
    assert "Infinity" not in text
    assert "-Infinity" not in text


# ---------------------------------------------------------------------------
# 10. PRESERVATION OF EXISTING PHASE ENDPOINTS (REGRESSION CHECK)
# ---------------------------------------------------------------------------

def test_existing_endpoints_remain_operational():
    # Phase 1: Health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200

    # Phase 2: Dataset
    r_data = client.get("/api/dataset/summary")
    assert r_data.status_code == 200

    # Phase 4: Correlation
    r_corr = client.get("/api/correlation/matrix")
    assert r_corr.status_code == 200

    # Phase 7: Regression
    r_reg = client.get("/api/regression/simple?feature=temp")
    assert r_reg.status_code == 200

    # Phase 11: Anomaly
    r_anom = client.get("/api/anomaly/summary")
    assert r_anom.status_code == 200

    # Phase 12: Timeseries
    r_ts = client.get("/api/timeseries/summary")
    assert r_ts.status_code == 200
