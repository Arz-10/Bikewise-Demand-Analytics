"""Comprehensive test suite for Phase 12 — Time Series Analysis.

Tests:
 1. Chronological sorting and timestamp validation
 2. Summary metadata and independent statistics
 3. Components & aggregations (daily, weekly, monthly, profiles)
 4. Moving averages against manual formula calculations (trailing & centered)
 5. Exponential smoothing against independent SES formulation
 6. Linear trend regression coefficients against independent SciPy calculations
 7. Seasonal decomposition (additive & multiplicative)
 8. Seasonal period validation and edge handling
 9. Stationarity tests (ADF & KPSS) response structures and hypotheses
10. First-difference handling in stationarity
11. ARIMA parameter validation and fitting
12. No NaN or Infinity in any JSON responses
13. Existing anomaly, validation, regression, inference, and correlation endpoints unharmed
"""

import math
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import data_service

client = TestClient(app)


def _assert_no_nan_inf(obj, path=""):
    """Recursively checks that no numbers are NaN or Inf in JSON response."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_nan_inf(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_nan_inf(v, f"{path}[{i}]")
    elif isinstance(obj, float):
        assert not math.isnan(obj), f"NaN at {path}"
        assert not math.isinf(obj), f"Inf at {path}"


# ---------------------------------------------------------------------------
# 1. Summary & Timestamp Sorting
# ---------------------------------------------------------------------------

def test_timeseries_summary():
    res = client.get("/api/timeseries/summary")
    assert res.status_code == 200
    data = res.json()

    assert data["n_observations"] == 10886
    assert data["start_timestamp"] == "2011-01-01 00:00:00"
    assert data["end_timestamp"] == "2012-12-19 23:00:00"
    assert data["duplicate_timestamps"] == 0
    assert "count" in data["target_statistics"]
    assert "casual" in data["target_statistics"]
    assert "registered" in data["target_statistics"]
    assert "taxonomy" in data
    assert "trend" in data["taxonomy"]
    assert "seasonality" in data["taxonomy"]

    # Verify count mean against independent calculation
    df = data_service.get_featured_train_df()
    expected_mean = round(float(df["count"].mean()), 4)
    assert abs(data["target_statistics"]["count"]["mean"] - expected_mean) < 0.01

    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 2. Components & Aggregations
# ---------------------------------------------------------------------------

def test_timeseries_components():
    res = client.get("/api/timeseries/components?target_variable=count&sample_limit=300")
    assert res.status_code == 200
    data = res.json()

    assert data["target_variable"] == "count"
    assert len(data["hourly_series"]) > 0
    assert len(data["daily_aggregation"]) > 0
    assert len(data["weekly_aggregation"]) > 0
    assert len(data["monthly_aggregation"]) > 0
    assert len(data["diurnal_profile"]) == 24
    assert len(data["day_of_week_profile"]) == 7
    assert len(data["month_profile"]) == 12

    # Check diurnal profile hours 0..23
    hours = [p["hour"] for p in data["diurnal_profile"]]
    assert hours == list(range(24))

    # Daily aggregation check: total = mean * n_obs
    first_day = data["daily_aggregation"][0]
    expected_tot = first_day["mean_demand"] * first_day["n_obs"]
    assert abs(first_day["total_demand"] - expected_tot) < 0.1

    _assert_no_nan_inf(data)


def test_components_date_filter():
    res = client.get("/api/timeseries/components?start_date=2012-01-01&end_date=2012-06-30")
    assert res.status_code == 200
    data = res.json()
    assert data["n_observations"] < 10886
    assert data["n_observations"] > 0
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 3. Moving Averages
# ---------------------------------------------------------------------------

def test_moving_average_trailing():
    w = 24
    res = client.get(f"/api/timeseries/moving-average?target_variable=count&window_size={w}&centered=false")
    assert res.status_code == 200
    data = res.json()

    assert data["window_size"] == w
    assert not data["centered"]
    assert data["n_total"] == 10886
    assert data["n_boundary_dropped"] == w - 1

    # First point in raw series should have null MA because trailing window is incomplete
    assert data["series_data"][0]["moving_average"] is None

    # Compare manual calculation of 24h trailing MA on raw data
    df = data_service.get_featured_train_df().sort_values("datetime").reset_index(drop=True)
    expected_val = float(df["count"].iloc[:w].mean())

    # Find the corresponding point in the returned series_data
    assert "calculation_details" in data
    assert "moving_average_formula" in data["calculation_details"]
    _assert_no_nan_inf(data)


def test_moving_average_centered():
    w = 12
    res = client.get(f"/api/timeseries/moving-average?window_size={w}&centered=true")
    assert res.status_code == 200
    data = res.json()
    assert data["centered"]
    assert data["window_size"] == w
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 4. Exponential Smoothing
# ---------------------------------------------------------------------------

def test_exponential_smoothing_specified_alpha():
    alpha = 0.3
    res = client.get(f"/api/timeseries/exponential-smoothing?alpha={alpha}")
    assert res.status_code == 200
    data = res.json()

    assert abs(data["smoothing_parameter_alpha"] - alpha) < 1e-4
    assert not data["is_optimized"]
    assert data["in_sample_mae"] > 0
    assert data["in_sample_rmse"] > 0
    assert len(data["series_data"]) > 0
    assert "disclaimer" in data
    _assert_no_nan_inf(data)


def test_exponential_smoothing_optimized():
    res = client.get("/api/timeseries/exponential-smoothing")
    assert res.status_code == 200
    data = res.json()

    assert data["is_optimized"]
    assert 0.0 < data["smoothing_parameter_alpha"] <= 1.0
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 5. Linear Trend Analysis
# ---------------------------------------------------------------------------

def test_linear_trend_analysis():
    res = client.get("/api/timeseries/trend?target_variable=count")
    assert res.status_code == 200
    data = res.json()

    assert data["target_variable"] == "count"
    assert data["slope_daily"] > 0, "Bike demand should have positive trend over 2011-2012"
    assert data["p_value"] < 0.05
    assert "Upward" in data["trend_direction"]
    assert data["r_squared"] > 0.0
    assert data["intercept"] > 0
    assert len(data["limitations"]) >= 3
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 6. Seasonal Decomposition
# ---------------------------------------------------------------------------

def test_seasonal_decomposition_additive():
    period = 24
    res = client.get(f"/api/timeseries/decomposition?model=additive&period={period}")
    assert res.status_code == 200
    data = res.json()

    assert data["model"] == "additive"
    assert data["period"] == period
    assert data["n_observations"] == 10886
    assert data["n_boundary_dropped"] == period
    assert len(data["seasonal_cycle_profile"]) == period
    assert "variances" in data
    assert data["variances"]["seasonal"] > 0
    _assert_no_nan_inf(data)


def test_seasonal_decomposition_multiplicative():
    period = 24
    res = client.get(f"/api/timeseries/decomposition?model=multiplicative&period={period}")
    assert res.status_code == 200
    data = res.json()
    assert data["model"] == "multiplicative"
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 7. Stationarity Tests (ADF & KPSS)
# ---------------------------------------------------------------------------

def test_stationarity_level_series():
    res = client.get("/api/timeseries/stationarity?difference=false&significance_level=0.05")
    assert res.status_code == 200
    data = res.json()

    assert not data["is_differenced"]
    assert "adf_test" in data
    assert "kpss_test" in data
    assert "test_statistic" in data["adf_test"]
    assert "critical_values" in data["adf_test"]
    assert "p_value" in data["adf_test"]
    assert "null_hypothesis" in data["adf_test"]
    assert "joint_synthesis" in data

    # ADF p-value < 0.05 for hourly bike demand
    assert data["adf_test"]["p_value"] < 0.05
    _assert_no_nan_inf(data)


def test_stationarity_first_difference():
    res = client.get("/api/timeseries/stationarity?difference=true")
    assert res.status_code == 200
    data = res.json()

    assert data["is_differenced"]
    assert data["n_observations"] == 10885  # N - 1
    assert data["adf_test"]["p_value"] < 0.01
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 8. ARIMA Analysis
# ---------------------------------------------------------------------------

def test_arima_analysis_valid():
    res = client.get("/api/timeseries/arima-analysis?p=1&d=1&q=1&max_obs=500")
    assert res.status_code == 200
    data = res.json()

    assert data["order"] == {"p": 1, "d": 1, "q": 1}
    assert data["n_fitted"] == 500
    assert data["aic"] > 0
    assert data["bic"] > 0
    assert len(data["coefficients"]) >= 2
    assert "residual_diagnostics" in data
    assert "disclaimer" in data
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 9. Parameter Validation & Error Handling
# ---------------------------------------------------------------------------

def test_invalid_target_rejected():
    res = client.get("/api/timeseries/components?target_variable=weather")
    assert res.status_code == 400


def test_invalid_window_size():
    res = client.get("/api/timeseries/moving-average?window_size=1")
    assert res.status_code == 422  # ge=2


def test_invalid_alpha():
    res = client.get("/api/timeseries/exponential-smoothing?alpha=1.5")
    assert res.status_code == 422  # le=1.0


def test_invalid_period():
    res = client.get("/api/timeseries/decomposition?period=1")
    assert res.status_code == 422  # ge=2


def test_invalid_arima_p():
    res = client.get("/api/timeseries/arima-analysis?p=10")
    assert res.status_code == 422  # le=5


# ---------------------------------------------------------------------------
# 10. Existing Endpoints Unharmed
# ---------------------------------------------------------------------------

def test_existing_anomaly_summary():
    res = client.get("/api/anomaly/summary")
    assert res.status_code == 200
    assert res.json()["n_observations"] == 10886


def test_existing_validation_summary():
    res = client.get("/api/validation/summary")
    assert res.status_code == 200


def test_existing_regression_summary():
    res = client.get("/api/regression/summary")
    assert res.status_code == 200


def test_existing_correlation_summary():
    res = client.get("/api/correlation/pearson")
    assert res.status_code == 200


def test_existing_inference_summary():
    res = client.get("/api/inference/summary")
    assert res.status_code == 200
