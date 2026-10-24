import math
import json
import numpy as np
import statsmodels.api as sm
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_regression_endpoints():
    print("=== TEST 1: GET /api/regression/summary ===")
    res = client.get("/api/regression/summary")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    summary = res.json()
    assert summary["target"] == "count"
    assert summary["sample_size"] == 10886
    assert "casual" in summary["forbidden_leakage_features"]
    assert "registered" in summary["forbidden_leakage_features"]
    assert "temp" in summary["available_predictors"]
    assert "season" in summary["available_predictors"]
    print("Summary endpoint PASSED.")

    print("\n=== TEST 2: GET /api/regression/simple ===")
    res = client.get("/api/regression/simple?predictor=temp")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    simple = res.json()
    assert simple["predictor"] == "temp"
    assert simple["target"] == "count"
    assert simple["sample_size"] == 10886
    assert abs(simple["slope"] - 9.17) < 0.1
    assert abs(simple["intercept"] - 6.05) < 0.2
    assert abs(simple["r_squared"] - 0.1556) < 0.005
    assert simple["mae"] > 0
    assert simple["mape"] > 0
    assert simple["mape_zero_excluded"] == 0  # min count in train.csv is 1
    assert len(simple["scatter_points"]) > 100
    assert len(simple["regression_line"]) == 2
    assert "calculation_details" in simple

    # Test leakage prevention: casual and registered
    res_leak = client.get("/api/regression/simple?predictor=casual")
    assert res_leak.status_code == 400
    assert "Data Leakage" in res_leak.json()["detail"]

    res_leak2 = client.get("/api/regression/simple?predictor=count")
    assert res_leak2.status_code == 400

    res_inv = client.get("/api/regression/simple?predictor=invalid_feature")
    assert res_inv.status_code == 400
    print("Simple regression endpoint PASSED.")

    print("\n=== TEST 3: GET /api/regression/multiple ===")
    # Default predictors
    res = client.get("/api/regression/multiple")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    mult = res.json()
    assert mult["target"] == "count"
    assert mult["sample_size"] == 10886
    assert mult["r_squared"] > simple["r_squared"]  # multiple R2 > simple R2
    assert abs(mult["r_squared"] - 0.3462) < 0.01
    assert abs(mult["adjusted_r_squared"] - 0.3455) < 0.01
    assert mult["f_statistic"] > 100.0
    assert mult["f_p_value"] == 0.0
    assert len(mult["coefficients"]) >= 7
    # Check dummy encoding for season and weather
    features = [c["feature"] for c in mult["coefficients"]]
    assert "season_2" in features or "season_3" in features
    assert "weather_2" in features or "weather_3" in features

    # Test explicit query params with comma-separated list
    res_custom = client.get("/api/regression/multiple?predictors=hour,temp,humidity,windspeed")
    assert res_custom.status_code == 200
    mult_custom = res_custom.json()
    assert len(mult_custom["predictors"]) == 4

    # Test leakage prevention in multiple regression
    res_mult_leak = client.get("/api/regression/multiple?predictors=temp,casual,humidity")
    assert res_mult_leak.status_code == 400
    print("Multiple regression endpoint PASSED.")

    print("\n=== TEST 4: GET /api/regression/metrics ===")
    res_m = client.get("/api/regression/metrics?model_type=multiple")
    assert res_m.status_code == 200, f"Error {res_m.status_code}: {res_m.text}"
    metrics_data = res_m.json()
    metric_ids = [m["id"] for m in metrics_data["metrics"]]
    assert "r_squared" in metric_ids
    assert "adj_r_squared" in metric_ids
    assert "mae" in metric_ids
    assert "rmse" in metric_ids
    assert "mape" in metric_ids

    res_m_simple = client.get("/api/regression/metrics?model_type=simple&predictor=temp")
    assert res_m_simple.status_code == 200
    print("Regression metrics endpoint PASSED.")

    print("\n=== TEST 5: GET /api/regression/coefficients ===")
    res_c = client.get("/api/regression/coefficients?model_type=multiple&alpha=0.05")
    assert res_c.status_code == 200, f"Error {res_c.status_code}: {res_c.text}"
    coef_data = res_c.json()
    assert len(coef_data["coefficients"]) > 5
    for c in coef_data["coefficients"]:
        assert "estimate" in c
        assert "standard_error" in c
        assert "t_statistic" in c
        assert "p_value" in c
        assert "ci_lower" in c
        assert "ci_upper" in c
        assert "partial_interpretation" in c
    print("Coefficients analysis endpoint PASSED.")

    print("\n=== TEST 6: GET /api/regression/residuals ===")
    res_r = client.get("/api/regression/residuals?model_type=multiple&sample_size=500")
    assert res_r.status_code == 200, f"Error {res_r.status_code}: {res_r.text}"
    resid_data = res_r.json()
    assert abs(resid_data["residual_moments"]["mean"]) < 1e-4  # OLS residual mean is ~0
    assert resid_data["residual_moments"]["std"] > 0
    assert len(resid_data["histogram"]) > 10
    assert len(resid_data["actual_vs_predicted"]) > 100
    assert len(resid_data["residuals_vs_fitted"]) > 100
    assert len(resid_data["identity_line"]) == 2
    print("Residual diagnostics endpoint PASSED.")

    print("\n=== TEST 7: REGRESSION SAFETY CHECK FOR EXISTING ENDPOINTS ===")
    res_inf = client.get("/api/inference/summary")
    assert res_inf.status_code == 200
    res_corr = client.get("/api/correlation/matrix")
    assert res_corr.status_code == 200
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    print("Existing Phase 1-8 endpoints remain fully operational.")

    print("\nALL REGRESSION ANALYSIS BACKEND TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_regression_endpoints()
