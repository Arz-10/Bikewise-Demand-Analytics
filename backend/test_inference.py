import math
import json
import numpy as np
from scipy import stats
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_inference_endpoints():
    print("=== TEST 1: GET /api/inference/summary ===")
    res = client.get("/api/inference/summary")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    summary = res.json()
    assert summary["sample_size"] == 10886
    assert "count" in summary["supported_variables"]
    assert "season" in summary["supported_groups"]
    print("Summary endpoint PASSED.")

    print("\n=== TEST 2: GET /api/inference/distribution/normal ===")
    res = client.get("/api/inference/distribution/normal?variable=temp&x_val=25.0")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    norm = res.json()
    assert norm["sample_size"] == 10886
    assert abs(norm["mean"] - 20.23) < 0.1
    assert abs(norm["std"] - 7.79) < 0.1
    assert "z_score" in norm
    assert "normality_test" in norm
    assert len(norm["histogram"]) > 10
    assert len(norm["fitted_curve"]) > 20
    # Test invalid variable
    res_inv = client.get("/api/inference/distribution/normal?variable=invalid_var")
    assert res_inv.status_code == 400
    print("Normal distribution endpoint PASSED.")

    print("\n=== TEST 3: GET /api/inference/distribution/poisson ===")
    res = client.get("/api/inference/distribution/poisson?variable=count")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    pois = res.json()
    assert pois["sample_size"] == 10886
    assert abs(pois["lambda_hat"] - 191.57) < 0.5
    assert pois["is_overdispersed"] is True
    assert pois["dispersion_ratio"] > 10.0  # huge variance in count
    assert "chi_square_test" in pois
    assert len(pois["comparison_data"]) == 20
    print("Poisson distribution endpoint PASSED.")

    print("\n=== TEST 4: GET /api/inference/clt ===")
    res = client.get("/api/inference/clt?variable=count&sample_size=30&num_samples=500&seed=42")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    clt = res.json()
    assert clt["simulation_parameters"]["sample_size"] == 30
    assert clt["simulation_parameters"]["num_samples"] == 500
    # Check theoretical SE ~ 181.14 / sqrt(30) ~ 33.07
    theo_se = clt["sampling_distribution"]["theoretical_se"]
    emp_se = clt["sampling_distribution"]["empirical_se"]
    assert abs(theo_se - 33.07) < 1.0
    assert abs(emp_se - theo_se) / theo_se < 0.15  # within 15% of theoretical
    # Check bounds validation
    res_bad = client.get("/api/inference/clt?sample_size=1")
    assert res_bad.status_code in [400, 422]
    print("CLT simulation endpoint PASSED.")

    print("\n=== TEST 5: GET /api/inference/confidence-interval ===")
    res = client.get("/api/inference/confidence-interval?variable=count&confidence_level=0.95")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    ci = res.json()
    assert ci["confidence_level"] == 0.95
    assert ci["lower_bound"] < ci["mean"] < ci["upper_bound"]
    assert abs(ci["mean"] - 191.57) < 0.5
    assert abs(ci["margin_of_error"] - 3.41) < 0.5
    # Invalid confidence level
    res_bad_ci = client.get("/api/inference/confidence-interval?confidence_level=0.50")
    assert res_bad_ci.status_code == 400
    print("Confidence interval endpoint PASSED.")

    print("\n=== TEST 6: GET /api/inference/z-test ===")
    res = client.get("/api/inference/z-test?variable=temp&hypothesized_mean=20.0&known_std=7.5&alternative=two-sided&alpha=0.05")
    assert res.status_code == 200, f"Error {res.status_code}: {res.text}"
    zt = res.json()
    assert zt["hypothesized_mean"] == 20.0
    assert zt["known_std"] == 7.5
    # z = (20.23 - 20.0) / (7.5 / sqrt(10886)) = 0.23 / 0.07187 ~ 3.20
    assert abs(zt["test_statistic"] - 3.2) < 0.3
    assert zt["reject_null"] is True
    # Test invalid known_std <= 0
    res_bad_z = client.get("/api/inference/z-test?known_std=-1.0")
    assert res_bad_z.status_code in [400, 422]
    print("Z-test endpoint PASSED.")

    print("\n=== TEST 7: GET /api/inference/t-test ===")
    # A. One-sample t-test
    res_1t = client.get("/api/inference/t-test?test_type=one-sample&variable=count&hypothesized_mean=190.0&alpha=0.05")
    assert res_1t.status_code == 200, f"Error {res_1t.status_code}: {res_1t.text}"
    t1 = res_1t.json()
    assert t1["test_type"] == "one-sample"
    assert t1["degrees_of_freedom"] == 10885
    # B. Two-sample Welch's t-test (workingday: 0 vs 1)
    res_2t = client.get("/api/inference/t-test?test_type=two-sample&variable=count&group_variable=workingday&group1=0&group2=1")
    assert res_2t.status_code == 200, f"Error {res_2t.status_code}: {res_2t.text}"
    t2 = res_2t.json()
    assert t2["test_type"] == "two-sample"
    assert "degrees_of_freedom" in t2
    assert "confidence_interval_difference" in t2
    print("t-test endpoint (one-sample & two-sample Welch) PASSED.")

    print("\n=== TEST 8: GET /api/inference/anova ===")
    res_a = client.get("/api/inference/anova?variable=count&group_variable=season&alpha=0.05")
    assert res_a.status_code == 200, f"Error {res_a.status_code}: {res_a.text}"
    anova = res_a.json()
    assert anova["num_groups"] == 4
    assert anova["anova_table"]["df_between"] == 3
    assert anova["anova_table"]["df_within"] == 10886 - 4
    assert anova["anova_table"]["f_statistic"] > 50.0  # highly significant season differences
    assert anova["reject_null"] is True
    assert len(anova["post_hoc_tukey"]) == 6  # 4 choose 2 = 6 pairwise combinations
    assert "levene_test" in anova["assumption_diagnostics"]
    print("One-Way ANOVA & Tukey HSD endpoint PASSED.")

    print("\n=== TEST 9: GET /api/inference/error-analysis ===")
    res_e = client.get("/api/inference/error-analysis?alpha=0.05&effect_size=0.2&sample_size=100")
    assert res_e.status_code == 200, f"Error {res_e.status_code}: {res_e.text}"
    err = res_e.json()
    assert err["type_1_error_rate"] == 0.05
    assert 0.0 < err["statistical_power"] < 1.0
    assert abs(err["type_1_error_rate"] + (1 - err["type_1_error_rate"]) - 1.0) < 1e-6
    assert abs(err["statistical_power"] + err["type_2_error_rate"] - 1.0) < 1e-6
    assert len(err["decision_matrix"]) == 2
    assert len(err["power_curve"]) == 8
    print("Type I / Type II error analysis endpoint PASSED.")

    print("\n=== TEST 10: CHECK EXISTING ENDPOINTS OPERATIONAL ===")
    res_corr = client.get("/api/correlation/matrix")
    assert res_corr.status_code == 200
    res_health = client.get("/api/health")
    assert res_health.status_code == 200
    print("Existing endpoints still fully operational.")

    print("\nALL INFERENTIAL STATISTICS TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_inference_endpoints()
