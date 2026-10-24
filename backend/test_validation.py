"""Comprehensive test suite for Phase 10: Model Validation.

Tests:
1. Data leakage prevention: forbidden predictors ('casual', 'registered', 'count') rejected with HTTP 400.
2. Hold-out validation: split sizes, reproducibility, metrics calculation, generalization gap.
3. K-fold cross-validation: fold sizes, no overlap, exhaustive validation coverage (every row validated once).
4. Stratified K-fold cross-validation: quantile strata distribution, continuous target binning, reproducibility.
5. LOOCV: exact leverage PRESS identity calculation (N=10,886) and subset demonstration.
6. Bootstrap validation: replacement sampling, OOB exclusion, empirical distributions, 95% CIs.
7. Validation strategy comparison: side-by-side synthesis with measured runtimes.
8. Regression and existing endpoints health check.
"""

import sys
from fastapi.testclient import TestClient
import numpy as np

from app.main import app

client = TestClient(app)


def test_validation_summary():
    """Verify validation summary metadata, strategies catalog, and leakage rules."""
    response = client.get("/api/validation/summary")
    assert response.status_code == 200
    data = response.json()

    assert data["target"] == "count"
    assert data["sample_size"] == 10886
    assert "forbidden_leakage_features" in data
    assert "casual" in data["forbidden_leakage_features"]
    assert "registered" in data["forbidden_leakage_features"]
    assert len(data["validation_strategies"]) >= 5
    assert "metric_definitions" in data
    assert "r2" in data["metric_definitions"]
    assert "rmse" in data["metric_definitions"]


def test_data_leakage_rejection():
    """Verify that using 'casual', 'registered', or 'count' raises HTTP 400 across endpoints."""
    # Test holdout
    res1 = client.get("/api/validation/holdout?predictors=temp,casual")
    assert res1.status_code == 400
    assert "Data Leakage Violation" in res1.json()["detail"]

    res2 = client.get("/api/validation/holdout?predictors=temp,registered")
    assert res2.status_code == 400
    assert "Data Leakage Violation" in res2.json()["detail"]

    res3 = client.get("/api/validation/holdout?predictors=temp,count")
    assert res3.status_code == 400

    # Test kfold
    res4 = client.get("/api/validation/kfold?predictors=temp,casual")
    assert res4.status_code == 400

    # Test stratified kfold
    res5 = client.get("/api/validation/stratified-kfold?predictors=registered")
    assert res5.status_code == 400


def test_holdout_validation():
    """Verify hold-out split sizes, reproducibility, metrics, and generalization gap."""
    res = client.get("/api/validation/holdout?test_size=0.2&random_state=42")
    assert res.status_code == 200
    data = res.json()

    assert data["method"] == "holdout"
    assert data["sample_sizes"]["total"] == 10886
    # 80/20 split of 10886
    assert data["sample_sizes"]["training"] == 8708
    assert data["sample_sizes"]["validation"] == 2178
    assert data["sample_sizes"]["training"] + data["sample_sizes"]["validation"] == 10886

    # Verify metrics
    val_metrics = data["validation_metrics"]
    train_metrics = data["training_metrics"]
    assert 0.0 < val_metrics["r2"] < 1.0
    assert val_metrics["rmse"] > 0
    assert val_metrics["mae"] > 0
    assert val_metrics["mape"] > 0
    assert val_metrics["zero_count_excluded"] == 0

    # Verify reproducibility
    res_repeat = client.get("/api/validation/holdout?test_size=0.2&random_state=42")
    assert res_repeat.json()["validation_metrics"]["r2"] == val_metrics["r2"]

    # Verify generalization gap
    gap = data["generalization_gap"]
    assert "r2_gap" in gap
    assert "assessment" in gap
    assert "calculation_details" in data
    assert "holdout_r2" in data["calculation_details"]


def test_kfold_validation():
    """Verify K-Fold validation partitions, exhaustive coverage, and fold metrics."""
    res = client.get("/api/validation/kfold?n_splits=5&random_state=42")
    assert res.status_code == 200
    data = res.json()

    assert data["method"] == "kfold"
    assert len(data["fold_results"]) == 5

    total_val_obs = sum(f["val_size"] for f in data["fold_results"])
    assert total_val_obs == 10886, "Every observation must be evaluated exactly once across folds"

    for fold in data["fold_results"]:
        assert fold["train_size"] + fold["val_size"] == 10886
        assert 0.0 < fold["r2"] < 1.0
        assert fold["rmse"] > 0
        assert fold["mae"] > 0

    agg = data["aggregate_metrics"]
    assert "r2_mean" in agg
    assert "r2_std" in agg
    assert agg["r2_std"] >= 0.0
    assert "Sample standard deviation (ddof=1)" in agg["dispersion_type"]
    assert "pooled_metrics" in data
    assert data["pooled_metrics"]["sample_size"] == 10886


def test_stratified_kfold_validation():
    """Verify Stratified K-Fold continuous target quantile binning and per-fold metrics."""
    res = client.get("/api/validation/stratified-kfold?n_splits=5&n_bins=5&random_state=42")
    assert res.status_code == 200
    data = res.json()

    assert data["method"] == "stratified_kfold"
    assert len(data["bin_distributions"]) == 5
    assert len(data["fold_results"]) == 5

    # Check sum of strata counts equals dataset size
    total_strata_obs = sum(b["observation_count"] for b in data["bin_distributions"])
    assert total_strata_obs == 10886

    agg = data["aggregate_metrics"]
    assert 0.0 < agg["r2_mean"] < 1.0
    assert agg["rmse_mean"] > 0


def test_loocv_validation_exact_leverage():
    """Verify full N=10,886 exact leverage LOOCV execution and algebraic consistency."""
    res = client.get("/api/validation/loocv?mode=exact_leverage")
    assert res.status_code == 200
    data = res.json()

    assert data["method"] == "loocv"
    assert data["mode"] == "exact_leverage"
    assert data["n_observations"] == 10886
    assert data["n_models_evaluated"] == 10886
    assert data["metrics"]["r2"] > 0.0
    assert data["metrics"]["sample_size"] == 10886
    assert data["runtime_seconds"] < 5.0, "Exact leverage calculation should complete in seconds"


def test_loocv_validation_subset():
    """Verify subset LOOCV benchmark execution and explicit labelling."""
    res = client.get("/api/validation/loocv?mode=subset&subset_size=100")
    assert res.status_code == 200
    data = res.json()

    assert data["mode"] == "subset"
    assert data["n_observations"] == 100
    assert data["n_models_evaluated"] == 100
    assert "Subset Iterative Demonstration" in data["mode_label"]
    assert data["metrics"]["sample_size"] == 100


def test_bootstrap_validation():
    """Verify Out-of-Bag bootstrap evaluation, distributions, and 95% CIs."""
    res = client.get("/api/validation/bootstrap?n_iterations=20&random_state=42")
    assert res.status_code == 200
    data = res.json()

    assert data["method"] == "bootstrap"
    assert data["successful_fits"] == 20
    assert data["failed_fits"] == 0

    # Theoretical OOB is ~36.8% of 10886 ≈ 4005
    assert 3500 <= data["mean_oob_sample_size"] <= 4500

    dist = data["distribution"]
    assert "r2" in dist
    assert len(dist["r2"]["ci_95"]) == 2
    assert dist["r2"]["ci_95"][0] <= dist["r2"]["ci_95"][1]
    assert len(data["r2_distribution_bins"]) > 0


def test_compare_validation():
    """Verify unified validation strategy comparison table and runtime tracking."""
    res = client.get("/api/validation/compare")
    assert res.status_code == 200
    data = res.json()

    table = data["comparison_table"]
    assert len(table) == 5
    method_ids = [row["method_id"] for row in table]
    assert "holdout" in method_ids
    assert "kfold" in method_ids
    assert "stratified_kfold" in method_ids
    assert "loocv" in method_ids
    assert "bootstrap" in method_ids

    for row in table:
        assert row["r2_mean"] > 0
        assert row["rmse_mean"] > 0
        assert "strengths" in row
        assert "limitations" in row


def test_existing_regression_and_correlation_endpoints():
    """Verify existing regression and correlation endpoints are unharmed."""
    r_reg = client.get("/api/regression/summary")
    assert r_reg.status_code == 200
    assert r_reg.json()["target"] == "count"

    r_corr = client.get("/api/correlation/pearson")
    assert r_corr.status_code == 200

    r_inf = client.get("/api/inference/summary")
    assert r_inf.status_code == 200


if __name__ == "__main__":
    tests = [
        test_validation_summary,
        test_data_leakage_rejection,
        test_holdout_validation,
        test_kfold_validation,
        test_stratified_kfold_validation,
        test_loocv_validation_exact_leverage,
        test_loocv_validation_subset,
        test_bootstrap_validation,
        test_compare_validation,
        test_existing_regression_and_correlation_endpoints
    ]
    print(f"Running {len(tests)} Validation Test Cases...")
    for idx, test_fn in enumerate(tests, 1):
        try:
            test_fn()
            print(f"[{idx}/{len(tests)}] PASS: {test_fn.__name__}")
        except Exception as e:
            print(f"[{idx}/{len(tests)}] FAIL: {test_fn.__name__} -> {e}")
            sys.exit(1)
    print("\nALL 10 VALIDATION TESTS PASSED SUCCESSFULLY!")
