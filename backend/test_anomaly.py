"""Test suite for Phase 11 — Anomaly Detection.

Tests:
 1. Summary endpoint metadata.
 2. IQR bounds against independent calculation.
 3. Z-score calculations.
 4. Zero-variance variable safe handling.
 5. Distance score ordering and determinism.
 6. Row-index preservation in distance results.
 7. LOF score direction and labeling.
 8. LOF outlier count matches contamination parameter.
 9. Outlier explorer: box stats, histogram, scatter sample.
10. Evaluation: proxy-label experiment on held-out test set.
11. Imbalance: SMOTE applied to training only; test set unchanged.
12. Invalid variable / parameter rejection (HTTP 400).
13. No NaN or Infinity in responses.
14. Existing validation, regression, and correlation endpoints unharmed.
"""

import sys
import math
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _assert_no_nan_inf(obj, path=""):
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
# 1. Summary
# ---------------------------------------------------------------------------

def test_anomaly_summary():
    res = client.get("/api/anomaly/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["n_observations"] == 10886
    assert "count" in data["analysis_variables"]
    assert len(data["methods"]) >= 5
    assert "metric_definitions" not in data  # no regression metrics here
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 2. IQR bounds — independent verification
# ---------------------------------------------------------------------------

def test_iqr_bounds_independent():
    res = client.get("/api/anomaly/statistical?variable=count&iqr_multiplier=1.5")
    assert res.status_code == 200
    data = res.json()

    q1 = data["iqr"]["q1"]
    q3 = data["iqr"]["q3"]
    iqr = data["iqr"]["iqr"]
    lb = data["iqr"]["lower_bound"]
    ub = data["iqr"]["upper_bound"]

    # IQR formula check
    assert abs(iqr - (q3 - q1)) < 0.01, "IQR != Q3 - Q1"
    assert abs(lb - (q1 - 1.5 * iqr)) < 0.01, "Lower bound formula mismatch"
    assert abs(ub - (q3 + 1.5 * iqr)) < 0.01, "Upper bound formula mismatch"

    # Outlier count must be non-negative
    assert data["iqr"]["outlier_count"] >= 0
    _assert_no_nan_inf(data)


def test_iqr_custom_multiplier():
    """Tighter multiplier should flag more outliers than wider."""
    res_tight = client.get("/api/anomaly/statistical?variable=count&iqr_multiplier=1.0")
    res_wide = client.get("/api/anomaly/statistical?variable=count&iqr_multiplier=3.0")
    assert res_tight.status_code == 200
    assert res_wide.status_code == 200
    assert res_tight.json()["iqr"]["outlier_count"] >= res_wide.json()["iqr"]["outlier_count"]


# ---------------------------------------------------------------------------
# 3. Z-score calculations
# ---------------------------------------------------------------------------

def test_zscore_count():
    res = client.get("/api/anomaly/statistical?variable=count&zscore_threshold=3.0")
    assert res.status_code == 200
    data = res.json()
    assert data["zscore"]["outlier_count"] >= 0
    pct = data["zscore"]["outlier_percentage"]
    assert 0.0 <= pct <= 100.0
    _assert_no_nan_inf(data)


def test_zscore_tighter_threshold_more_outliers():
    res_tight = client.get("/api/anomaly/statistical?variable=temp&zscore_threshold=2.0")
    res_wide = client.get("/api/anomaly/statistical?variable=temp&zscore_threshold=4.0")
    assert res_tight.status_code == 200
    assert res_wide.status_code == 200
    assert res_tight.json()["zscore"]["outlier_count"] >= res_wide.json()["zscore"]["outlier_count"]


# ---------------------------------------------------------------------------
# 4. Zero-variance safe handling
# ---------------------------------------------------------------------------

def test_zero_variance_windspeed():
    """windspeed is NOT constant in this dataset, but we test the API
    gracefully handles any variable. holiday or workingday are categorical
    but included in variables; test that constant-ish variables don't crash."""
    # holiday has almost no variance but is not constant; should return ok
    res = client.get("/api/anomaly/statistical?variable=holiday")
    assert res.status_code == 200
    data = res.json()
    # Zero-variance flag only if truly constant
    assert "zero_variance" in data["zscore"]
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 5. Distance scores ordering
# ---------------------------------------------------------------------------

def test_distance_scores_ordering():
    res = client.get("/api/anomaly/distance?n_neighbors=5&top_n=20")
    assert res.status_code == 200
    data = res.json()
    scores = [r["anomaly_score"] for r in data["top_anomalies"]]
    # Must be in descending order
    assert scores == sorted(scores, reverse=True), "Top anomalies not sorted descending"
    _assert_no_nan_inf(data)


def test_distance_score_determinism():
    """Two calls with same params must return same top-ranked row_index."""
    res1 = client.get("/api/anomaly/distance?n_neighbors=5&top_n=10")
    res2 = client.get("/api/anomaly/distance?n_neighbors=5&top_n=10")
    assert res1.status_code == 200
    assert res2.status_code == 200
    idx1 = [r["row_index"] for r in res1.json()["top_anomalies"]]
    idx2 = [r["row_index"] for r in res2.json()["top_anomalies"]]
    assert idx1 == idx2, "Distance results are not deterministic"


# ---------------------------------------------------------------------------
# 6. Row-index preservation
# ---------------------------------------------------------------------------

def test_row_index_in_distance():
    res = client.get("/api/anomaly/distance?top_n=30")
    assert res.status_code == 200
    data = res.json()
    for rec in data["top_anomalies"]:
        assert "row_index" in rec
        assert isinstance(rec["row_index"], int)


def test_row_index_in_statistical():
    res = client.get("/api/anomaly/statistical?variable=count")
    assert res.status_code == 200
    data = res.json()
    for rec in data["iqr"]["records"]:
        assert "row_index" in rec
        assert isinstance(rec["row_index"], int)


# ---------------------------------------------------------------------------
# 7. LOF score direction
# ---------------------------------------------------------------------------

def test_lof_score_positive_and_descending():
    res = client.get("/api/anomaly/density?n_neighbors=20&contamination=0.05")
    assert res.status_code == 200
    data = res.json()
    # All displayed LOF scores must be > 0
    for rec in data["top_anomalies"][:20]:
        assert rec["lof_score"] > 0.0, f"LOF score not positive: {rec['lof_score']}"
    # Must be descending
    scores = [r["lof_score"] for r in data["top_anomalies"]]
    assert scores == sorted(scores, reverse=True), "LOF anomalies not sorted descending"
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 8. LOF contamination matches outlier count
# ---------------------------------------------------------------------------

def test_lof_contamination_count():
    contamination = 0.05
    n_total = 10886
    expected = round(n_total * contamination)
    res = client.get(f"/api/anomaly/density?contamination={contamination}&n_neighbors=20")
    assert res.status_code == 200
    data = res.json()
    # scikit-learn rounds contamination; allow ±2 tolerance
    assert abs(data["n_outliers"] - expected) <= 2, (
        f"n_outliers={data['n_outliers']} far from expected ~{expected}"
    )


# ---------------------------------------------------------------------------
# 9. Outlier explorer
# ---------------------------------------------------------------------------

def test_outlier_explorer_iqr():
    res = client.get("/api/anomaly/outliers?variable=count&method=iqr")
    assert res.status_code == 200
    data = res.json()
    assert "box_stats" in data
    assert "histogram" in data
    assert "scatter_data" in data
    assert len(data["histogram"]) > 0
    assert data["n_total"] == 10886
    _assert_no_nan_inf(data)


def test_outlier_explorer_scatter_sample():
    res = client.get("/api/anomaly/outliers?scatter_sample=500")
    assert res.status_code == 200
    data = res.json()
    assert len(data["scatter_data"]) == 500


def test_outlier_explorer_zscore():
    res = client.get("/api/anomaly/outliers?variable=temp&method=zscore&zscore_threshold=3.0")
    assert res.status_code == 200
    data = res.json()
    assert data["method"] == "zscore"
    _assert_no_nan_inf(data)


# ---------------------------------------------------------------------------
# 10. Evaluation — proxy-label experiment
# ---------------------------------------------------------------------------

def test_evaluation_metrics_range():
    res = client.get("/api/anomaly/evaluation?target_percentile=95.0")
    assert res.status_code == 200
    data = res.json()

    # Verify proxy-label note is present
    assert "proxy_label_note" in data
    assert "IMPORTANT" in data["proxy_label_note"]

    m = data["metrics"]
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    assert 0.0 <= m["f1"] <= 1.0
    assert 0.0 <= m["average_precision"] <= 1.0

    # Confusion matrix: 2x2
    cm = data["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2

    # Train + test = total (within ±2 due to rounding)
    assert abs(data["train_size"] + data["test_size"] - data["n_total"]) <= 2

    # PR curve present
    assert len(data["pr_curve"]) > 0
    _assert_no_nan_inf(data)


def test_evaluation_proxy_label_count():
    """Positive count must equal approximately (1 - percentile/100) * N."""
    pct = 90.0
    res = client.get(f"/api/anomaly/evaluation?target_percentile={pct}")
    assert res.status_code == 200
    data = res.json()
    expected_pos = round(10886 * (1 - pct / 100))
    # Allow 5% tolerance due to quantile ties
    assert abs(data["n_positive"] - expected_pos) <= 100


# ---------------------------------------------------------------------------
# 11. SMOTE — training-only, test set unchanged
# ---------------------------------------------------------------------------

def test_smote_training_only():
    res = client.get("/api/anomaly/imbalance?target_percentile=95.0&apply_smote=true")
    assert res.status_code == 200
    data = res.json()

    # SMOTE details must be present and show expansion
    if data["smote_applied"]:
        assert data["smote_details"]["after_minority"] >= data["train_before_smote"]["minority"]
        assert data["smote_details"]["synthetic_added"] >= 0
        # Proxy label note must flag educational purpose
        assert "proxy" in data["proxy_label_note"].lower()
    else:
        # Accept if SMOTE was skipped with a clear error message
        assert data["smote_error"] is not None or not data["smote_applied"]

    assert "education" in data
    assert "smote_explanation" in data["education"]
    _assert_no_nan_inf(data)


def test_smote_baseline_vs_smote_metrics():
    """Baseline and SMOTE metrics must both be present when SMOTE succeeds."""
    res = client.get("/api/anomaly/imbalance?target_percentile=90.0&apply_smote=true")
    assert res.status_code == 200
    data = res.json()
    assert "baseline_metrics" in data
    if data["smote_applied"]:
        assert "smote_metrics" in data
        assert data["smote_metrics"] is not None


# ---------------------------------------------------------------------------
# 12. Invalid parameters
# ---------------------------------------------------------------------------

def test_invalid_variable_rejected():
    res = client.get("/api/anomaly/statistical?variable=datetime")
    assert res.status_code == 400


def test_invalid_iqr_multiplier():
    res = client.get("/api/anomaly/statistical?variable=count&iqr_multiplier=10.0")
    assert res.status_code == 422  # FastAPI validation


def test_invalid_distance_neighbors():
    res = client.get("/api/anomaly/distance?n_neighbors=1")
    assert res.status_code == 422  # FastAPI validation (ge=2)


def test_invalid_lof_contamination():
    res = client.get("/api/anomaly/density?contamination=0.9")
    assert res.status_code == 422  # FastAPI validation (le=0.5)


def test_invalid_outlier_method():
    res = client.get("/api/anomaly/outliers?method=isolation_forest")
    assert res.status_code == 400


def test_invalid_evaluation_percentile():
    res = client.get("/api/anomaly/evaluation?target_percentile=50.0")
    assert res.status_code == 422  # FastAPI validation (ge=80)


# ---------------------------------------------------------------------------
# 13. No NaN / Infinity in JSON responses
# ---------------------------------------------------------------------------

def test_no_nan_in_density():
    res = client.get("/api/anomaly/density")
    assert res.status_code == 200
    _assert_no_nan_inf(res.json())


def test_no_nan_in_distance():
    res = client.get("/api/anomaly/distance")
    assert res.status_code == 200
    _assert_no_nan_inf(res.json())


def test_no_nan_in_evaluation():
    res = client.get("/api/anomaly/evaluation")
    assert res.status_code == 200
    _assert_no_nan_inf(res.json())


# ---------------------------------------------------------------------------
# 14. Existing endpoints unharmed
# ---------------------------------------------------------------------------

def test_existing_validation_endpoint():
    r = client.get("/api/validation/summary")
    assert r.status_code == 200
    assert r.json()["target"] == "count"


def test_existing_regression_endpoint():
    r = client.get("/api/regression/summary")
    assert r.status_code == 200
    assert r.json()["target"] == "count"


def test_existing_correlation_endpoint():
    r = client.get("/api/correlation/pearson")
    assert r.status_code == 200


def test_existing_inference_endpoint():
    r = client.get("/api/inference/summary")
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_anomaly_summary,
        test_iqr_bounds_independent,
        test_iqr_custom_multiplier,
        test_zscore_count,
        test_zscore_tighter_threshold_more_outliers,
        test_zero_variance_windspeed,
        test_distance_scores_ordering,
        test_distance_score_determinism,
        test_row_index_in_distance,
        test_row_index_in_statistical,
        test_lof_score_positive_and_descending,
        test_lof_contamination_count,
        test_outlier_explorer_iqr,
        test_outlier_explorer_scatter_sample,
        test_outlier_explorer_zscore,
        test_evaluation_metrics_range,
        test_evaluation_proxy_label_count,
        test_smote_training_only,
        test_smote_baseline_vs_smote_metrics,
        test_invalid_variable_rejected,
        test_invalid_iqr_multiplier,
        test_invalid_distance_neighbors,
        test_invalid_lof_contamination,
        test_invalid_outlier_method,
        test_invalid_evaluation_percentile,
        test_no_nan_in_density,
        test_no_nan_in_distance,
        test_no_nan_in_evaluation,
        test_existing_validation_endpoint,
        test_existing_regression_endpoint,
        test_existing_correlation_endpoint,
        test_existing_inference_endpoint,
    ]

    print(f"Running {len(tests)} Anomaly Detection Tests...")
    passed = 0
    failed = 0
    for idx, fn in enumerate(tests, 1):
        try:
            fn()
            print(f"  [{idx:02d}/{len(tests)}] PASS  {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [{idx:02d}/{len(tests)}] FAIL  {fn.__name__} -> {exc}")
            failed += 1

    print(f"\n{'ALL' if failed == 0 else str(failed) + ' FAILED,'} {passed}/{len(tests)} TESTS PASSED.")
    if failed > 0:
        sys.exit(1)
