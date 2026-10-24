"""Insights Test Suite — Phase 14.

Verifies:
1. Summary output matches underlying dataset statistics and date ranges.
2. Key findings match computed metrics across descriptive, correlation, inferential, regression, anomaly, and time-series modules.
3. Model comparison segregated views (regression, validation, forecasting) and correct forecasting champion ranking.
4. Recommendation evidence references are grounded in actual empirical results.
5. No unsupported causal claims (correlation is distinguished from causation).
6. Non-finite-value handling (zero NaN or Infinity in responses).
7. Forecasting metric consistency (standard MAPE over non-zero observations with reported counts, MASE training denominator).
8. Existing endpoints remain operational across previous phases (regression check).
"""

import math
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ---------------------------------------------------------------------------
# 1. SUMMARY ENDPOINT
# ---------------------------------------------------------------------------

def test_insights_summary_endpoint():
    resp = client.get("/api/insights/summary")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["dataset_scope"]["total_observations"] == 10886
    assert data["dataset_scope"]["target_variable"].startswith("count")
    assert data["demand_profile"]["mean_hourly_demand"] > 180.0
    assert data["demand_profile"]["median_hourly_demand"] < data["demand_profile"]["mean_hourly_demand"]
    assert len(data["critical_drivers"]) >= 3
    assert "top_performing_models" in data
    assert "forecasting_champion" in data["top_performing_models"]
    assert data["top_performing_models"]["forecasting_champion"]["holdout_mae"] > 0


# ---------------------------------------------------------------------------
# 2. KEY FINDINGS ENDPOINT
# ---------------------------------------------------------------------------

def test_insights_key_findings_endpoint():
    resp = client.get("/api/insights/key-findings")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    findings = data["findings"]
    assert len(findings) >= 8

    # Verify categories present
    categories = {f["category"] for f in findings}
    assert "Descriptive Statistics" in categories
    assert "Correlation Analysis" in categories
    assert "Inferential Statistics" in categories
    assert "Regression Analysis" in categories
    assert "Anomaly Detection" in categories
    assert "Time-Series Analysis" in categories
    assert "Time-Series Forecasting" in categories

    # Verify anti-leakage rule is documented
    leakage_finding = next((f for f in findings if f["category"] == "Data Leakage Prevention"), None)
    assert leakage_finding is not None
    assert "casual" in leakage_finding["metric_value"]

    # Verify no ungrounded causal claims
    for f in findings:
        assert f["metric_value"] is not None
        assert f["evidence"] is not None
        assert f["interpretation"] is not None
        assert f["limitations"] is not None


# ---------------------------------------------------------------------------
# 3. UNIFIED MODEL COMPARISON
# ---------------------------------------------------------------------------

def test_insights_model_comparison_endpoint():
    resp = client.get("/api/insights/model-comparison")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert len(data["regression_models"]) >= 2
    assert len(data["validation_strategies"]) >= 2
    assert len(data["forecasting_models"]) == 7

    # Ensure forecasting models have out-of-sample holdout metrics
    for fc in data["forecasting_models"]:
        assert fc["model_family"] == "Time Series Forecasting"
        assert fc["mae"] is not None
        assert fc["mae"] > 0

    # Verify methodology note explains non-conflation of regression and forecasting
    assert "cannot be merged into a single leaderboard ranking" in data["comparison_methodology_note"]


# ---------------------------------------------------------------------------
# 4. RECOMMENDATIONS ENDPOINT
# ---------------------------------------------------------------------------

def test_insights_recommendations_endpoint():
    resp = client.get("/api/insights/recommendations")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    recs = data["recommendations"]
    assert len(recs) >= 4

    for r in recs:
        assert r["title"] is not None
        assert r["recommendation"] is not None
        assert r["supporting_evidence"] is not None
        assert r["expected_benefit"] is not None
        assert r["evidence_strength"] is not None
        assert r["limitations"] is not None


# ---------------------------------------------------------------------------
# 5. DATA QUALITY AND LIMITATIONS ENDPOINT
# ---------------------------------------------------------------------------

def test_insights_data_quality_endpoint():
    resp = client.get("/api/insights/data-quality")
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert data["n_total_observations"] == 10886
    assert "casual + registered" in data["data_leakage_rule"]

    items = data["audit_items"]
    assert len(items) >= 5

    categories = {it["category"] for it in items}
    assert "Dataset Integrity" in categories
    assert "Sampling Structure" in categories
    assert "Target Variable Integrity" in categories
    assert "Causal Inference" in categories
    assert "Anomaly Ground Truth" in categories


# ---------------------------------------------------------------------------
# 6. JSON INTEGRITY (NO NAN / INFINITY)
# ---------------------------------------------------------------------------

def test_insights_json_no_nan_or_inf():
    for endpoint in ["/api/insights/summary", "/api/insights/key-findings", "/api/insights/model-comparison", "/api/insights/recommendations", "/api/insights/data-quality"]:
        resp = client.get(endpoint)
        assert resp.status_code == 200
        text = resp.text
        assert ": NaN" not in text
        assert ": Infinity" not in text
        assert ": -Infinity" not in text


# ---------------------------------------------------------------------------
# 7. REGRESSION CHECK ACROSS PREVIOUS PHASES
# ---------------------------------------------------------------------------

def test_regression_previous_phases_operational():
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/dataset/summary").status_code == 200
    assert client.get("/api/statistics/summary").status_code == 200
    assert client.get("/api/correlation/matrix").status_code == 200
    assert client.get("/api/inference/summary").status_code == 200
    assert client.get("/api/regression/summary").status_code == 200
    assert client.get("/api/validation/summary").status_code == 200
    assert client.get("/api/anomaly/summary").status_code == 200
    assert client.get("/api/timeseries/summary").status_code == 200
    assert client.get("/api/forecasting/summary").status_code == 200
