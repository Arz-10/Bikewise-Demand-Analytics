import math
import json
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_endpoints():
    print("--- 1. Testing GET /api/correlation/pearson ---")
    res = client.get("/api/correlation/pearson")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    p_data = res.json()
    assert p_data["type"] == "pearson"
    assert p_data["sample_size"] == 10886
    matrix = p_data["matrix"]
    p_vals = p_data["p_values"]
    n = len(p_data["variables"])
    assert n == 7
    # Check symmetric and diagonal = 1, no NaN/Inf
    for i in range(n):
        for j in range(n):
            val = matrix[i][j]
            pval = p_vals[i][j]
            assert not math.isnan(val) and not math.isinf(val), f"NaN/Inf found at [{i}][{j}]"
            assert not math.isnan(pval) and not math.isinf(pval), f"NaN/Inf pval at [{i}][{j}]"
            if i == j:
                assert val == 1.0, f"Diagonal not 1.0 at [{i}][{j}]: {val}"
                assert pval == 0.0, f"Diagonal p-val not 0.0 at [{i}][{j}]: {pval}"
            else:
                assert matrix[i][j] == matrix[j][i], f"Matrix not symmetric at [{i}][{j}] vs [{j}][{i}]"
    print("Pearson matrix test PASSED. Matrix is symmetric, diagonal=1.0, N=10886, no NaN/Inf.")

    print("\n--- 2. Testing GET /api/correlation/spearman ---")
    res = client.get("/api/correlation/spearman")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    s_data = res.json()
    assert s_data["type"] == "spearman"
    assert s_data["sample_size"] == 10886
    s_matrix = s_data["matrix"]
    s_p_vals = s_data["p_values"]
    for i in range(n):
        for j in range(n):
            val = s_matrix[i][j]
            pval = s_p_vals[i][j]
            assert not math.isnan(val) and not math.isinf(val), f"NaN/Inf found at [{i}][{j}]"
            assert not math.isnan(pval) and not math.isinf(pval), f"NaN/Inf pval at [{i}][{j}]"
            if i == j:
                assert val == 1.0, f"Diagonal not 1.0 at [{i}][{j}]: {val}"
                assert pval == 0.0, f"Diagonal p-val not 0.0 at [{i}][{j}]: {pval}"
            else:
                assert s_matrix[i][j] == s_matrix[j][i], f"Spearman matrix not symmetric at [{i}][{j}]"
    print("Spearman matrix test PASSED. Matrix is symmetric, diagonal=1.0, N=10886, no NaN/Inf.")

    print("\n--- 3. Testing GET /api/correlation/matrix ---")
    res = client.get("/api/correlation/matrix")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    c_data = res.json()
    assert "pearson" in c_data and "spearman" in c_data
    assert c_data["sample_size"] == 10886
    print("Combined matrix test PASSED.")

    print("\n--- 4. Testing GET /api/correlation/pair?x=temp&y=count ---")
    res = client.get("/api/correlation/pair?x=temp&y=count")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    pair_data = res.json()
    assert pair_data["sample_size"] == 10886, f"Expected 10886, got {pair_data['sample_size']}"
    assert pair_data["is_identical"] is False
    assert pair_data["is_leakage"] is False
    assert pair_data["pearson"]["r"] == 0.394
    assert pair_data["spearman"]["rho"] == 0.408
    assert "calculation_details" in pair_data
    assert "pearson" in pair_data["calculation_details"]
    assert "spearman" in pair_data["calculation_details"]
    assert "p_value" in pair_data["calculation_details"]
    print("Pairwise correlation (temp x count) PASSED: r=0.394, rho=0.408, N=10886.")

    print("\n--- 5. Testing identical-variable selection (x=temp&y=temp) ---")
    res = client.get("/api/correlation/pair?x=temp&y=temp")
    assert res.status_code == 200, f"Expected 200, got {res.status_code}: {res.text}"
    ident_data = res.json()
    assert ident_data["is_identical"] is True
    assert ident_data["pearson"]["r"] == 1.0
    assert ident_data["pearson"]["p_value"] == 0.0
    assert ident_data["spearman"]["rho"] == 1.0
    assert ident_data["spearman"]["p_value"] == 0.0
    print("Identical variable selection PASSED: is_identical=True, r=1.0, rho=1.0, p=0.0.")

    print("\n--- 6. Testing invalid-variable handling ---")
    res = client.get("/api/correlation/pair?x=invalid_var&y=count")
    assert res.status_code == 400, f"Expected 400 for invalid variable, got {res.status_code}"
    print("Invalid variable test PASSED: correctly returns HTTP 400 Bad Request.")

    print("\n--- 7. Testing data leakage detection (casual x count) ---")
    res = client.get("/api/correlation/pair?x=casual&y=count")
    assert res.status_code == 200
    leak_data = res.json()
    assert leak_data["is_leakage"] is True
    print("Data leakage test PASSED: is_leakage=True for casual x count.")

    print("\nALL BACKEND CORRELATION TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_endpoints()
