"""API endpoint tests (health, predict, invalid input, model info)."""
from __future__ import annotations


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_root_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "docs" in r.json()


def test_model_info(client):
    r = client.get("/api/v1/model/info")
    assert r.status_code == 200
    body = r.json()
    assert body["loaded"] is True
    assert body["n_features"] > 0


def test_model_metrics(client):
    r = client.get("/api/v1/model/metrics")
    assert r.status_code == 200
    assert "roc_auc" in r.json()


def test_predict_valid(client, valid_payload):
    r = client.post("/api/v1/predict", json=valid_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["default_probability"] <= 1.0
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert body["recovery_priority"] in {
        "NORMAL",
        "FOLLOW_UP",
        "HIGH_PRIORITY",
        "CRITICAL_PRIORITY",
    }
    assert body["model_version"]


def test_predict_persists_borrower(client, valid_payload):
    client.post("/api/v1/predict", json=valid_payload)
    r = client.get("/api/v1/borrowers/BR-FIXTURE-0001")
    assert r.status_code == 200
    assert r.json()["borrower_id"] == "BR-FIXTURE-0001"
    assert r.json()["default_probability"] is not None


def test_predict_invalid_missing_field(client, valid_payload):
    payload = dict(valid_payload)
    payload.pop("age")
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_invalid_negative(client, valid_payload):
    payload = dict(valid_payload)
    payload["age"] = -5
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_impossible_probability_field(client, valid_payload):
    payload = dict(valid_payload)
    payload["RevolvingUtilizationOfUnsecuredLines"] = 9999
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_extra_field_rejected(client, valid_payload):
    payload = dict(valid_payload)
    payload["unknown_field"] = 123
    r = client.post("/api/v1/predict", json=payload)
    assert r.status_code == 422


def test_predict_batch(client, valid_payload):
    items = [
        {"borrower_id": f"BATCH-{i}", "request": valid_payload} for i in range(3)
    ]
    r = client.post("/api/v1/predict/batch", json=items)
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 3
    assert len(body["predictions"]) == 3


def test_dashboard_stats(client, valid_payload):
    client.post("/api/v1/predict", json=valid_payload)
    r = client.get("/api/v1/dashboard/stats")
    assert r.status_code == 200
    assert r.json()["total_borrowers"] >= 1
    assert "avg_default_probability" in r.json()


def test_borrower_list(client, valid_payload):
    client.post("/api/v1/predict", json=valid_payload)
    r = client.get("/api/v1/borrowers")
    assert r.status_code == 200
    assert r.json()["total"] >= 1


def test_borrower_not_found(client):
    r = client.get("/api/v1/borrowers/DOES-NOT-EXIST")
    assert r.status_code == 404


def test_recovery_queue(client, valid_payload):
    client.post("/api/v1/predict", json=valid_payload)
    r = client.get("/api/v1/recovery/priority")
    assert r.status_code == 200
    assert "items" in r.json()


def test_explain(client, valid_payload):
    r = client.post("/api/v1/explain", json=valid_payload)
    assert r.status_code == 200, r.text
    body = r.json()
    assert "top_factors" in body
    assert body["default_probability"] is not None
