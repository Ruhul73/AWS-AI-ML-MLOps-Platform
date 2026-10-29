"""
Section 6: Automated Verification Suite (scripts/test_endpoints.py)
Validates end-to-end contract compliance for Project 4.1 (Customer Churn Inference)
and Project 4.2 (Bedrock Serverless RAG with S3 source citations).
"""

import json
import pytest
from fastapi.testclient import TestClient
from project_4_1_churn_mlops.lambda_handler import lambda_handler
from project_4_2_bedrock_rag.rag_lambda_api import query_enterprise_rag
from project_4_2_bedrock_rag.app_ui import app

client = TestClient(app)


def test_churn_inference_contract():
    """
    Validates Step 4.1.B Lambda handler contract for high-risk customer profile:
    Features: [tenure=3, charges=110.0, calls=6, contract=0 (month-to-month)]
    """
    event = {"body": json.dumps({"features": [3, 110.0, 6, 0]})}  # High risk profile
    res = lambda_handler(event, None)
    body = json.loads(res["body"])

    assert res["statusCode"] == 200
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] in ["HIGH", "LOW"]
    assert body["risk_tier"] == "HIGH"


def test_churn_inference_low_risk_contract():
    """
    Validates Step 4.1.B Lambda handler contract for low-risk customer profile:
    Features: [tenure=60, charges=25.0, calls=0, contract=2 (two-year)]
    """
    event = {"body": json.dumps({"features": [60, 25.0, 0, 2]})}
    res = lambda_handler(event, None)
    body = json.loads(res["body"])

    assert res["statusCode"] == 200
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_tier"] == "LOW"


def test_bedrock_rag_citations():
    """
    Validates Step 4.2.A Bedrock RAG retrieval & synthesis:
    Ensures non-empty grounded answer and valid s3:// citation URIs.
    """
    res = query_enterprise_rag("What is the company reimbursement limit for home office equipment?")
    assert len(res["answer"]) > 20
    assert isinstance(res["citations"], list)
    assert len(res["citations"]) > 0
    # Phase 6 verification gate: RAG response includes valid s3:// source citation URIs
    for citation in res["citations"]:
        assert citation.startswith("s3://"), f"Citation must be an S3 URI: {citation}"
    assert any("remote_work_and_equipment_policy" in c for c in res["citations"])


def test_fastapi_ask_endpoint():
    """
    Validates Step 4.2.B FastAPI /ask route for interactive client.
    """
    response = client.post("/ask", json={"question": "What is the reimbursement limit for home office?"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert len(data["answer"]) > 20
    assert any("s3://" in c for c in data["citations"])


def test_fastapi_ask_validation_error():
    """
    Validates that empty questions return HTTP 400.
    """
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


def test_fastapi_predict_endpoint():
    """
    Validates Step 4.2.B FastAPI /predict route proxying to churn model.
    """
    response = client.post("/predict", json={
        "tenure": 3,
        "charges": 110.0,
        "calls": 6,
        "contract": 0
    })
    assert response.status_code == 200
    data = response.json()
    assert "churn_probability" in data
    assert "risk_tier" in data
    assert data["risk_tier"] == "HIGH"
