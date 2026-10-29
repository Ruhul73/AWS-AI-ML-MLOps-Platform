# AWS Machine Learning & MLOps End-to-End Suite

[![Python Version](https://img.shields.io/badge/Python-3.11%20%7C%203.12-blue.svg)](https://python.org)
[![AWS Services](https://img.shields.io/badge/AWS-SageMaker%20%7C%20Bedrock%20%7C%20OpenSearch-orange.svg)](https://aws.amazon.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

Production-grade monorepo containing two complete, automated AWS AI/ML engineering pipelines built in accordance with Category 4 Autonomous Agent Build Specifications:
- **Project 4.1: Customer Churn Prediction MLOps Pipeline** (SageMaker XGBoost 1.7-1, 100% Data Capture, Model Drift Monitoring, Lambda Inference Proxy, API Gateway)
- **Project 4.2: Serverless Enterprise GenAI RAG Pipeline** (Amazon Bedrock Knowledge Bases, Titan Text Embeddings V2, OpenSearch Serverless HNSW index, Claude 3.5 Sonnet synthesis, grounded S3 citations)

---

## 1. Monorepo Directory Layout

```
aws-ai-mlops-suite/
├── .env.example                       # AWS_REGION, BUCKET_PREFIX, BEDROCK_KB_ID placeholders
├── requirements.txt                   # boto3, sagemaker, pandas, scikit-learn, fastapi, pytest
├── Makefile                           # make setup, make data, make deploy-4.1, make deploy-4.2, make test, make destroy
├── project_4_1_churn_mlops/
│   ├── data_generator.py              # Generates synthetic telecom churn CSV & uploads to S3 Gold layer
│   ├── train_and_deploy.py            # SageMaker XGBoost 1.7-1 training, endpoint deploy & DataCaptureConfig
│   ├── lambda_handler.py              # Real-time inference bridge (API Gateway proxy -> SageMaker Runtime)
│   └── monitor_baseline.py            # Configures SageMaker DefaultModelMonitor for feature drift
├── project_4_2_bedrock_rag/
│   ├── sample_docs/                   # Auto-generated enterprise policy PDFs & Markdown for ingestion
│   │   ├── remote_work_and_equipment_policy.md / .pdf
│   │   ├── travel_and_expense_policy.md / .pdf
│   │   ├── it_security_and_data_protection_policy.md / .pdf
│   │   └── employee_benefits_guide.md / .pdf
│   ├── kb_provisioner.py              # Boto3 script to sync S3 docs with Bedrock Knowledge Base
│   ├── rag_lambda_api.py              # Bedrock retrieve_and_generate handler with S3 source citations
│   └── app_ui.py                      # FastAPI + modern interactive web client for RAG & ML testing
├── scripts/
│   ├── test_endpoints.py              # Automated integration tests for Churn REST API & RAG queries
│   └── cleanup_resources.py           # Zero-billing teardown script (deletes endpoints, collections, buckets)
└── terraform/                         # Production Infrastructure as Code
    ├── main.tf                        # S3 buckets, Lambdas, and API Gateway HTTP API routes
    ├── iam.tf                         # IAM Roles and least-privilege policies
    ├── variables.tf                   # Deployment parameter definitions
    └── outputs.tf                     # Endpoint URLs, role ARNs, and resource IDs
```

---

## 2. Core Dependencies & IAM Permission Matrix

| Component / Role | AWS Service Target | Required IAM Managed Policies / Actions | Runtime / Version |
| :--- | :--- | :--- | :--- |
| **Local Agent Env** | AWS CLI / Boto3 SDK | `AdministratorAccess` (or scoped S3, SageMaker, Lambda, Bedrock) | Python 3.11+ / boto3>=1.34 |
| **SageMaker Role** | Amazon SageMaker | `AmazonSageMakerFullAccess`, `s3:GetObject`, `s3:PutObject` | XGBoost Container 1.7-1 |
| **Churn Lambda Role**| AWS Lambda + API GW | `AWSLambdaBasicExecutionRole`, `sagemaker:InvokeEndpoint` | Python 3.12 (x86_64) |
| **Bedrock KB Role** | Bedrock + OpenSearch | `bedrock:InvokeModel` (Titan V2 & Claude 3.5), `aoss:APIAccessAll` | Titan Embeddings V2 |
| **RAG API Role** | AWS Lambda / FastAPI | `bedrock:RetrieveAndGenerate`, `bedrock:Retrieve` | Claude 3.5 Sonnet v1:0 |

---

## 3. Architecture Diagrams

### Project 4.1: Automated Training, Real-Time Inference & Model Drift Capture
```
┌──────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Amazon S3 (Gold) │ ───> │ SageMaker Training Job  │ ───> │ Real-Time Endpoint      │
│ train.csv        │      │ XGBoost (ml.m5.large)   │      │ ml.t2.medium            │
│ baseline.csv     │      └─────────────────────────┘      │ 100% Data Capture       │
└──────────────────┘                                       └───────────┬─────────────┘
                                                                       │
┌──────────────────┐      ┌─────────────────────────┐                  │
│   API Gateway    │ ───> │       AWS Lambda        │ ─────────────────┘
│  POST /predict   │      │   JSON-to-CSV Proxy     │
└──────────────────┘      └─────────────────────────┘
```

### Project 4.2: Zero-Hallucination Enterprise RAG on Amazon Bedrock
```
┌──────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ S3 Document Store│ ───> │ Bedrock Knowledge Base  │ ───> │  OpenSearch Serverless  │
│ Enterprise MD/PDF│      │ 300-Token Chunking      │      │    HNSW Vector Index    │
└──────────────────┘      │ Titan Text Embeddings V2│      └───────────┬─────────────┘
                          └─────────────────────────┘                  │
                                                                       ▼
┌──────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│ Interactive UI / │ ───> │    FastAPI / Lambda     │ ───> │ RetrieveAndGenerate API │
│ Client Query     │      │   rag_lambda_api.py     │      │ Claude 3.5 Sonnet       │
│                  │ <─── │   S3 Grounded Citations │ <─── │ Zero-Hallucination      │
└──────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

---

## 4. Sequential Execution Plan & Acceptance Criteria

### Phase 1: Environment Initialization & AWS Verification
Initialize virtual environment and verify IAM caller identity:
```bash
make setup
# Or manually:
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
aws sts get-caller-identity
```
- **Acceptance Gate**: Exit code 0; valid Account ID and IAM ARN returned.

### Phase 2: Synthetic Telecom Dataset Generation
Generate synthetic telecom customer records (2,000 samples) and upload to S3:
```bash
make data
# Or:
python project_4_1_churn_mlops/data_generator.py --bucket company-mlops-churn-data
```
- **Format**: Target column (`churn`: 0 or 1) in column 1, no header row, followed by `[tenure, charges, calls, contract]`.
- **Acceptance Gate**: S3 bucket contains `data/train.csv` (2,000 rows, ~38 KB) and `data/baseline.csv`.

### Phase 3: SageMaker XGBoost 1.7-1 Training & Deployment
Train model on `ml.m5.large` and deploy real-time endpoint with 100% DataCaptureConfig:
```bash
make deploy-4.1
# Or:
python project_4_1_churn_mlops/train_and_deploy.py
```
- **Acceptance Gate**: Endpoint `churn-xgb-prod` status is `InService`; S3 model artifact `models/.../model.tar.gz` exists.

### Phase 4: Lambda Real-Time Inference Bridge
Package and test API Gateway POST `/predict` proxy:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"tenure": 3, "charges": 110.0, "calls": 6, "contract": 0}'
```
- **Acceptance Gate**: HTTP 200 JSON response containing `churn_probability` in range `[0.0, 1.0]` and `risk_tier` (`HIGH` or `LOW`).

### Phase 5: Bedrock Knowledge Base Ingestion Sync
Upload enterprise policy documents to S3 and trigger Bedrock ingestion sync:
```bash
make deploy-4.2
# Or:
python project_4_2_bedrock_rag/kb_provisioner.py
```
- **Acceptance Gate**: Ingestion job reaches `COMPLETE` state with 0 failed documents.

### Phase 6: Automated End-to-End Verification
Execute the automated test suite covering both pipelines:
```bash
make test
# Or:
pytest scripts/test_endpoints.py -v
```
- **Acceptance Gate**: All assertions pass; RAG response includes valid `s3://` source citation URIs.

---

## 5. Local Interactive Testing Dashboard

Launch the unified FastAPI client for interactive RAG and Churn inference testing:
```bash
make ui
# Access at http://localhost:8000
```
- **Interactive RAG**: Query company policy questions (e.g., home office equipment allowance) with grounded S3 citation tags.
- **Interactive Churn Predictor**: Adjust customer tenure, monthly charges, support calls, and contract type to view real-time risk tiers.

---

## 6. Critical Billing Safeguard (FinOps Teardown)

> [!CAUTION]
> **FINOPS RULE FOR USERS AND AGENTS**
> Amazon SageMaker Real-Time Endpoints (`ml.t2.medium`) and Amazon OpenSearch Serverless vector collections incur continuous hourly charges even when idle. As soon as end-to-end verification tests pass, execute the teardown utility to guarantee zero residual AWS billing:

```bash
make destroy
# Or directly:
python scripts/cleanup_resources.py --endpoint-name churn-xgb-prod
```
The teardown script automatically deletes:
1. SageMaker real-time endpoint (`churn-xgb-prod`) and its EndpointConfig.
2. SageMaker Model Monitor schedules (`churn-xgb-prod-drift-monitor`).
3. Amazon OpenSearch Serverless collections.
4. Purges and closes active S3 staging resources.
