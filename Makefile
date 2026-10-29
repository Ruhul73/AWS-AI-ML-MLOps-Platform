# AWS Machine Learning & MLOps Suite Makefile
SHELL := /bin/bash
VENV := .venv
PYTHON := $(VENV)/bin/python
PYTEST := $(VENV)/bin/pytest
UVICORN := $(VENV)/bin/uvicorn

.PHONY: help setup data deploy-4.1 deploy-4.2 test ui monitor destroy

help:
	@echo "Available commands:"
	@echo "  make setup       - Create virtual environment, install requirements, verify env"
	@echo "  make data        - Generate synthetic telecom churn CSV and upload to S3"
	@echo "  make deploy-4.1  - Train SageMaker XGBoost 1.7-1 and deploy real-time endpoint"
	@echo "  make deploy-4.2  - Upload docs and sync Bedrock Knowledge Base"
	@echo "  make test        - Run automated end-to-end integration tests via pytest"
	@echo "  make ui          - Launch FastAPI interactive RAG & Churn testing dashboard"
	@echo "  make monitor     - Configure SageMaker Model Monitor baseline and schedules"
	@echo "  make destroy     - Execute FinOps teardown to delete endpoints and eliminate billing"

setup:
	@echo "==> Setting up Python virtual environment..."
	@if [ ! -d "$(VENV)" ]; then \
		export PATH="$$HOME/.local/bin:$$PATH" && uv venv --python 3.11 $(VENV); \
	fi
	@echo "==> Installing production requirements..."
	@export PATH="$$HOME/.local/bin:$$PATH" && uv pip install --python $(PYTHON) -r requirements.txt
	@echo "==> Verifying environment..."
	@$(PYTHON) -c "import boto3, sagemaker, pandas, fastapi, pytest; print('✓ Core libraries imported successfully.')"
	@echo "==> Checking AWS Identity..."
	@aws sts get-caller-identity || echo "[Notice] AWS credentials not configured. Mock/offline fallback mode active."

data:
	@echo "==> Running Step 4.1.A: Synthetic Telecom Dataset Generator..."
	@PYTHONPATH=. $(PYTHON) project_4_1_churn_mlops/data_generator.py

deploy-4.1:
	@echo "==> Running Step 4.1.B: SageMaker XGBoost Training and Deployment..."
	@PYTHONPATH=. $(PYTHON) project_4_1_churn_mlops/train_and_deploy.py

deploy-4.2:
	@echo "==> Running Step 4.2.A: Bedrock Knowledge Base Ingestion Sync..."
	@PYTHONPATH=. $(PYTHON) project_4_2_bedrock_rag/kb_provisioner.py

test:
	@echo "==> Running Section 6 Automated Verification Suite..."
	@PYTHONPATH=. $(PYTEST) scripts/test_endpoints.py -v

ui:
	@echo "==> Launching Step 4.2.B FastAPI & UI Dashboard..."
	@PYTHONPATH=. $(PYTHON) project_4_2_bedrock_rag/app_ui.py

monitor:
	@echo "==> Configuring SageMaker Model Monitor baseline..."
	@PYTHONPATH=. $(PYTHON) project_4_1_churn_mlops/monitor_baseline.py --local

destroy:
	@echo "==> Running Section 7 FinOps Automated Teardown..."
	@PYTHONPATH=. $(PYTHON) scripts/cleanup_resources.py
