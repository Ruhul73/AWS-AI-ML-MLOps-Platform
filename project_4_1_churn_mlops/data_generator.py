#!/usr/bin/env python3
"""
Step 4.1.A: Synthetic Telecom Dataset Generator (data_generator.py)
Generates synthetic telecom churn CSV & uploads to S3 Gold layer.
SageMaker XGBoost requires CSV inputs with the target column (churn: 0 or 1)
in the first position and no header row.
"""

import os
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
import pandas as pd
import numpy as np


def generate_churn_dataset(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic telecom customer churn dataset."""
    np.random.seed(seed)
    tenure = np.random.randint(1, 72, n)
    monthly_charges = np.round(np.random.uniform(20.0, 120.0, n), 2)
    support_calls = np.random.poisson(1.8, n)
    contract_type = np.random.choice([0, 1, 2], n, p=[0.5, 0.3, 0.2])  # 0:Month, 1:1Yr, 2:2Yr

    # Ground-truth logit function for realistic signal
    logit = -1.5 - 0.04 * tenure + 0.02 * monthly_charges + 0.45 * support_calls - 0.8 * contract_type
    probs = 1 / (1 + np.exp(-logit))
    churn = (probs > 0.5).astype(int)

    df = pd.DataFrame({
        'churn': churn,
        'tenure': tenure,
        'charges': monthly_charges,
        'calls': support_calls,
        'contract': contract_type
    })
    return df


def ensure_bucket_exists(s3_client, bucket_name: str, region: str = "us-east-1"):
    """Verify S3 bucket existence or create it if absent."""
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("404", "NoSuchBucket"):
            print(f"Bucket {bucket_name} does not exist. Creating bucket in {region}...")
            if region == "us-east-1":
                s3_client.create_bucket(Bucket=bucket_name)
            else:
                s3_client.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={"LocationConstraint": region}
                )
            print(f"Bucket {bucket_name} created successfully.")
        else:
            raise


def generate_and_upload(bucket_name: str, region: str = "us-east-1") -> pd.DataFrame:
    """
    Generates synthetic telecom churn CSV and uploads train.csv (no header, churn first)
    and baseline.csv (features with header) to S3.
    """
    df = generate_churn_dataset(n=2000, seed=42)

    # SageMaker XGBoost format: Target in first column, no header row, no index
    os.makedirs("data", exist_ok=True)
    local_train_path = "train.csv"
    data_train_path = os.path.join("data", "train.csv")
    df.to_csv(local_train_path, index=False, header=False)
    df.to_csv(data_train_path, index=False, header=False)

    # Baseline dataset for SageMaker Model Monitor (features only, with headers)
    baseline_df = df[['tenure', 'charges', 'calls', 'contract']]
    baseline_path = os.path.join("data", "baseline.csv")
    baseline_df.to_csv(baseline_path, index=False, header=True)

    print(f"Locally generated 2,000 records at {local_train_path} and {baseline_path}")

    # Upload to S3 Gold layer
    try:
        s3 = boto3.client("s3", region_name=region)
        ensure_bucket_exists(s3, bucket_name, region)
        s3.upload_file(local_train_path, bucket_name, "data/train.csv")
        s3.upload_file(baseline_path, bucket_name, "data/baseline.csv")
        print(f"Uploaded 2,000 records to s3://{bucket_name}/data/train.csv")
        print(f"Uploaded baseline records to s3://{bucket_name}/data/baseline.csv")
    except (NoCredentialsError, ClientError) as e:
        print(f"[Notice] AWS S3 upload skipped or credentials not configured: {e}")
        print("Generated files preserved locally in ./data/ for offline/mock pipelines.")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Telecom Churn Data Generator")
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("BUCKET_PREFIX", "company-mlops-churn-data"),
        help="Target S3 Bucket Name"
    )
    parser.add_argument(
        "--region",
        type=str,
        default=os.environ.get("AWS_REGION", "us-east-1"),
        help="AWS Region"
    )
    args = parser.parse_args()
    generate_and_upload(bucket_name=args.bucket, region=args.region)
