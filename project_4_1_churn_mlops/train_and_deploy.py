#!/usr/bin/env python3
"""
Step 4.1.B: SageMaker Training, Deployment & Data Capture Configuration
Trains SageMaker XGBoost 1.7-1 model on customer churn dataset,
configures 100% DataCaptureConfig for Model Monitor, and deploys
to a real-time endpoint (ml.t2.medium).
"""

import os
os.environ["SAGEMAKER_SUPPRESS_V2_WARNING"] = "1"
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model_monitor import DataCaptureConfig


def run_pipeline(
    bucket: str,
    role_arn: str,
    endpoint_name: str = "churn-xgb-prod",
    train_instance_type: str = "ml.m5.large",
    deploy_instance_type: str = "ml.t2.medium",
    wait: bool = True
):
    """
    Executes end-to-end SageMaker training and real-time endpoint deployment with DataCapture.
    """
    print(f"Initializing SageMaker Session for bucket: {bucket}, endpoint: {endpoint_name}")
    sess = sagemaker.Session()
    region = sess.boto_region_name or os.environ.get("AWS_REGION", "us-east-1")

    # Retrieve official AWS XGBoost 1.7-1 container URI
    xgb_img = image_uris.retrieve("xgboost", region, "1.7-1")
    print(f"Retrieved XGBoost 1.7-1 image URI: {xgb_img}")

    # Configure SageMaker Estimator
    est = Estimator(
        image_uri=xgb_img,
        role=role_arn,
        instance_count=1,
        instance_type=train_instance_type,
        output_path=f"s3://{bucket}/models/",
        sagemaker_session=sess
    )

    # Set Hyperparameters according to specification
    est.set_hyperparameters(
        objective="binary:logistic",
        num_round=80,
        max_depth=5,
        eta=0.2,
        subsample=0.8
    )

    train_s3_uri = f"s3://{bucket}/data/train.csv"
    print(f"Starting SageMaker Training Job using data from: {train_s3_uri}")
    est.fit({"train": TrainingInput(train_s3_uri, content_type="text/csv")}, wait=wait)

    # Configure Data Capture for SageMaker Model Monitor (100% sampling)
    capture_destination = f"s3://{bucket}/monitor/capture"
    print(f"Configuring DataCaptureConfig pointing to: {capture_destination}")
    capture = DataCaptureConfig(
        enable_capture=True,
        sampling_percentage=100,
        destination_s3_uri=capture_destination
    )

    # Deploy real-time endpoint
    print(f"Deploying Real-Time Endpoint '{endpoint_name}' on instance '{deploy_instance_type}'...")
    predictor = est.deploy(
        initial_instance_count=1,
        instance_type=deploy_instance_type,
        endpoint_name=endpoint_name,
        data_capture_config=capture
    )
    print(f"Endpoint '{endpoint_name}' deployed successfully.")
    return predictor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SageMaker Churn Training and Deployment")
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("BUCKET_PREFIX", "company-mlops-churn-data"),
        help="S3 bucket name holding training data"
    )
    parser.add_argument(
        "--role",
        type=str,
        default=os.environ.get("SAGEMAKER_ROLE_ARN", "arn:aws:iam::123456789012:role/SageMakerExecutionRole"),
        help="SageMaker execution IAM role ARN"
    )
    parser.add_argument(
        "--endpoint-name",
        type=str,
        default=os.environ.get("ENDPOINT_NAME", "churn-xgb-prod"),
        help="Target endpoint name"
    )
    args = parser.parse_args()

    try:
        run_pipeline(bucket=args.bucket, role_arn=args.role, endpoint_name=args.endpoint_name)
    except (NoCredentialsError, ClientError) as e:
        print(f"[Notice] SageMaker job execution requires valid AWS credentials and IAM permissions: {e}")
