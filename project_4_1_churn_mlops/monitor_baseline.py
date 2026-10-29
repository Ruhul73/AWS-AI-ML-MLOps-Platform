#!/usr/bin/env python3
"""
SageMaker DefaultModelMonitor Baseline & Drift Monitoring Configuration
Configures SageMaker Model Monitor to establish baseline statistics and constraints
for feature drift detection on data captured from real-time endpoint churn-xgb-prod.
"""

import os
os.environ["SAGEMAKER_SUPPRESS_V2_WARNING"] = "1"
import argparse
import boto3
from botocore.exceptions import NoCredentialsError, ClientError
import sagemaker
from sagemaker.model_monitor import (
    DefaultModelMonitor,
    DatasetFormat,
    CronExpressionGenerator
)


def configure_model_monitor(
    bucket: str,
    role_arn: str,
    endpoint_name: str = "churn-xgb-prod",
    instance_type: str = "ml.m5.large"
):
    """
    Sets up baseline constraints and attaches an hourly monitoring schedule to the endpoint.
    """
    print(f"Configuring SageMaker DefaultModelMonitor for endpoint: {endpoint_name}")
    sess = sagemaker.Session()

    my_monitor = DefaultModelMonitor(
        role=role_arn,
        instance_count=1,
        instance_type=instance_type,
        volume_size_in_gb=20,
        max_runtime_in_seconds=3600,
        sagemaker_session=sess
    )

    baseline_data_uri = f"s3://{bucket}/data/baseline.csv"
    baseline_output_uri = f"s3://{bucket}/monitor/baselines"

    print(f"Submitting baseline suggestion job from: {baseline_data_uri}")
    print(f"Baseline statistics & constraints destination: {baseline_output_uri}")

    try:
        my_monitor.suggest_baseline(
            baseline_dataset=baseline_data_uri,
            dataset_format=DatasetFormat.csv(header=True),
            output_s3_uri=baseline_output_uri,
            wait=True
        )
        print("Baseline suggestion job completed successfully.")

        # Attach schedule to the real-time endpoint
        schedule_name = f"{endpoint_name}-drift-monitor"
        reports_destination = f"s3://{bucket}/monitor/reports"
        print(f"Creating hourly monitoring schedule '{schedule_name}' -> {reports_destination}")

        my_monitor.create_monitoring_schedule(
            monitor_schedule_name=schedule_name,
            endpoint_input=endpoint_name,
            output_s3_uri=reports_destination,
            schedule_cron_expression=CronExpressionGenerator.hourly(),
            enable_cloudwatch_metrics=True
        )
        print(f"Monitoring schedule '{schedule_name}' activated successfully.")
        return my_monitor

    except (NoCredentialsError, ClientError) as e:
        print(f"[Notice] Live SageMaker Model Monitor configuration skipped: {e}")
        return None


def generate_local_baseline_statistics(data_path: str = "data/baseline.csv"):
    """
    Generate local statistical summary and constraints for verification/offline tests.
    """
    import pandas as pd
    if not os.path.exists(data_path):
        print(f"Baseline file {data_path} not found locally. Run data_generator.py first.")
        return None

    df = pd.read_csv(data_path)
    summary = {
        "record_count": len(df),
        "features": list(df.columns),
        "statistics": df.describe().to_dict()
    }
    print(f"Local baseline statistics generated for {len(df)} records:")
    print(df.describe())
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Configure SageMaker Model Monitor Baseline")
    parser.add_argument(
        "--bucket",
        type=str,
        default=os.environ.get("BUCKET_PREFIX", "company-mlops-churn-data"),
        help="S3 bucket name"
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
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run local statistical baseline analysis"
    )
    args = parser.parse_args()

    if args.local:
        generate_local_baseline_statistics()
    else:
        configure_model_monitor(bucket=args.bucket, role_arn=args.role, endpoint_name=args.endpoint_name)
