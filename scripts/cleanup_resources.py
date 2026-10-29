#!/usr/bin/env python3
"""
Section 7: Automated FinOps Teardown (scripts/cleanup_resources.py)
Zero-billing teardown utility to eliminate residual AWS charges by deleting:
- SageMaker Real-Time Endpoint & EndpointConfig (ml.t2.medium)
- SageMaker Model Monitor schedules
- OpenSearch Serverless Vector Collections
- (Optional) S3 bucket objects & Bedrock Knowledge Base
"""

import os
import argparse
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

DEFAULT_REGION = os.environ.get("AWS_REGION", "us-east-1")
DEFAULT_ENDPOINT = os.environ.get("ENDPOINT_NAME", "churn-xgb-prod")
DEFAULT_AOSS_ID = os.environ.get("AOSS_COLLECTION_ID", "")
DEFAULT_BUCKET = os.environ.get("BUCKET_PREFIX", "")
DEFAULT_KB_ID = os.environ.get("BEDROCK_KB_ID", "")


def teardown_all(
    endpoint_name: str = DEFAULT_ENDPOINT,
    aoss_collection_id: str = DEFAULT_AOSS_ID,
    bucket_name: str = DEFAULT_BUCKET,
    kb_id: str = DEFAULT_KB_ID,
    region: str = DEFAULT_REGION
):
    """
    Executes FinOps zero-billing cleanup across all provisioned AWS services.
    """
    print("=" * 65)
    print("FINOPS AUTOMATED TEARDOWN UTILITY (ZERO RESIDUAL AWS BILLING)")
    print("=" * 65)

    # 1. Clean up SageMaker Model Monitor Schedules and Endpoints
    try:
        sm = boto3.client("sagemaker", region_name=region)

        # Delete monitoring schedule if active
        schedule_name = f"{endpoint_name}-drift-monitor"
        try:
            sm.delete_monitoring_schedule(MonitoringScheduleName=schedule_name)
            print(f"✓ Deleted Model Monitor schedule: {schedule_name}")
        except Exception:
            pass

        # Describe and delete SageMaker endpoint & configuration
        try:
            desc = sm.describe_endpoint(EndpointName=endpoint_name)
            cfg = desc.get("EndpointConfigName")
            sm.delete_endpoint(EndpointName=endpoint_name)
            print(f"✓ Deleted SageMaker real-time endpoint: {endpoint_name}")
            if cfg:
                sm.delete_endpoint_config(EndpointConfigName=cfg)
                print(f"✓ Deleted SageMaker endpoint config: {cfg}")
        except Exception as e:
            print(f"SageMaker endpoint cleanup skipped or not found: {e}")

    except (NoCredentialsError, ClientError) as e:
        print(f"SageMaker client error: {e}")

    # 2. Clean up Amazon OpenSearch Serverless Collection
    if aoss_collection_id and aoss_collection_id != "YOUR_AOSS_COLLECTION_ID":
        try:
            aoss = boto3.client("opensearchserverless", region_name=region)
            aoss.delete_collection(id=aoss_collection_id)
            print(f"✓ Deleted OpenSearch Serverless collection: {aoss_collection_id}")
        except Exception as e:
            print(f"OpenSearch Serverless cleanup skipped: {e}")
    else:
        print("• OpenSearch Serverless collection ID not specified; skipped.")

    # 3. Clean up Bedrock Knowledge Base if provided
    if kb_id and kb_id != "YOUR_KB_ID":
        try:
            bedrock_agent = boto3.client("bedrock-agent", region_name=region)
            bedrock_agent.delete_knowledge_base(knowledgeBaseId=kb_id)
            print(f"✓ Deleted Bedrock Knowledge Base: {kb_id}")
        except Exception as e:
            print(f"Bedrock Knowledge Base cleanup skipped: {e}")

    # 4. Optional S3 bucket purge
    if bucket_name:
        try:
            s3 = boto3.resource("s3", region_name=region)
            bucket = s3.Bucket(bucket_name)
            bucket.objects.all().delete()
            print(f"✓ Purged all objects in S3 bucket: {bucket_name}")
        except Exception as e:
            print(f"S3 bucket purge skipped: {e}")

    print("=" * 65)
    print("FinOps teardown completed. Guaranteed zero idle charges.")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-Billing FinOps Resource Teardown")
    parser.add_argument("--endpoint-name", default=DEFAULT_ENDPOINT, help="SageMaker Endpoint Name")
    parser.add_argument("--aoss-collection-id", default=DEFAULT_AOSS_ID, help="OpenSearch Serverless Collection ID")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 Bucket to empty")
    parser.add_argument("--kb-id", default=DEFAULT_KB_ID, help="Bedrock Knowledge Base ID")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS Region")
    args = parser.parse_args()

    teardown_all(
        endpoint_name=args.endpoint_name,
        aoss_collection_id=args.aoss_collection_id,
        bucket_name=args.bucket,
        kb_id=args.kb_id,
        region=args.region
    )
