terraform {
  required_version = ">= 1.5.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.30"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# 1. S3 Gold Data Lake & Document Store Bucket
resource "aws_s3_bucket" "gold_storage" {
  bucket = var.bucket_name

  tags = {
    Project     = var.project_name
    Environment = "production"
    ManagedBy   = "Terraform"
  }
}

resource "aws_s3_bucket_versioning" "gold_versioning" {
  bucket = aws_s3_bucket.gold_storage.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "gold_encryption" {
  bucket = aws_s3_bucket.gold_storage.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# 2. Package Lambda Handlers
data "archive_file" "churn_lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/build/churn_lambda.zip"

  source {
    content  = file("${path.module}/../project_4_1_churn_mlops/lambda_handler.py")
    filename = "lambda_handler.py"
  }
}

data "archive_file" "rag_lambda_zip" {
  type        = "zip"
  output_path = "${path.module}/build/rag_lambda.zip"

  source {
    content  = file("${path.module}/../project_4_2_bedrock_rag/rag_lambda_api.py")
    filename = "rag_lambda_api.py"
  }
}

# 3. Churn Prediction Lambda Function
resource "aws_lambda_function" "churn_inference_lambda" {
  filename         = data.archive_file.churn_lambda_zip.output_path
  function_name    = "${var.project_name}-churn-infer"
  role             = aws_iam_role.churn_lambda_role.arn
  handler          = "lambda_handler.lambda_handler"
  runtime          = "python3.12"
  timeout          = 30
  memory_size      = 256
  source_code_hash = data.archive_file.churn_lambda_zip.output_base64sha256

  environment {
    variables = {
      ENDPOINT_NAME = var.sagemaker_endpoint_name
      AWS_REGION    = var.aws_region
      MOCK_AWS      = "false"
    }
  }
}

# 4. Bedrock RAG Lambda Function
resource "aws_lambda_function" "bedrock_rag_lambda" {
  filename         = data.archive_file.rag_lambda_zip.output_path
  function_name    = "${var.project_name}-rag-infer"
  role             = aws_iam_role.rag_lambda_role.arn
  handler          = "rag_lambda_api.lambda_handler"
  runtime          = "python3.12"
  timeout          = 60
  memory_size      = 512
  source_code_hash = data.archive_file.rag_lambda_zip.output_base64sha256

  environment {
    variables = {
      AWS_REGION        = var.aws_region
      BUCKET_PREFIX     = var.bucket_name
      BEDROCK_MODEL_ARN = var.bedrock_model_arn
      MOCK_AWS          = "false"
    }
  }
}

# 5. API Gateway HTTP API
resource "aws_apigatewayv2_api" "http_api" {
  name          = "${var.project_name}-http-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"]
    allow_methods = ["POST", "GET", "OPTIONS"]
    allow_headers = ["content-type", "authorization"]
  }
}

# Integrations
resource "aws_apigatewayv2_integration" "churn_integration" {
  api_id                 = aws_apigatewayv2_api.http_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.churn_inference_lambda.arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "churn_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /predict"
  target    = "integrations/${aws_apigatewayv2_integration.churn_integration.id}"
}

resource "aws_apigatewayv2_integration" "rag_integration" {
  api_id                 = aws_apigatewayv2_api.http_api.id
  integration_type       = "AWS_PROXY"
  integration_uri        = aws_lambda_function.bedrock_rag_lambda.arn
  payload_format_version = "2.0"
}

resource "aws_apigatewayv2_route" "rag_route" {
  api_id    = aws_apigatewayv2_api.http_api.id
  route_key = "POST /ask"
  target    = "integrations/${aws_apigatewayv2_integration.rag_integration.id}"
}

resource "aws_apigatewayv2_stage" "prod_stage" {
  api_id      = aws_apigatewayv2_api.http_api.id
  name        = "$default"
  auto_deploy = true
}

# Lambda Permissions for API Gateway
resource "aws_lambda_permission" "api_gw_churn" {
  statement_id  = "AllowAPIGatewayInvokeChurn"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.churn_inference_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}

resource "aws_lambda_permission" "api_gw_rag" {
  statement_id  = "AllowAPIGatewayInvokeRAG"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.bedrock_rag_lambda.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}
