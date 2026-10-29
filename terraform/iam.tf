# Section 2: Core Dependencies & IAM Permission Matrix

# 1. SageMaker Execution Role
resource "aws_iam_role" "sagemaker_role" {
  name = "${var.project_name}-sagemaker-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "sagemaker.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full_access" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

resource "aws_iam_policy" "sagemaker_s3_policy" {
  name = "${var.project_name}-sagemaker-s3-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::${var.bucket_name}",
        "arn:aws:s3:::${var.bucket_name}/*"
      ]
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_s3_attach" {
  role       = aws_iam_role.sagemaker_role.name
  policy_arn = aws_iam_policy.sagemaker_s3_policy.arn
}

# 2. Churn Lambda Inference Bridge Role
resource "aws_iam_role" "churn_lambda_role" {
  name = "${var.project_name}-churn-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "churn_lambda_basic" {
  role       = aws_iam_role.churn_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_policy" "churn_lambda_invoke_sagemaker" {
  name = "${var.project_name}-lambda-sagemaker-invoke"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect   = "Allow"
      Action   = "sagemaker:InvokeEndpoint"
      Resource = "arn:aws:sagemaker:${var.aws_region}:*:endpoint/${var.sagemaker_endpoint_name}"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "churn_lambda_sagemaker_attach" {
  role       = aws_iam_role.churn_lambda_role.name
  policy_arn = aws_iam_policy.churn_lambda_invoke_sagemaker.arn
}

# 3. Bedrock Knowledge Base Role
resource "aws_iam_role" "bedrock_kb_role" {
  name = "${var.project_name}-bedrock-kb-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "bedrock.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_policy" "bedrock_kb_policy" {
  name = "${var.project_name}-bedrock-kb-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = [
          "bedrock:InvokeModel"
        ]
        Resource = [
          "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0",
          var.bedrock_model_arn
        ]
      },
      {
        Effect   = "Allow"
        Action   = [
          "aoss:APIAccessAll"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:ListBucket"
        ]
        Resource = [
          "arn:aws:s3:::${var.bucket_name}",
          "arn:aws:s3:::${var.bucket_name}/*"
        ]
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "bedrock_kb_attach" {
  role       = aws_iam_role.bedrock_kb_role.name
  policy_arn = aws_iam_policy.bedrock_kb_policy.arn
}

# 4. RAG API Lambda Role
resource "aws_iam_role" "rag_lambda_role" {
  name = "${var.project_name}-rag-lambda-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "lambda.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "rag_lambda_basic" {
  role       = aws_iam_role.rag_lambda_role.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_policy" "rag_lambda_bedrock" {
  name = "${var.project_name}-rag-lambda-bedrock"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "bedrock:RetrieveAndGenerate",
        "bedrock:Retrieve"
      ]
      Resource = "*"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "rag_lambda_bedrock_attach" {
  role       = aws_iam_role.rag_lambda_role.name
  policy_arn = aws_iam_policy.rag_lambda_bedrock.arn
}
