variable "aws_region" {
  description = "Target AWS Region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project identifier prefix"
  type        = string
  default     = "aws-ai-mlops-suite"
}

variable "bucket_name" {
  description = "S3 Gold storage bucket for MLOps datasets, models, and Bedrock docs"
  type        = string
  default     = "company-mlops-churn-data"
}

variable "sagemaker_endpoint_name" {
  description = "SageMaker Real-Time Endpoint Name"
  type        = string
  default     = "churn-xgb-prod"
}

variable "bedrock_model_arn" {
  description = "Bedrock Foundation Model ARN for LLM synthesis"
  type        = string
  default     = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20240620-v1:0"
}
