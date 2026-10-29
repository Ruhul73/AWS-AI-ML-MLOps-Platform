output "s3_gold_bucket" {
  description = "S3 Gold storage bucket name"
  value       = aws_s3_bucket.gold_storage.id
}

output "api_gateway_invoke_url" {
  description = "Base API Gateway execution URL"
  value       = aws_apigatewayv2_stage.prod_stage.invoke_url
}

output "churn_predict_endpoint_url" {
  description = "Live REST URL for Customer Churn inference"
  value       = "${aws_apigatewayv2_stage.prod_stage.invoke_url}/predict"
}

output "bedrock_rag_ask_url" {
  description = "Live REST URL for Bedrock RAG queries"
  value       = "${aws_apigatewayv2_stage.prod_stage.invoke_url}/ask"
}

output "sagemaker_execution_role_arn" {
  description = "IAM Role ARN for SageMaker Estimator"
  value       = aws_iam_role.sagemaker_role.arn
}

output "bedrock_kb_role_arn" {
  description = "IAM Role ARN for Bedrock Knowledge Base"
  value       = aws_iam_role.bedrock_kb_role.arn
}
