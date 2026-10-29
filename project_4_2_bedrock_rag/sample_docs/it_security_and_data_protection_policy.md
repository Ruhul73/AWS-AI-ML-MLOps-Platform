# Enterprise Information Security & AI Usage Policy

## 1. Data Classification Levels
- **Public**: Marketing content, published blog posts, public documentation.
- **Internal**: Internal engineering memos, roadmap presentations, sprint backlogs.
- **Confidential**: Proprietary algorithms, customer behavioral churn records, unredacted telecommunications telemetry.
- **Restricted**: PII (Personally Identifiable Information), customer payment cards, healthcare records, AWS IAM root credentials.

## 2. Generative AI & Cloud Services Policy
Employees interacting with foundation models (LLMs) or retrieval systems must ensure:
- Zero customer PII or raw authentication keys are submitted to public chatbots.
- Enterprise-approved generative AI services (such as Amazon Bedrock Claude 3.5 Sonnet inside private VPC endpoints) are strictly required for corporate data synthesis.
- Vector embeddings generated via Amazon Titan Text Embeddings V2 must be stored within encrypted OpenSearch Serverless collections with strict AWS IAM and data access policies.

## 3. Device Security Standards
All corporate laptops must utilize FileVault/BitLocker full-disk encryption, active CrowdStrike Falcon EDR endpoint protection, and enforce 15-minute screen lockouts.
