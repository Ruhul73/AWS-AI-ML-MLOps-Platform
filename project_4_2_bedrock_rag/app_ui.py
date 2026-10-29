"""
Step 4.2.B: FastAPI / Local Interactive Interface (app_ui.py)
Exposes the enterprise Bedrock RAG /ask API and provides a modern web interface
for interactive testing of both Bedrock RAG and Churn ML inference.
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from project_4_2_bedrock_rag.rag_lambda_api import query_enterprise_rag
from project_4_1_churn_mlops.lambda_handler import lambda_handler as churn_lambda_handler

app = FastAPI(
    title="Enterprise Bedrock RAG Service",
    description="Zero-Hallucination Enterprise Policy RAG powered by Amazon Bedrock & Claude 3.5 Sonnet",
    version="1.0.0"
)


class QueryRequest(BaseModel):
    question: str


class ChurnPredictionRequest(BaseModel):
    tenure: int = 12
    charges: float = 85.5
    calls: int = 4
    contract: int = 0  # 0: Month-to-Month, 1: 1 Year, 2: 2 Year


@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "Bedrock-RAG-Churn-Suite"}


@app.post("/ask")
def ask_question(req: QueryRequest):
    """
    Query enterprise policies using Amazon Bedrock RetrieveAndGenerate API.
    """
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    return query_enterprise_rag(req.question)


@app.post("/predict")
def predict_churn(req: ChurnPredictionRequest):
    """
    Proxy to the Churn Inference Lambda / SageMaker Endpoint.
    """
    import json
    features = [req.tenure, req.charges, req.calls, req.contract]
    event = {"body": json.dumps({"features": features})}
    result = churn_lambda_handler(event, None)
    return json.loads(result["body"])


@app.get("/", response_class=HTMLResponse)
def index_ui():
    """
    Interactive web UI for testing Bedrock RAG and SageMaker Churn Prediction.
    """
    return """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AWS AI/ML & MLOps Enterprise Suite</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-primary: #0a0e17;
            --bg-secondary: #111827;
            --bg-card: rgba(17, 24, 39, 0.7);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-blue: #3b82f6;
            --accent-cyan: #06b6d4;
            --accent-purple: #8b5cf6;
            --accent-emerald: #10b981;
            --accent-rose: #f43f5e;
            --text-primary: #f9fafb;
            --text-secondary: #9ca3af;
            --text-muted: #6b7280;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 2rem 1rem;
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(59, 130, 246, 0.08) 0%, transparent 40%),
                radial-gradient(circle at 90% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 40%);
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        .badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            background: rgba(59, 130, 246, 0.15);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #93c5fd;
            padding: 0.35rem 0.85rem;
            border-radius: 9999px;
            font-size: 0.8rem;
            font-weight: 500;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        h1 {
            font-size: 2.5rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            background: linear-gradient(135deg, #ffffff 30%, #94a3b8 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        .subtitle {
            color: var(--text-secondary);
            font-size: 1.05rem;
            max-width: 650px;
            margin: 0 auto;
        }

        .grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.5rem;
        }

        @media (max-width: 900px) {
            .grid { grid-template-columns: 1fr; }
        }

        .card {
            background: var(--bg-card);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.75rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3);
            display: flex;
            flex-direction: column;
        }

        .card-header {
            display: flex;
            align-items: center;
            gap: 0.75rem;
            margin-bottom: 1.25rem;
            padding-bottom: 1rem;
            border-bottom: 1px solid var(--border-color);
        }

        .card-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.25rem;
        }

        .rag-icon { background: rgba(139, 92, 246, 0.2); border: 1px solid rgba(139, 92, 246, 0.4); }
        .churn-icon { background: rgba(6, 182, 212, 0.2); border: 1px solid rgba(6, 182, 212, 0.4); }

        .card-title {
            font-size: 1.25rem;
            font-weight: 600;
        }

        .card-desc {
            font-size: 0.85rem;
            color: var(--text-secondary);
        }

        .form-group {
            margin-bottom: 1rem;
        }

        label {
            display: block;
            font-size: 0.85rem;
            font-weight: 500;
            color: var(--text-secondary);
            margin-bottom: 0.4rem;
        }

        input, select, textarea {
            width: 100%;
            padding: 0.75rem 1rem;
            background: rgba(15, 23, 42, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            color: var(--text-primary);
            font-family: inherit;
            font-size: 0.95rem;
            transition: all 0.2s ease;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: var(--accent-blue);
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }

        .btn {
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: white;
            border: none;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
            width: 100%;
            margin-top: 0.5rem;
        }

        .btn:hover {
            opacity: 0.95;
            transform: translateY(-1px);
        }

        .quick-queries {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
            margin: 0.75rem 0;
        }

        .pill-btn {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid var(--border-color);
            color: var(--text-secondary);
            padding: 0.35rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.78rem;
            cursor: pointer;
            transition: all 0.2s;
        }

        .pill-btn:hover {
            background: rgba(255, 255, 255, 0.1);
            color: var(--text-primary);
            border-color: rgba(255, 255, 255, 0.2);
        }

        .result-box {
            margin-top: 1.25rem;
            padding: 1rem;
            background: rgba(10, 14, 23, 0.8);
            border: 1px solid var(--border-color);
            border-radius: 10px;
            min-height: 120px;
            display: none;
        }

        .result-box.visible {
            display: block;
        }

        .citation-tag {
            display: inline-block;
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #6ee7b7;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            padding: 0.2rem 0.5rem;
            border-radius: 4px;
            margin-top: 0.5rem;
            word-break: break-all;
        }

        .risk-badge {
            display: inline-block;
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-weight: 700;
            font-size: 1rem;
            margin-top: 0.5rem;
        }

        .risk-high { background: rgba(244, 63, 94, 0.2); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.4); }
        .risk-low { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.4); }

        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="badge">Production Suite • Category 4 Specification</div>
            <h1>AWS AI/ML & MLOps Platform</h1>
            <p class="subtitle">SageMaker Customer Churn Real-Time Inference & Serverless Bedrock Enterprise RAG</p>
        </header>

        <div class="grid">
            <!-- Project 4.2: Bedrock Serverless RAG -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon rag-icon">⚡</div>
                    <div>
                        <div class="card-title">Project 4.2: Enterprise Bedrock RAG</div>
                        <div class="card-desc">Amazon Bedrock Claude 3.5 Sonnet + OpenSearch Serverless Vector Index</div>
                    </div>
                </div>

                <div class="form-group">
                    <label for="rag-question">Enterprise Policy Question</label>
                    <textarea id="rag-question" rows="3" placeholder="e.g. What is the company reimbursement limit for home office equipment?"></textarea>
                </div>

                <div class="quick-queries">
                    <button class="pill-btn" onclick="setQuestion('What is the company reimbursement limit for home office equipment?')">Home Office Allowance</button>
                    <button class="pill-btn" onclick="setQuestion('What are the domestic hotel and meal per diem limits?')">Travel & Meals</button>
                    <button class="pill-btn" onclick="setQuestion('What are the security guidelines for Generative AI and customer data?')">AI Security</button>
                </div>

                <button class="btn" id="ask-btn" onclick="executeRAG()">
                    <span>Query Knowledge Base</span>
                </button>

                <div class="result-box" id="rag-result">
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">Synthesized Response:</div>
                    <div id="rag-answer" style="font-size: 0.95rem; line-height: 1.5; color: #e2e8f0;"></div>
                    <div id="rag-citations" style="margin-top: 0.75rem;"></div>
                </div>
            </div>

            <!-- Project 4.1: Customer Churn MLOps -->
            <div class="card">
                <div class="card-header">
                    <div class="card-icon churn-icon">📊</div>
                    <div>
                        <div class="card-title">Project 4.1: Customer Churn Predictor</div>
                        <div class="card-desc">SageMaker XGBoost 1.7-1 Real-Time Endpoint Proxy (POST /predict)</div>
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label for="tenure">Tenure (Months)</label>
                        <input type="number" id="tenure" value="3" min="1" max="72">
                    </div>
                    <div class="form-group">
                        <label for="charges">Monthly Charges ($)</label>
                        <input type="number" id="charges" value="110.0" step="0.5">
                    </div>
                </div>

                <div class="form-row">
                    <div class="form-group">
                        <label for="calls">Support Calls</label>
                        <input type="number" id="calls" value="6" min="0" max="20">
                    </div>
                    <div class="form-group">
                        <label for="contract">Contract Type</label>
                        <select id="contract">
                            <option value="0" selected>Month-to-Month (High Risk)</option>
                            <option value="1">1 Year Contract</option>
                            <option value="2">2 Year Contract</option>
                        </select>
                    </div>
                </div>

                <button class="btn" style="background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue));" onclick="predictChurn()">
                    <span>Predict Customer Churn</span>
                </button>

                <div class="result-box" id="churn-result">
                    <div style="font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.5rem;">Inference Result:</div>
                    <div id="churn-badge"></div>
                    <div id="churn-prob" style="margin-top: 0.5rem; font-size: 0.9rem; color: var(--text-secondary);"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        function setQuestion(q) {
            document.getElementById('rag-question').value = q;
        }

        async function executeRAG() {
            const q = document.getElementById('rag-question').value.trim();
            if (!q) return alert("Please enter a question.");
            const btn = document.getElementById('ask-btn');
            const resBox = document.getElementById('rag-result');
            const ansDiv = document.getElementById('rag-answer');
            const citeDiv = document.getElementById('rag-citations');

            btn.disabled = true;
            btn.innerHTML = '<span>Retrieving & Synthesizing...</span>';

            try {
                const resp = await fetch('/ask', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({question: q})
                });
                const data = await resp.json();
                ansDiv.textContent = data.answer || "No response generated.";
                citeDiv.innerHTML = '<div style="font-size: 0.75rem; color: var(--text-muted); margin-bottom: 0.25rem;">Grounded S3 Source Citations:</div>' +
                    (data.citations || []).map(c => `<div class="citation-tag">${c}</div>`).join('');
                resBox.classList.add('visible');
            } catch (err) {
                ansDiv.textContent = "Error executing RAG: " + err.message;
                resBox.classList.add('visible');
            } finally {
                btn.disabled = false;
                btn.innerHTML = '<span>Query Knowledge Base</span>';
            }
        }

        async function predictChurn() {
            const tenure = parseInt(document.getElementById('tenure').value, 10);
            const charges = parseFloat(document.getElementById('charges').value);
            const calls = parseInt(document.getElementById('calls').value, 10);
            const contract = parseInt(document.getElementById('contract').value, 10);

            const resBox = document.getElementById('churn-result');
            const badgeDiv = document.getElementById('churn-badge');
            const probDiv = document.getElementById('churn-prob');

            try {
                const resp = await fetch('/predict', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({tenure, charges, calls, contract})
                });
                const data = await resp.json();
                const isHigh = data.risk_tier === "HIGH";
                badgeDiv.innerHTML = `<span class="risk-badge ${isHigh ? 'risk-high' : 'risk-low'}">Risk Tier: ${data.risk_tier}</span>`;
                probDiv.textContent = `Estimated Churn Probability: ${(data.churn_probability * 100).toFixed(2)}% (${data.churn_probability})`;
                resBox.classList.add('visible');
            } catch (err) {
                badgeDiv.innerHTML = `<span class="risk-badge risk-high">Error: ${err.message}</span>`;
                resBox.classList.add('visible');
            }
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Enterprise Bedrock RAG & Churn Service on port {port}...")
    uvicorn.run("project_4_2_bedrock_rag.app_ui:app", host="0.0.0.0", port=port, reload=True)
