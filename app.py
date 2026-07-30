"""
Credit Card Fraud Detection Dashboard - FastAPI Backend

Serves a precomputed feed of transactions (from the test set) with XGBoost
fraud risk scores and SHAP-based explanations, styled as a bank fraud
monitoring dashboard.

Note on design: the source dataset (Kaggle mlg-ulb/creditcardfraud) uses
PCA-anonymized features (V1-V28), so there is no way to compute these from
a live user-entered transaction (the PCA transform was never released).
That's why this app plays back real test-set transactions rather than
accepting arbitrary user input - this mirrors how real fraud systems work
anyway (banks monitor a stream of transactions, customers don't self-check).
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "transactions.json"

app = FastAPI(title="Fraud Detection Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load precomputed transactions once at startup
with open(DATA_PATH) as f:
    TRANSACTIONS = json.load(f)


@app.get("/api/transactions")
def get_transactions(limit: int = 100, risk: str = "all"):
    """
    Return a list of transactions with risk scores.
    risk: 'all' | 'high' (score >= 0.5) | 'low' (score < 0.5)
    """
    data = TRANSACTIONS
    if risk == "high":
        data = [t for t in data if t["risk_score"] >= 0.5]
    elif risk == "low":
        data = [t for t in data if t["risk_score"] < 0.5]

    return {
        "count": len(data[:limit]),
        "total_available": len(data),
        "transactions": data[:limit],
    }


@app.get("/api/transactions/{transaction_id}")
def get_transaction(transaction_id: int):
    """Return full detail + SHAP explanation for one transaction."""
    for t in TRANSACTIONS:
        if t["id"] == transaction_id:
            return t
    raise HTTPException(status_code=404, detail="Transaction not found")


@app.get("/api/stats")
def get_stats():
    """Summary stats for the dashboard header."""
    total = len(TRANSACTIONS)
    flagged = sum(1 for t in TRANSACTIONS if t["predicted_fraud"] == 1)
    actual_fraud = sum(1 for t in TRANSACTIONS if t["true_label"] == 1)
    avg_risk = sum(t["risk_score"] for t in TRANSACTIONS) / total if total else 0

    return {
        "total_transactions": total,
        "flagged_as_fraud": flagged,
        "actual_fraud_in_sample": actual_fraud,
        "average_risk_score": round(avg_risk, 4),
    }


@app.get("/health")
def health():
    return {"status": "ok"}


# Serve the frontend
app.mount("/", StaticFiles(directory=BASE_DIR / "static", html=True), name="static")
