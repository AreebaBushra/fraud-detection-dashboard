# Credit Card Fraud Detection Dashboard

A fraud monitoring dashboard powered by an XGBoost model trained on the
[mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
dataset (284,807 anonymized European card transactions, 492 confirmed fraud
cases), with per-transaction explanations generated using SHAP.

## Overview

Features `V1`-`V28` are PCA-anonymized by the dataset provider to protect
cardholder privacy, so live transaction input isn't possible with this data.
The app is built as an internal fraud monitoring dashboard instead —
reviewing a stream of test-set transactions with risk scores and
explanations, similar to how fraud analysts monitor transactions in
practice.

## Model

- **Algorithm:** XGBoost Classifier
- **Class imbalance handling:** `scale_pos_weight` tuned to the ~577:1
  normal-to-fraud ratio in the training data
- **Performance on held-out test set:**
  - ROC-AUC: 0.977
  - Precision (fraud class): 0.844
  - Recall (fraud class): 0.827
- **Explainability:** SHAP `TreeExplainer`, top-5 contributing features
  precomputed per transaction

## Reading the SHAP plots

- **Bar chart** (`plot_type='bar'`): ranked feature importance — longer bar
  = more influence on predictions overall.
- **Dot plot** (default `summary_plot`): same ranking, plus direction and
  value. Right side = pushes toward fraud, left side = pushes toward
  normal. Red = high feature value, blue = low.

## Project structure

```
fraud_dashboard/
├── app.py                  # FastAPI backend
├── requirements.txt
├── Procfile                 # Render start command
├── data/
│   └── transactions.json    # Precomputed sample transactions + SHAP explanations
├── model/
│   ├── xgb_fraud_model.pkl
│   ├── time_scaler.pkl
│   └── amount_scaler.pkl
├── static/
│   └── index.html            # Dashboard frontend (vanilla HTML/CSS/JS)
└── notebooks/
    ├── creditcardfrauddetection.ipynb   # EDA + model training
    └── shap_explainability.ipynb         # SHAP analysis
```

## Running locally

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```





