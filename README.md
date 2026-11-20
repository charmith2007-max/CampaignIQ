# Banking Term Deposit Subscription Prediction System

A production-grade machine learning architecture designed to predict whether bank clients will subscribe to a term deposit (`y = "yes" | "no"`). The system integrates dual dataset modeling, target leakage prevention, ensemble strategies (Soft Voting & Out-of-Fold Stacking), and an intelligent unified inference routing service.

---

## 1. Project Overview & Architecture

Banks conduct direct marketing campaigns (telemarketing calls) to secure term deposits. This project models customer subscription likelihood using two complementary datasets from the UCI Machine Learning Repository:

1. **`datasets/bank-full.csv`** (45,211 records, 17 attributes): Comprehensive customer demographic, personal financial, and campaign contact history.
2. **`datasets/bank-additional-full.csv`** (41,188 records, 21 attributes): Customer demographic, personal financial, campaign history, and 5 key macroeconomic indicators.

### Strict Data Integrity Notice
The two datasets represent distinct snapshots and experimental setups; they **must never be assumed to contain identical customer records** or be naively joined. They train two distinct baseline models that are strategically unified at the prediction interface.

---

## 2. Model Taxonomy

### Model A — Customer Information Model
- **Training Dataset**: `datasets/bank-full.csv`
- **Algorithm**: Random Forest Classifier (`RandomForestClassifier`)
- **Feature Space**: Customer demographic features (`age`, `job`, `marital`, `education`, `default`, `housing`, `loan`, `balance`) and campaign history (`contact`, `day`, `month`, `campaign`, `pdays`, `previous`, `poutcome`).
- **Leakage Prevention**: The `duration` attribute is **strictly omitted**. In production, call duration is unknown prior to initiating a call. Using `duration` introduces severe target leakage and unrealistic optimistic evaluation bias.

### Model B — Economic Information Model
- **Training Dataset**: `datasets/bank-additional-full.csv`
- **Algorithm**: Random Forest Classifier (`RandomForestClassifier`)
- **Feature Space**: Customer demographics, campaign context, and **all five macroeconomic indicators**:
  - `emp.var.rate`: Employment variation rate (quarterly indicator)
  - `cons.price.idx`: Consumer price index (monthly indicator)
  - `cons.conf.idx`: Consumer confidence index (monthly indicator)
  - `euribor3m`: Euribor 3-month rate (daily indicator)
  - `nr.employed`: Number of employees (quarterly indicator)
- **Leakage Prevention**: The `duration` attribute is **strictly omitted**.

### Model C — Soft Voting Ensemble
- Combines calibrated predicted probability distributions from Model A and Model B.
- Evaluates both baseline equal weights ($w_A = 0.5, w_B = 0.5$) and metric-optimized weights (via validation ROC-AUC maximization).
- Requires aligned evaluation samples to ensure fair probability aggregation.

### Model D — Stacking Ensemble
- Combines predictions using a trained meta-classifier (e.g., Logistic Regression).
- Employs **Out-Of-Fold (OOF)** predictions across $K$-fold stratified splits to construct the meta-features matrix, strictly eliminating data leakage.

---

## 3. Unified Prediction System

The production API provides a single entry point (`/predict`) that intelligently routes inference requests based on data availability:

```
                          ┌───────────────────────────┐
                          │ Incoming Prediction Payload│
                          └─────────────┬─────────────┘
                                        │
                         Customer Info (Mandatory)
                                        │
                      ┌─────────────────┴─────────────────┐
                      │                                   │
       Macroeconomic indicators            Macroeconomic indicators
          MISSING or INCOMPLETE              ALL 5 FULLY PROVIDED
                      │                                   │
                      ▼                                   ▼
        ┌───────────────────────────┐       ┌───────────────────────────┐
        │          Route 1          │       │          Route 2          │
        │   Customer-Only Model     │       │   Economic-Aware Model    │
        │        (Model A)          │       │    (Model B / Ensemble)   │
        └─────────────┬─────────────┘       └─────────────┬─────────────┘
                      │                                   │
                      └─────────────────┬─────────────────┘
                                        │
                                        ▼
                          ┌───────────────────────────┐
                          │ Unified Prediction Result │
                          │ (Prediction, Prob, Route) │
                          └───────────────────────────┘
```

### Critical Operating Principle
The system **never invents missing economic indicators** or silently substitutes arbitrary imputation values (e.g. mean or zero). If macroeconomic indicators are not completely provided, the system gracefully routes to the pure Customer Information Model (Model A).

---

## 4. Directory Structure

```
Banking-ML-Project/
│
├── datasets/                       # Original CSV datasets (read-only, uncommitted)
│   ├── bank-full.csv               # Model A dataset (45,211 rows)
│   └── bank-additional-full.csv    # Model B dataset with economic indicators (41,188 rows)
│
├── notebooks/                      # Exploratory and prototyping notebooks
│   ├── experiment_A.ipynb          # Model A EDA, feature engineering, and RF baseline
│   └── experiment_C.ipynb          # Model C/D ensemble experiments and OOF evaluation
│
├── src/                            # Production modular Python codebase
│   ├── __init__.py                 # Package declaration
│   ├── config.py                   # Centralized configuration, paths, and feature definitions
│   ├── data_loader.py              # Semicolon CSV ingestion, validation, and stratified splits
│   ├── preprocessing.py            # Pipelines, transformers, and duration leakage remover
│   ├── train_models.py             # Random Forest training and hyperparameter optimization
│   ├── ensemble.py                 # Soft Voting (Model C) and Stacking (Model D) logic
│   ├── evaluation.py               # Metrics (ROC-AUC, PR-AUC, F1), reports, and ROC/PR plots
│   ├── prediction.py               # Unified inference service and routing engine
│   └── model_registry.py           # In-memory artifact caching and model serialization
│
├── models/                         # Serialized model estimators (.joblib) [gitkept]
│   └── .gitkeep
│
├── preprocessors/                  # Serialized ColumnTransformer pipelines [gitkept]
│   └── .gitkeep
│
├── evaluation/                     # Generated charts, metrics, and comparison CSVs [gitkept]
│   └── .gitkeep
│
├── app/                            # Web and API service layer
│   ├── main.py                     # FastAPI application endpoints (/health, /predict)
│   ├── schemas.py                  # Pydantic validation contracts
│   └── frontend/                   # Web user interface assets [gitkept]
│       └── .gitkeep
│
├── tests/                          # Test suites (unit and integration tests) [gitkept]
│   └── .gitkeep
│
├── requirements.txt                # Python project dependencies
├── README.md                       # Comprehensive architecture documentation
├── .gitignore                      # Git exclusion rules
└── .env.example                    # Environment configuration template
```

---

## 5. Setup & Environment

### Installation
```bash
cd Banking-ML-Project
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### Verification
Run `pytest` to verify testing hooks (once implemented):
```bash
pytest tests/
```
