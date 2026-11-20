
from fastapi import FastAPI
from pydantic import BaseModel, field_validator
import pandas as pd
import numpy as np

from model_loader import (
    original_model,
    original_preprocessor,
    v2_model,
    v2_preprocessor
)

app = FastAPI(
    title="CampaignIQ API",
    description="AI-powered Bank Telemarketing Prediction API",
    version="2.0.0"
)



class Customer(BaseModel):
    # Required customer information
    age: int
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str

    # Optional campaign information
    contact: str | None = None
    month: str | None = None
    day_of_week: str | None = None
    campaign: int | None = None
    pdays: int | None = None
    previous: int | None = None
    poutcome: str | None = None

    # Optional economic indicators
    emp_var_rate: float | None = None
    cons_price_idx: float | None = None
    cons_conf_idx: float | None = None
    euribor3m: float | None = None
    nr_employed: float | None = None

    # Reject invalid customer age
    @field_validator("age")
    @classmethod
    def validate_age(cls, value):
        if value < 17 or value > 100:
            raise ValueError("Enter a valid customer age between 17 and 100.")
        return value

    # Reject empty or Swagger placeholder customer information
    @field_validator(
        "job", "marital", "education",
        "default", "housing", "loan"
    )
    @classmethod
    def validate_customer_fields(cls, value):
        if not value or value.strip().lower() in [
            "string", "none", "null", "undefined"
        ]:
            raise ValueError(
                "Valid customer information is required. "
                "Placeholder values are not accepted."
            )
        return value.strip()

    # Reject placeholder values in optional text fields too
    @field_validator(
        "contact", "month", "day_of_week", "poutcome"
    )
    @classmethod
    def validate_optional_text(cls, value):
        if value is not None and value.strip().lower() in [
            "string", "none", "null", "undefined"
        ]:
            raise ValueError(
                "Use null for missing information instead of placeholders."
            )
        return value.strip() if value is not None else None



@app.get("/")
def home():
    return {
        "message": "Welcome to CampaignIQ API",
        "status": "running",
        "models": [
            "Original XGBoost",
            "Missingness-Aware V2"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "original_model_loaded": original_model is not None,
        "original_preprocessor_loaded": original_preprocessor is not None,
        "v2_model_loaded": v2_model is not None,
        "v2_preprocessor_loaded": v2_preprocessor is not None
    }


def prepare_customer_data(customer):
    data = pd.DataFrame([customer.model_dump()])

    # Match training dataset column names
    data = data.rename(columns={
        "emp_var_rate": "emp.var.rate",
        "cons_price_idx": "cons.price.idx",
        "cons_conf_idx": "cons.conf.idx",
        "nr_employed": "nr.employed"
    })

    data = data.replace({None: np.nan})

    # Create the same engineered features used during training
    data["was_previously_contacted"] = np.where(
        data["pdays"].isna(),
        np.nan,
        (data["pdays"] != 999).astype(float)
    )

    data["had_previous_campaign"] = np.where(
        data["previous"].isna(),
        np.nan,
        (data["previous"] > 0).astype(float)
    )

    # Ensure exact training feature order
    data = data[FEATURES].copy()

    for column in NUMERIC_FEATURES:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    return data



def choose_model(data):
    missing_count = int(
        data[OPTIONAL_FEATURES].isna().sum().sum()
    )

    total_optional = len(OPTIONAL_FEATURES)

    missing_percentage = (
        missing_count / total_optional
    ) * 100

    if missing_count <= 3:
        # Original XGBoost: 0–3 missing fields
        model = original_model
        preprocessor = original_preprocessor
        model_name = "Original XGBoost"

        if missing_count == 0:
            scenario = "Complete"
            threshold = 0.73046
        elif missing_count <= 1:
            scenario = "10% missing"
            threshold = 0.55532
        else:
            scenario = "25% missing"
            threshold = 0.55385

    else:
        # V2: more than 3 missing fields
        model = v2_model
        preprocessor = v2_preprocessor
        model_name = "Missingness-Aware V2"

        if missing_count <= 7:
            scenario = "40% missing"
            threshold = 0.66003
        elif missing_count <= 11:
            scenario = "60% missing"
            threshold = 0.58101
        else:
            scenario = "100% missing"
            threshold = 0.49670

    return {
        "supported": True,
        "model": model,
        "preprocessor": preprocessor,
        "model_name": model_name,
        "scenario": scenario,
        "threshold": threshold,
        "missing_count": missing_count,
        "missing_percentage": round(missing_percentage, 2)
    }


@app.post("/predict")
def predict(customer: Customer):

    # Step 1: Prepare input
    data = prepare_customer_data(customer)

    # Step 2: Select model based on missing information
    selection = choose_model(data)

    # Do not silently apply an unvalidated routing rule
    if not selection["supported"]:
        return {
            "status": "manual_review_required",
            "message": (
                "This missing-information pattern was not "
                "directly evaluated during model validation."
            ),
            "missing_optional_fields": selection["missing_count"],
            "missing_percentage": selection["missing_percentage"],
            "prediction": None
        }

    model = selection["model"]
    preprocessor = selection["preprocessor"]

    # Step 3: Preprocess using the selected model's preprocessor
    processed_data = preprocessor.transform(data)

    # Step 4: Predict subscription probability
    probability = float(
        model.predict_proba(processed_data)[0][1]
    )

    # Step 5: Apply the selected validation threshold
    threshold = selection["threshold"]
    prediction = int(probability >= threshold)

    # Step 6: Return result
    return {
        "status": "success",
        "selected_model": selection["model_name"],
        "scenario": selection["scenario"],
        "missing_optional_fields": selection["missing_count"],
        "missing_percentage": selection["missing_percentage"],
        "prediction": prediction,
        "result": (
            "Likely to Subscribe"
            if prediction == 1
            else "Unlikely to Subscribe"
        ),
        "subscription_probability": round(probability, 4),
        "subscription_percentage": round(probability * 100, 2),
        "validation_threshold": threshold
    }

