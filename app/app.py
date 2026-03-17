from pathlib import Path
from typing import Literal
import json
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"


clf = joblib.load(ARTIFACTS_DIR / "model_classifier.pkl")
reg = joblib.load(ARTIFACTS_DIR / "model_regression.pkl")
features: list[str] = joblib.load(ARTIFACTS_DIR / "regression_features.pkl")
dtypes: dict = joblib.load(ARTIFACTS_DIR / "regression_dtypes.pkl")


app = FastAPI(
    title="Loan Approval API",
    description="Predicts loan approval and, when approved, estimates the interest rate.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="frontend")

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(FRONTEND_DIR / "index.html")


class LoanRequest(BaseModel):
    age: int = Field(..., ge=0, le=100, description="Applicant age (0–100)")
    gender: Literal["M", "F"] = Field(..., description="Gender: 'M' or 'F'")
    employment_status: Literal["employed", "unemployed", "autonomous"] = Field(
        ..., description="Employment status"
    )
    salary: float = Field(..., gt=0, description="Monthly salary in BRL")
    credit_score: int = Field(..., ge=300, le=1000, description="Credit score (300–1000)")
    previous_delinquencies: int = Field(..., ge=0, description="Number of previous delinquencies")
    existing_loans_count: int = Field(..., ge=0, description="Number of active loans")
    savings_account: Literal["yes", "no"] = Field(..., description="Has savings account")
    loan_amount: float = Field(..., gt=0, description="Requested loan amount in BRL")
    loan_term_months: int = Field(..., ge=1, le=60, description="Loan term in months (1–60)")
    monthly_debt: float = Field(..., ge=0, description="Monthly debt payment in BRL")

    @field_validator("monthly_debt")
    @classmethod
    def debt_must_not_exceed_salary(cls, v: float, info) -> float:
        salary = info.data.get("salary")
        if salary and v > salary:
            raise ValueError("monthly_debt cannot exceed salary")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 35,
                "gender": "M",
                "employment_status": "employed",
                "salary": 8000,
                "credit_score": 720,
                "previous_delinquencies": 0,
                "existing_loans_count": 1,
                "savings_account": "yes",
                "loan_amount": 25000,
                "loan_term_months": 36,
                "monthly_debt": 1200,
            }
        }
    }


class LoanResponse(BaseModel):
    approved: bool
    interest_rate: float | None = Field(
        None,
        description="Estimated annual interest rate (%). Only present when approved.",
    )
    message: str



def build_dataframe(req: LoanRequest) -> pd.DataFrame:
    debt_to_income = req.monthly_debt / req.salary

    raw = {
        "age": req.age,
        "salary": req.salary,
        "credit_score": req.credit_score,
        "Previous_Delinquencies": req.previous_delinquencies,
        "loan_amount": req.loan_amount,
        "Loan_Term_Months": req.loan_term_months,
        "Existing_Loans_Count": req.existing_loans_count,
        "debt_to_income": debt_to_income,
        "employment_status": req.employment_status,
        "gender": req.gender,
        "Savings_Account": req.savings_account,
    }

    df = pd.DataFrame([raw]).reindex(columns=features)
    for col, dtype in dtypes.items():
        if col in df.columns:
            df[col] = df[col].astype(dtype)
    return df



@app.get("/healthcheck", tags=["Infra"])
def healthcheck():
    """Returns service health status."""
    return {"status": "ok"}


@app.post("/predict", response_model=LoanResponse, tags=["Prediction"])
def predict(request: LoanRequest):
    """
    Evaluates a loan application.

    - Returns **approved: true** and an estimated **interest_rate** when the
      classifier approves the credit.
    - Returns **approved: false** with no interest rate when denied.
    """
    try:
        df = build_dataframe(request)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    approved: bool = bool(clf.predict(df)[0] == 1)

    if approved:
        interest_rate: float = round(float(reg.predict(df)[0]), 4)
        return LoanResponse(
            approved=True,
            interest_rate=interest_rate,
            message=f"Credit approved. Estimated interest rate: {interest_rate:.2f}%",
        )

    return LoanResponse(
        approved=False,
        interest_rate=None,
        message="Credit not approved.",
    )