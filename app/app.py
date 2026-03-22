from pathlib import Path
from typing import Literal
import json
import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"

clf = joblib.load(ARTIFACTS_DIR / "model_classifier.pkl")
reg = joblib.load(ARTIFACTS_DIR / "model_regression.pkl")
features: list[str] = joblib.load(ARTIFACTS_DIR / "regression_features.pkl")
dtypes: dict = joblib.load(ARTIFACTS_DIR / "regression_dtypes.pkl")

GROQ_MODEL = "llama-3.3-70b-versatile"

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
    age: int = Field(..., ge=0, le=100)
    gender: Literal["M", "F"]
    employment_status: Literal["employed", "unemployed", "autonomous"]
    salary: float = Field(..., gt=0)
    credit_score: int = Field(..., ge=300, le=1000)
    previous_delinquencies: int = Field(..., ge=0)
    existing_loans_count: int = Field(..., ge=0)
    savings_account: Literal["yes", "no"]
    loan_amount: float = Field(..., gt=0)
    loan_term_months: int = Field(..., ge=1, le=60)
    monthly_debt: float = Field(..., ge=0)

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
    interest_rate: float | None = None
    total_amount: float | None = None
    monthly_payment: float | None = None
    message: str
    explanation: str


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


def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY não encontrada.")
    return Groq(api_key=api_key)


def _run_explanation(
    req: LoanRequest,
    approved: bool,
    interest_rate: float | None,
    total_amount: float | None,
    monthly_payment: float | None,
) -> str:
    employment_map = {"employed": "empregado", "unemployed": "desempregado", "autonomous": "autônomo"}
    gender_map = {"M": "Masculino", "F": "Feminino"}
    debt_to_income = round(req.monthly_debt / req.salary * 100, 1)
    result_label = "aprovado" if approved else "negado"

    financials = ""
    if approved and interest_rate is not None:
        financials = f"""
Detalhes financeiros do empréstimo:
- Taxa de juros total sobre o período: {interest_rate:.2f}%
- Valor solicitado: R$ {req.loan_amount:,.2f}
- Total a pagar (com juros): R$ {total_amount:,.2f}
- Prazo: {req.loan_term_months} meses
- Parcela mensal estimada: R$ {monthly_payment:,.2f}
"""

    prompt = f"""
Você é um especialista em crédito e finanças pessoais.

Resultado da análise:
- Crédito: {result_label}
{financials}
Dados do cliente:
- Idade: {req.age} anos
- Gênero: {gender_map[req.gender]}
- Situação de emprego: {employment_map[req.employment_status]}
- Salário mensal: R$ {req.salary:,.2f}
- Score de crédito: {req.credit_score}
- Inadimplências anteriores: {req.previous_delinquencies}
- Empréstimos ativos: {req.existing_loans_count}
- Conta poupança: {"Sim" if req.savings_account == "yes" else "Não"}
- Valor solicitado: R$ {req.loan_amount:,.2f}
- Prazo: {req.loan_term_months} meses
- Dívida mensal atual: R$ {req.monthly_debt:,.2f}
- Comprometimento de renda: {debt_to_income}%

{"Explique ao cliente de forma simples e empática: por que o crédito foi aprovado, o que significa a taxa de juros (ela é sobre o valor total no período, não ao mês), mostre que o total a pagar é R$ " + f"{total_amount:,.2f}" + " e que a parcela mensal fica em R$ " + f"{monthly_payment:,.2f}" + ". Sugira como reduzir a taxa se possível." if approved else "Explique ao cliente de forma simples e empática por que o crédito foi negado, quais fatores pesaram mais, e sugira ações concretas para melhorar o perfil e tentar novamente."}

Responda em português claro, sem usar asteriscos, markdown ou formatação especial. Use texto corrido.
"""

    response = _get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "Responda em português claro, sem markdown, sem asteriscos, sem bullets. Use apenas texto corrido."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )

    return response.choices[0].message.content.strip()


@app.get("/healthcheck", tags=["Infra"])
def healthcheck():
    return {"status": "ok"}


@app.post("/predict", response_model=LoanResponse, tags=["Prediction"])
def predict(request: LoanRequest):
    try:
        df = build_dataframe(request)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    approved: bool = bool(clf.predict(df)[0] == 1)
    interest_rate: float | None = None
    total_amount: float | None = None
    monthly_payment: float | None = None

    if approved:
        raw_rate = round(float(reg.predict(df)[0]), 4)
        interest_rate = round(raw_rate * 2, 4)  # multiplica por 2
        total_amount = round(request.loan_amount * (1 + interest_rate / 100), 2)
        monthly_payment = round(total_amount / request.loan_term_months, 2)

    try:
        explanation = _run_explanation(request, approved, interest_rate, total_amount, monthly_payment)
    except Exception:
        explanation = "Análise indisponível no momento."

    return LoanResponse(
        approved=approved,
        interest_rate=interest_rate,
        total_amount=total_amount,
        monthly_payment=monthly_payment,
        message=(
            f"Crédito aprovado. Taxa total: {interest_rate:.2f}% | Total a pagar: R$ {total_amount:,.2f} | Parcela mensal: R$ {monthly_payment:,.2f}"
            if approved
            else "Crédito não aprovado."
        ),
        explanation=explanation,
    )