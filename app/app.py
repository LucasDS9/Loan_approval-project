import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import numpy as np


st.set_page_config(
    page_title="Análise de Crédito",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Simulador Completo de Empréstimo")
st.write("Preencha **todas** as informações abaixo para testar a precisão do modelo.")


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

classifier = joblib.load(os.path.join(ARTIFACTS_DIR, "model_classifier.pkl"))
regressor = joblib.load(os.path.join(ARTIFACTS_DIR, "model_regression.pkl"))


features = joblib.load(os.path.join(ARTIFACTS_DIR, "regression_features.pkl"))
dtypes = joblib.load(os.path.join(ARTIFACTS_DIR, "regression_dtypes.pkl"))


X_test, y_test = joblib.load(os.path.join(ARTIFACTS_DIR, "test_classifier.pkl"))
X_test_reg, y_test_reg = joblib.load(os.path.join(ARTIFACTS_DIR, "test_regression.pkl"))


st.subheader("📋 Dados Pessoais")

age = st.number_input("Idade", 18, 100, 30)
gender = st.selectbox("Gênero", options=["M", "F"])
employment_status = st.selectbox("Status de Emprego", options=["employed", "unemployed", "autonomous"])

st.subheader("💰 Informações Financeiras")
salary = st.number_input("Salário mensal (R$)", min_value=1_000, max_value=100_000, value=5_000, step=500)
credit_score = st.slider("Credit Score", 300, 1000, 650)
previous_delinquencies = st.number_input("Delinquências prévias", 0, 20, 0)
existing_loans_count = st.number_input("Quantidade de empréstimos ativos", 0, 10, 0)
savings_account = st.selectbox("Possui conta poupança?", options=["yes", "no"])

st.subheader("📊 Informações do Empréstimo")
loan_amount = st.number_input(
    "Valor do empréstimo (R$)",
    min_value=1_000,
    max_value=1_000_000,
    value=10_000,
    step=1_000
)
loan_term_months = st.slider("Prazo do empréstimo (meses)", 1, 60, 36)

st.subheader("📉 Dívida")
monthly_debt = st.number_input(
    "Gasto mensal com dívidas (R$)",
    min_value=0.0,
    value=1_000.0
)

debt_to_income = monthly_debt / salary
st.write(f"**Debt-to-Income (DTI):** {debt_to_income:.3f}")


if st.button("🔍 Avaliar Empréstimo"):

    input_data = pd.DataFrame([{
        "age": age,
        "salary": salary,
        "credit_score": credit_score,
        "Previous_Delinquencies": previous_delinquencies,
        "loan_amount": loan_amount,
        "Loan_Term_Months": loan_term_months,
        "Existing_Loans_Count": existing_loans_count,
        "debt_to_income": debt_to_income,
        "employment_status": employment_status,
        "gender": gender,
        "Savings_Account": savings_account
    }])

    # 🔒 PASSO 2 — força ordem e tipos corretos
    input_data = input_data.reindex(columns=features)

    for col, dtype in dtypes.items():
        input_data[col] = input_data[col].astype(dtype)

    st.divider()
    st.write("📌 **Dados enviados ao modelo (após alinhamento):**")
    st.dataframe(input_data)


    approval = classifier.predict(input_data)[0]

    st.divider()

    if approval == 1:
        st.success("✅ Crédito aprovado")
        interest_rate = regressor.predict(input_data)[0]
        st.metric(label="📈 Taxa de Juros Estimada", value=f"{interest_rate:.2f}%")
    else:
        st.error("❌ Crédito não aprovado")
