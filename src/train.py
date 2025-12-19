from pathlib import Path
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from pipeline import classifier_pipeline, regression_pipeline


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "Loan_dataset.xlsx"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

ARTIFACTS_DIR.mkdir(exist_ok=True)


df = pd.read_excel(DATA_PATH)
df.drop(columns=["ID", "cpf"], inplace=True)


X = df.drop(columns=["approved", "interest_rate"])
y = df["approved"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0
)

clf = classifier_pipeline()
clf.fit(X_train, y_train)

joblib.dump(clf, ARTIFACTS_DIR / "model_classifier.pkl")
joblib.dump((X_test, y_test), ARTIFACTS_DIR / "test_classifier.pkl")


approved_df = df[df["approved"] == 1]

X_reg = approved_df.drop(columns=["approved", "interest_rate"])
y_reg = approved_df["interest_rate"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=0
)

reg = regression_pipeline()
reg.fit(X_train_reg, y_train_reg)

joblib.dump(reg, ARTIFACTS_DIR / "model_regression.pkl")
joblib.dump(
    (X_test_reg, y_test_reg),
    ARTIFACTS_DIR / "test_regression.pkl",
)


# ===============================
# 🔒 CONGELAR VERDADE DO MODELO
# ===============================
FEATURES_REG = X_train_reg.columns.tolist()
DTYPES_REG = X_train_reg.dtypes.astype(str).to_dict()

joblib.dump(FEATURES_REG, ARTIFACTS_DIR / "regression_features.pkl")
joblib.dump(DTYPES_REG, ARTIFACTS_DIR / "regression_dtypes.pkl")


# ===============================
# Logs
# ===============================
print("✅ Treino finalizado e artefatos salvos em /artifacts")
