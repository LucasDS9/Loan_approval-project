from pathlib import Path
import joblib
from sklearn.metrics import classification_report, r2_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"


clf = joblib.load(ARTIFACTS_DIR / "model_classifier.pkl")
X_test, y_test = joblib.load(ARTIFACTS_DIR / "test_classifier.pkl")

y_pred = clf.predict(X_test)

print(" CLASSIFICAÇÃO :")
print(classification_report(y_test, y_pred))


reg = joblib.load(ARTIFACTS_DIR / "model_regression.pkl")
X_test_reg, y_test_reg = joblib.load(ARTIFACTS_DIR / "test_regression.pkl")

y_pred_reg = reg.predict(X_test_reg)

print("\nREGRESSÃO : ")
print(f"R²: {r2_score(y_test_reg, y_pred_reg):.4f}")
print(f"MAE: {mean_absolute_error(y_test_reg, y_pred_reg):.4f}")
