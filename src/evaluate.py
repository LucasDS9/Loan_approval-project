from pathlib import Path
import json
import joblib
from sklearn.metrics import classification_report, r2_score, mean_absolute_error


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
METRICS_PATH = BASE_DIR / "artifacts" / "metrics.json"


def evaluate_classifier():
    clf = joblib.load(ARTIFACTS_DIR / "model_classifier.pkl")
    X_test, y_test = joblib.load(ARTIFACTS_DIR / "test_classifier.pkl")
    y_pred = clf.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    print(" CLASSIFICAÇÃO :")
    print(classification_report(y_test, y_pred))
    return report


def evaluate_regression():
    reg = joblib.load(ARTIFACTS_DIR / "model_regression.pkl")
    X_test_reg, y_test_reg = joblib.load(ARTIFACTS_DIR / "test_regression.pkl")
    y_pred_reg = reg.predict(X_test_reg)

    r2 = r2_score(y_test_reg, y_pred_reg)
    mae = mean_absolute_error(y_test_reg, y_pred_reg)

    print("\nREGRESSÃO : ")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")

    return {"r2": round(r2, 4), "mae": round(mae, 4)}


def save_metrics(clf_report: dict, reg_metrics: dict):
    metrics = {
        "classification": {
            "accuracy": round(clf_report.get("accuracy", 0), 4),
            "precision_macro": round(clf_report.get("macro avg", {}).get("precision", 0), 4),
            "recall_macro": round(clf_report.get("macro avg", {}).get("recall", 0), 4),
            "f1_macro": round(clf_report.get("macro avg", {}).get("f1-score", 0), 4),
        },
        "regression": reg_metrics,
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n✅ Metrics saved to {METRICS_PATH}")
    return metrics


if __name__ == "__main__":
    clf_report = evaluate_classifier()
    reg_metrics = evaluate_regression()
    save_metrics(clf_report, reg_metrics)