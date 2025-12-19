# pipeline.py

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OrdinalEncoder


def build_preprocess(categorical, numeric_knn):
    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                ),
                categorical,
            ),
            ("median", SimpleImputer(strategy="median"), ["age"]),
            ("knn", KNNImputer(n_neighbors=5), numeric_knn),
        ],
        remainder="passthrough",
    )
    return preprocess


def classifier_pipeline():
    categorical = ["employment_status", "Savings_Account", "gender"]
    numeric_knn = ["credit_score", "Previous_Delinquencies"]

    preprocess = build_preprocess(categorical, numeric_knn)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "model",
                DecisionTreeClassifier(
                    criterion="gini",
                    splitter="best",
                    max_depth=5,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    random_state=0,
                ),
            ),
        ]
    )
    return pipe


def regression_pipeline():
    categorical = ["employment_status", "Savings_Account", "gender"]
    numeric_knn = ["credit_score", "Previous_Delinquencies"]

    preprocess = build_preprocess(categorical, numeric_knn)

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression()),
        ]
    )
    return pipe

print(1)