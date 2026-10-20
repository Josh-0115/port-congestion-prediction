
from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load data
df = pd.read_csv("data/synthetic_ports.csv")

# Inputs available at prediction time
feature_columns = [
    "arrival_density",
    "port_capacity",
    "queue_length",
    "weather_severity",
    "historical_avg_delay",
    "hour_of_day",
    "season",
]

X = df[feature_columns]
y = df["congestion_label"]

numeric_features = [
    "arrival_density",
    "port_capacity",
    "queue_length",
    "weather_severity",
    "historical_avg_delay",
    "hour_of_day",
]

categorical_features = ["season"]

# Preprocess numerical and categorical inputs
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000),
        ),
    ]
)

# Hold out test data for an initial evaluation
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# Train and predict
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.3f}")

print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save the trained pipeline
model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(model, model_dir / "logistic_regression.joblib")

print("\nModel saved to models/logistic_regression.joblib")
