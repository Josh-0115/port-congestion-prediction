
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# 1. Load the synthetic dataset
df = pd.read_csv("data/synthetic_ports.csv")

# 2. Separate input features and target
X = df.drop(columns=["delay_hours", "congestion_label"])
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

# 3. Use the same split for every model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

# 4. Preprocess numeric and categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
    ]
)

# 5. Define three models
models = {
    "Dummy Baseline": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=5,
        random_state=42,
    ),
}

results = []

# 6. Train and evaluate each model
for name, model in models.items():
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1 Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC-AUC": roc_auc_score(y_test, probabilities),
    }

    results.append(metrics)

    print(f"\n{'=' * 45}")
    print(name)
    print("=" * 45)
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    print(f"Accuracy:  {metrics['Accuracy']:.3f}")
    print(f"Precision: {metrics['Precision']:.3f}")
    print(f"Recall:    {metrics['Recall']:.3f}")
    print(f"F1 Score:  {metrics['F1 Score']:.3f}")
    print(f"ROC-AUC:   {metrics['ROC-AUC']:.3f}")

# 7. Display a comparison table
comparison = pd.DataFrame(results).sort_values(
    by="ROC-AUC", ascending=False
)

print("\n\nMODEL COMPARISON")
print(comparison.to_string(index=False, float_format="%.3f"))

# Save results for the technical report
comparison.to_csv("data/model_comparison.csv", index=False)
print("\nSaved comparison to data/model_comparison.csv")
