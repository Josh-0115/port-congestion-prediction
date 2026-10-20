
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load data
df = pd.read_csv("data/synthetic_ports.csv")

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

# Reproduce the same train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

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

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

model.fit(X_train, y_train)

# Select one test example so its actual label is available for comparison
sample = X_test.iloc[[0]]
actual_label = y_test.iloc[0]

# Predict congestion probability
probability = model.predict_proba(sample)[0, 1]
predicted_label = int(model.predict(sample)[0])

print("\nPORT CONGESTION PREDICTION")
print("-" * 40)
print(f"Estimated congestion probability: {probability:.1%}")
print(
    "Predicted class:",
    "Congestion" if predicted_label == 1 else "No congestion",
)
print(
    "Actual simulated label:",
    "Congestion" if actual_label == 1 else "No congestion",
)

# Get the transformed features and model coefficients
preprocessor_fitted = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

transformed_sample = preprocessor_fitted.transform(sample)

if hasattr(transformed_sample, "toarray"):
    transformed_sample = transformed_sample.toarray()

feature_names = preprocessor_fitted.get_feature_names_out()
coefficients = classifier.coef_[0]

# Each contribution is measured in log-odds units
contributions = transformed_sample[0] * coefficients

explanation = pd.DataFrame(
    {
        "Feature": feature_names,
        "Contribution": contributions,
    }
)

explanation["Direction"] = explanation["Contribution"].apply(
    lambda value: (
        "Increases predicted risk"
        if value > 0
        else "Decreases predicted risk"
        if value < 0
        else "No contribution"
    )
)

explanation["Absolute contribution"] = (
    explanation["Contribution"].abs()
)

explanation = explanation.sort_values(
    "Absolute contribution",
    ascending=False,
)

print("\nTOP FACTORS IN THIS PREDICTION")
print("-" * 40)

for _, row in explanation.head(5).iterrows():
    feature = row["Feature"].replace("num__", "").replace("cat__", "")
    direction = row["Direction"]
    contribution = row["Contribution"]

    print(
        f"{feature}: {direction} "
        f"(contribution: {contribution:+.3f})"
    )

print(
    "\nNote: Contributions explain this model's calculation. "
    "They do not establish real-world causation."
)
