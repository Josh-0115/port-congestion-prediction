
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Port Congestion Predictor",
    page_icon="🚢",
    layout="wide",
)

MODEL_PATH = Path("models/logistic_regression.joblib")

# -----------------------------
# Load the saved model
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

# -----------------------------
# Application header
# -----------------------------
st.title("🚢 Port Congestion & Shipment Delay Predictor")

st.write(
    "Estimate congestion risk from vessel traffic, port capacity, "
    "queue length, weather and historical delay patterns."
)

st.warning(
    "Prototype limitation: this model was trained on synthetic data. "
    "Its probabilities and risk categories are not validated against "
    "real port operations and should not be used for operational decisions."
)

if model is None:
    st.error(
        "Trained model not found. Run `python train_model.py` "
        "from the project root first."
    )
    st.stop()

st.divider()

# -----------------------------
# User input form
# -----------------------------
st.subheader("Enter port conditions")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        arrival_density = st.number_input(
            "Vessel arrival density",
            min_value=0.0,
            value=38.0,
            step=1.0,
            help="Number of vessel arrivals in the dataset's defined time window.",
        )

        port_capacity = st.number_input(
            "Port capacity",
            min_value=1.0,
            value=22.0,
            step=1.0,
            help="Synthetic capacity indicator; use the same scale as the training data.",
        )

        queue_length = st.number_input(
            "Queue length",
            min_value=0.0,
            value=19.0,
            step=1.0,
        )

        weather_severity = st.slider(
            "Weather severity",
            min_value=0,
            max_value=5,
            value=3,
            help="Synthetic severity scale from 0 (low) to 5 (high).",
        )

    with col2:
        historical_avg_delay = st.number_input(
            "Historical average delay (hours)",
            min_value=0.0,
            value=16.5,
            step=0.5,
        )

        hour_of_day = st.slider(
            "Hour of day",
            min_value=0,
            max_value=23,
            value=18,
        )

        season = st.selectbox(
            "Season",
            options=["spring", "summer", "autumn", "winter"],
            index=3,
        )

    submitted = st.form_submit_button(
        "Predict congestion risk",
        type="primary",
        use_container_width=True,
    )

# -----------------------------
# Prediction and explanation
# -----------------------------
if submitted:
    input_data = pd.DataFrame(
        [{
            "arrival_density": arrival_density,
            "port_capacity": port_capacity,
            "queue_length": queue_length,
            "weather_severity": weather_severity,
            "historical_avg_delay": historical_avg_delay,
            "hour_of_day": hour_of_day,
            "season": season,
        }]
    )

    probability = float(model.predict_proba(input_data)[0, 1])
    predicted_class = int(model.predict(input_data)[0])

    if probability >= 0.70:
        risk = "High"
    elif probability >= 0.40:
        risk = "Medium"
    else:
        risk = "Low"

    st.divider()
    st.subheader("Prediction results")

    result1, result2, result3 = st.columns(3)

    result1.metric(
        "Estimated congestion probability",
        f"{probability:.1%}",
    )

    result2.metric(
        "Predicted outcome",
        "Congestion" if predicted_class == 1 else "No congestion",
    )

    result3.metric("Risk category", risk)

    st.progress(probability)

    st.caption(
        "The risk bands (below 40%, 40–69.9%, and 70% or above) "
        "are illustrative thresholds, not calibrated operational standards."
    )


    # -----------------------------
    # Explain this prediction
    # -----------------------------
    st.subheader("Why did the model make this prediction?")

    st.caption(
        "Feature contributions describe how the fitted model calculated "
        "this result. They do not establish real-world causes."
    )

    try:
        preprocessor = model.named_steps["preprocessor"]
        classifier = model.named_steps["classifier"]

        transformed = preprocessor.transform(input_data)

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = preprocessor.get_feature_names_out()
        coefficients = classifier.coef_[0]
        contributions = transformed[0] * coefficients

        explanation = pd.DataFrame({
            "Feature": feature_names,
            "Contribution": contributions,
        })

        explanation["Direction"] = explanation["Contribution"].apply(
            lambda value: (
                "Increases risk"
                if value > 0
                else "Decreases risk"
                if value < 0
                else "No contribution"
            )
        )

        explanation["Magnitude"] = explanation["Contribution"].abs()

        explanation = explanation.sort_values(
            "Magnitude",
            ascending=False,
        ).head(5).copy()

        # Convert technical feature names to readable labels
        def readable_feature(name):
            name = name.replace("numeric__", "")
            name = name.replace("categorical__", "")

            labels = {
                "arrival_density": "Vessel arrival density",
                "port_capacity": "Port capacity",
                "queue_length": "Queue length",
                "weather_severity": "Weather severity",
                "historical_avg_delay": "Historical average delay",
                "hour_of_day": "Hour of day",
            }

            if name.startswith("season_"):
                season_name = name.replace("season_", "").title()
                return f"{season_name} season"

            return labels.get(name, name.replace("_", " ").title())

        explanation["Feature"] = explanation["Feature"].apply(
            readable_feature
        )

        explanation = explanation.rename(
            columns={"Contribution": "Log-odds contribution"}
        )

        st.dataframe(
            explanation[
                ["Feature", "Direction", "Log-odds contribution"]
            ].round(3),
            use_container_width=True,
            hide_index=True,
        )

        # Build a summary from the strongest contributions
        positive = explanation[
            explanation["Log-odds contribution"] > 0
        ].sort_values("Log-odds contribution", ascending=False)

        negative = explanation[
            explanation["Log-odds contribution"] < 0
        ].sort_values("Log-odds contribution")

        st.subheader("Prediction summary")

        if probability >= 0.70:
            st.warning(
                f"The model estimates elevated congestion risk "
                f"({probability:.1%} on the model's probability scale)."
            )
        elif probability >= 0.40:
            st.info(
                f"The model estimates an intermediate congestion risk "
                f"({probability:.1%} on the model's probability scale)."
            )
        else:
            st.success(
                f"The model estimates a lower congestion risk "
                f"({probability:.1%} on the model's probability scale)."
            )

        if not positive.empty:
            top_positive = ", ".join(
                positive["Feature"].head(3).tolist()
            )
            st.write(
                f"**Factors pushing toward congestion:** {top_positive}."
            )

        if not negative.empty:
            top_negative = ", ".join(
                negative["Feature"].head(3).tolist()
            )
            st.write(
                f"**Factors pushing away from congestion:** {top_negative}."
            )

        st.caption(
            "These thresholds are illustrative. The model was trained on "
            "synthetic data, and its probability estimates have not been "
            "validated or calibrated for real-world port operations."
        )

    except (AttributeError, KeyError, ValueError) as error:
        st.info(
            "The prediction succeeded, but its explanation could not "
            "be generated for this model artifact."
        )
        st.caption(f"Technical detail: {error}")

    # -----------------------------
    # Show entered conditions
    # -----------------------------
    with st.expander("Review the input conditions"):
        st.dataframe(
            input_data,
            use_container_width=True,
            hide_index=True,
        )
