
import joblib
import pandas as pd

model = joblib.load("models/logistic_regression.joblib")

# Illustrative new port observation
sample = pd.DataFrame([{
    "arrival_density": 38,
    "port_capacity": 22,
    "queue_length": 19,
    "weather_severity": 3,
    "historical_avg_delay": 16.5,
    "hour_of_day": 18,
    "season": "winter",
}])

# Predict the class and estimated probability
predicted_label = int(model.predict(sample)[0])
delay_probability = float(model.predict_proba(sample)[0, 1])

if delay_probability >= 0.70:
    risk_level = "High"
elif delay_probability >= 0.40:
    risk_level = "Medium"
else:
    risk_level = "Low"

print("Port congestion prediction")
print("-------------------------")
print(f"Estimated delay probability: {delay_probability:.1%}")
print(f"Predicted class: {predicted_label}")
print(f"Risk category: {risk_level}")
