# Predicting Port Congestion and Shipment Delays Using AI

A machine learning project that estimates port congestion risk using operational, historical, and environmental features. It includes synthetic data generation, exploratory analysis, model training, model comparison, prediction explanations, and a Streamlit demo.

## 1. Project Objective

The objective is to demonstrate how machine learning can estimate the likelihood of port congestion and shipment delays from features such as vessel arrival density, port capacity, queue length, weather severity, historical average delay, time of day, and season.

**Important:** This project uses synthetic data for demonstration. Its predictions and evaluation metrics do not establish real-world performance at actual ports.

## 2. Dataset

The project generates a synthetic dataset of 300 records.

Features include:
- `arrival_density`
- `port_capacity`
- `queue_length`
- `weather_severity`
- `historical_avg_delay`
- `hour_of_day`
- `season`

The target, `congestion_label`, is generated using an illustrative rule based on simulated `delay_hours`: delays greater than 12 hours are labeled as congestion (`1`); other records are labeled `0`.

The generated dataset contains 208 positive labels and 92 negative labels, with no missing values in the dataset examined.

## 3. Project Structure

```text
port-congestion-prediction/
├── app.py
├── data/
│   ├── synthetic_ports.csv
│   ├── label_distribution.png
│   └── model_comparison.csv
├── explore_data.py
├── generate_data.py
├── train_model.py
├── evaluate_models.py
├── explain_prediction.py
├── predict.py
├── models/
│   └── logistic_regression.joblib
├── requirements.txt
└── README.md
```

## 4. Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit

## 5. Setup and Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 6. Run the Project

Run the scripts from the project root in the following order.

### Generate the dataset

```powershell
python generate_data.py
```

### Explore the data

```powershell
python explore_data.py
```

### Train the logistic regression model

```powershell
python train_model.py
```

### Compare models

```powershell
python evaluate_models.py
```

### Explain an example prediction

```powershell
python explain_prediction.py
```

### Generate an example prediction

```powershell
python predict.py
```

### Launch the Streamlit application

```powershell
python -m streamlit run app.py
```

The application provides an interactive interface for entering port-related features and viewing an estimated congestion probability, predicted class, illustrative risk category, and feature contributions.

## 7. Model Evaluation

The following metrics were obtained from a stratified 80/20 train-test split using the generated synthetic dataset.

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.783 | 0.854 | 0.833 | 0.843 | 0.913 |
| Decision Tree | 0.817 | 0.860 | 0.881 | 0.871 | 0.837 |
| Dummy Baseline | 0.700 | 0.700 | 1.000 | 0.824 | 0.500 |

The Decision Tree achieved the highest accuracy, recall, and F1-score among these tested models. Logistic Regression achieved the highest ROC-AUC.

These results are specific to the synthetic dataset and its label-generation rule. They should not be interpreted as validated performance on real-world shipping data.

## 8. Prediction Explanations

The project demonstrates feature-contribution explanations for Logistic Regression predictions. Contributions indicate how input features influence the model's calculated output relative to its learned coefficients.

These explanations describe model behavior, not causal relationships. They do not establish that a particular feature causes port congestion.

## 9. Limitations

- The dataset is synthetic and does not represent verified operational records from real ports.
- The target labels are generated from an illustrative delay rule.
- Model performance has not been validated on real-world or independently collected port data.
- Predicted probabilities have not been calibrated against real-world outcomes; they should not be treated as validated confidence scores.
- Risk thresholds used in the demo are illustrative and would require validation before operational use.
- Feature contributions are model explanations, not causal evidence.

## 10. Future Improvements

- Evaluate on real port, vessel arrival, weather, and shipment-delay data.
- Incorporate time-based validation and test on unseen periods or ports.
- Calibrate predicted probabilities and assess calibration quality.
- Compare additional classification models and tune their hyperparameters.
- Add monitoring and retraining as new verified data becomes available.

## Conclusion

This project demonstrates an end-to-end machine learning workflow for a port congestion prediction use case, including synthetic data generation, exploratory analysis, model training, evaluation, prediction explanations, and an interactive demo. Further real-world data collection and validation would be necessary before using the system for operational decisions.
