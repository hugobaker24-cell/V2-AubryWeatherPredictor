import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


# 1. LOAD DATA

df = pd.read_csv("WeatherAus.csv")


# 2. TARGET: TRUE FORECASTING

# RainTomorrow: Yes / No → 1 / 0
df = df.dropna(subset=["RainTomorrow"])
df["rain_tomorrow"] = df["RainTomorrow"].map({"Yes": 1, "No": 0})
# RainToday: Yes / No → 1 / 0
df["rain_today"] = df["RainToday"].map({"Yes": 1, "No": 0})


# 3. FEATURE SET (BOM DATA)

features = [
    "rain_today",
    "Temp9am",
    "Temp3pm",
    "Humidity9am",
    "Humidity3pm",
    "Pressure9am",
    "Pressure3pm",
    "WindSpeed9am",
    "WindSpeed3pm",
    "Cloud9am",
    "Cloud3pm"
]

df = df[features + ["rain_tomorrow"]].dropna()

X = df[features]
y = df["rain_tomorrow"]


# 4. TRAIN / TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# 5. TRAIN MODEL

model = RandomForestClassifier(
    n_estimators=400,
    max_depth=12,
    min_samples_leaf=2,
    class_weight={0: 1, 1: 3},
    random_state=42
)

model.fit(X_train, y_train)


# SAVE FULL FORECASTING PIPELINE

pipeline = {
    "model": model,
    "features": features,
    "threshold": 0.30,
    "description": "RandomForest RainTomorrow forecaster with RainToday feature"
}

joblib.dump(pipeline, "rain_forecast_V2.joblib")
print("Forecasting pipeline saved as rain_forecast_V2.joblib")

# ------------------------------
# 6. EVALUATION
# ------------------------------
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.30).astype(int)

print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("Decision Threshold: 30% (RainTomorrow)\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 7. FEATURE IMPORTANCE

importance = pd.Series(
    model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(importance)


# 8. EXAMPLE FORECAST

example_day = pd.DataFrame([{
    "rain_today": 1,
    "Temp9am": 14.5,
    "Temp3pm": 21.0,
    "Humidity9am": 85,
    "Humidity3pm": 60,
    "Pressure9am": 1016,
    "Pressure3pm": 1013,
    "WindSpeed9am": 12,
    "WindSpeed3pm": 22,
    "Cloud9am": 7,
    "Cloud3pm": 6
}])

example_day = example_day[features]

rain_probability = model.predict_proba(example_day)[0][1] * 100

print("\nForecast for Tomorrow:")
print(f"Rain Probability: {rain_probability:.1f}%")
print("Rain Tomorrow →", "YES 🌧️" if rain_probability >= 30 else "NO ☀️")