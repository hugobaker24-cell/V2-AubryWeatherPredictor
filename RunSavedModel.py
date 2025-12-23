import joblib
import pandas as pd

# Load pipeline
pipeline = joblib.load("rain_forecast_V2.joblib")

model = pipeline["model"]
features = pipeline["features"]
threshold = pipeline["threshold"]

# New forecast input
new_day = pd.DataFrame([{
    "rain_today": 0,
    "Temp9am": 13.2,
    "Temp3pm": 19.8,
    "Humidity9am": 78,
    "Humidity3pm": 55,
    "Pressure9am": 1018,
    "Pressure3pm": 1015,
    "WindSpeed9am": 10,
    "WindSpeed3pm": 18,
    "Cloud9am": 5,
    "Cloud3pm": 4
}])[features]

# Predict
prob = model.predict_proba(new_day)[0][1]

print(f"Rain probability: {prob*100:.1f}%")
print("Rain Tomorrow →", "YES 🌧️" if prob >= threshold else "NO ☀️")