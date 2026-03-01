import joblib
import numpy as np

rf_model = joblib.load("posture_rf_model.pkl")

def predict(features):
    features = np.array(features).reshape(1, -1)
    prediction = rf_model.predict(features)[0]
    return int(prediction)