import joblib
import pandas as pd

def load_model(path="models/best_pipeline.joblib"):
    return joblib.load(path)

def predict(model, data: pd.DataFrame):
    return model.predict(data)