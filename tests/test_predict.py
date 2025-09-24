import pandas as pd
from src.predict import load_model, predict

def test_predict_shape():
    model = load_model("models/best_pipeline.joblib")
    sample = pd.DataFrame({
        "V1": [0.1], "V2": [-1.2], "V3": [0.3], # … fill all required features
        "Amount": [50], "Time": [1000]
    })
    preds = predict(model, sample)
    assert preds.shape[0] == sample.shape[0]