import sys
import os
import pandas as pd

# Add project root to sys.path BEFORE importing src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.predict import load_model, predict

def test_predict_shape():
    model = load_model("models/best_pipeline.joblib")  # relative path from tests/
    
    columns = [
        "V1","V2","V3","V4","V5","V6","V7","V8","V9","V10",
        "V11","V12","V13","V14","V15","V16","V17","V18","V19","V20",
        "V21","V22","V23","V24","V25","V26","V27","V28",
        "Amount","Hour","Day","Amount_log","Amount_scaled","Amount_log_scaled","PCA1","PCA2"
    ]

    values = [
        -1.3598071336738,-0.0727811733098497,2.53634673796914,1.37815522427443,-0.338320769942518,
        0.462387777762292,0.239598554061257,0.0986979012610507,0.363786969611213,0.0907941719789316,
        -0.551599533260813,-0.617800855762348,-0.991389847235408,-0.311169353699879,1.46817697209427,
        -0.470400525259478,0.207971241929242,0.0257905801985591,0.403992960255733,0.251412098239705,
        -0.018306777944153,0.277837575558899,-0.110473910188767,0.0669280749146731,0.128539358273528,
        -0.189114843888824,0.133558376740387,-0.0210530534538215,149.62,0.0,0.0,5.014760108673205,
        0.24496426337017327,1.1243033414119306,-0.34271466910341264,-0.002374971344161145
    ]

    sample = pd.DataFrame([values], columns=columns)
    preds = predict(model, sample)

    assert preds.shape[0] == sample.shape[0]