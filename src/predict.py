import pandas as pd
import numpy as np
import joblib
import argparse

def create_features(df):
    """
    Creates all the necessary features from the raw data.
    This function must be identical to the feature engineering
    steps used to train the model.
    """
    print("Creating features...")
    
    # Time-based features
    df['hour'] = (df['Time'] // 3600) % 24
    df['business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)

    # Amount-based features
    df['log_amount'] = np.log1p(df['Amount'])
    df['amount_percentile'] = df['Amount'].rank(pct=True)
    
    # Drop original columns that have been transformed
    df = df.drop(['Time', 'Amount'], axis=1)
    
    return df

def main(input_path, output_path, model_path):
    """
    Main function to load data, preprocess, predict, and save results.
    """
    print(f"Loading data from {input_path}...")
    new_data = pd.read_csv(input_path)
    
    # Store the original data to append predictions to it later
    original_data = new_data.copy()

    # Create features
    processed_data = create_features(new_data)
    
    print(f"Loading model from {model_path}...")
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
        
    # Ensure columns match the order the model was trained on
    # NOTE: It's critical that these are the exact features the model expects.
    # You may need to adjust this list.
    features = [col for col in processed_data.columns if col.startswith('V') or col in ['hour', 'business_hours', 'log_amount', 'amount_percentile']]
    
    try:
        data_for_prediction = processed_data[features]
    except KeyError as e:
        print(f"Error: A required feature is missing from the data. Details: {e}")
        return

    print("Making predictions...")
    # Get class predictions (0 or 1)
    predictions = model.predict(data_for_prediction)
    
    # Get fraud probability scores
    probabilities = model.predict_proba(data_for_prediction)[:, 1]

    # Add predictions and probabilities to the original data
    original_data['predicted_class'] = predictions
    original_data['fraud_probability'] = probabilities
    
    print(f"Saving results to {output_path}...")
    original_data.to_csv(output_path, index=False)
    
    print("Prediction process completed successfully!")

if __name__ == "__main__":
    # Set up argument parser to make the script flexible
    parser = argparse.ArgumentParser(description="Predict fraud on new transaction data.")
    parser.add_argument('--input', required=True, help="Path to the input CSV file with new data.")
    parser.add_argument('--output', required=True, help="Path to save the output CSV file with predictions.")
    parser.add_argument('--model', default="models/best_model.pkl", help="Path to the trained model .pkl file.")
    
    args = parser.parse_args()
    
    main(args.input, args.output, args.model)