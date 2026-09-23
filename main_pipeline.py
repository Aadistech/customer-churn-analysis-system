import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# --- Configuration & File Paths ---
RAW_DATA_PATH = 'data/customer_churn.csv'
CLEANED_DATA_PATH = 'output/cleaned_customer_churn.csv'
MODEL_PATH = 'models/random_forest_model.pkl'
FINAL_PREDICTIONS_PATH = 'output/bulk_prediction_results.csv'

# Safety check: Ensure necessary folders exist
os.makedirs('output', exist_ok=True)
os.makedirs('models', exist_ok=True)

# --- STEP 1: Preprocess the Data ---
def clean_data(raw_csv, cleaned_csv_path):
    print("1. Loading and cleaning data...")
    if not os.path.exists(raw_csv):
        raise FileNotFoundError(f"Missing file: {raw_csv}. Please ensure your dataset is inside the 'data/' folder.")
        
    df = pd.read_csv(raw_csv)
    
    # Convert Yes/No strings to binary 1/0
    binary_cols = ['International plan', 'Voice mail plan']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
    
    df['Churn'] = df['Churn'].astype(int)
    
    # One-hot encode categorical features
    df_clean = pd.get_dummies(df, columns=['State', 'Area code'])
    
    # Save the cleaned intermediate file
    df_clean.to_csv(cleaned_csv_path, index=False)
    return df, df_clean

# --- STEP 2: Train and Evaluate the Model ---
def train_model(df_clean, model_save_path):
    print("2. Training Random Forest model...")
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']
    
    # Split data while maintaining the churn ratio
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train model with balanced weights to handle class imbalance
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)
    
    print("\n--- Model Performance ---")
    print(classification_report(y_test, rf_model.predict(X_test)))
    
    # Save the trained model to disk
    joblib.dump(rf_model, model_save_path)
    print(f"Model successfully saved to '{model_save_path}'")
    return rf_model, X

# --- STEP 3: Generate Business Output ---
def generate_output(df_raw, model, X_features, output_csv_path):
    print("3. Generating business predictions...")
    predictions = model.predict(X_features)
    probabilities = model.predict_proba(X_features)
    
    # Map predictions back to the original dataframe
    df_raw['Prediction'] = np.where(predictions == 1, 'CUSTOMER WILL CHURN', 'CUSTOMER WILL STAY')
    df_raw['Churn Probability (%)'] = np.round(probabilities[:, 1] * 100, 1)
    
    # Assign Risk Levels based on probability
    df_raw['Risk Level'] = pd.cut(
        df_raw['Churn Probability (%)'], 
        bins=[-1, 30, 70, 100], 
        labels=['LOW', 'MEDIUM', 'HIGH']
    )
    
    # Assign actionable business rules
    df_raw['Recommended action'] = np.where(
        df_raw['Risk Level'] == 'HIGH', 
        'Urgent: contact the customer within 24 hours and offer a tailored retention plan.',
        'Maintain engagement and monitor future usage.'
    )
    
    # Save the final file
    df_raw.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! Final business predictions saved to '{output_csv_path}'.")

if __name__ == "__main__":
    print("=== Starting Customer Churn Analysis and Prediction System ===\n")
    
    # Run the pipeline steps
    df_raw, df_clean = clean_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    model, X_features = train_model(df_clean, MODEL_PATH)
    generate_output(df_raw, model, X_features, FINAL_PREDICTIONS_PATH)