import pandas as pd
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# --- Configuration & File Paths ---
RAW_DATA_PATH = 'data/customer_churn.csv'
CLEANED_DATA_PATH = 'output/cleaned_customer_churn.csv'
MODEL_PATH = 'models/random_forest_model.pkl'
FINAL_PREDICTIONS_PATH = 'output/bulk_prediction_results.csv'
METRICS_PATH = 'output/evaluation_metrics.json'

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
        df[col] = df[col].map({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 1: 1, 0: 0})

    df['Churn'] = df['Churn'].astype(int)

    # One-hot encode categorical features
    df_clean = pd.get_dummies(df, columns=['State', 'Area code'])

    # Save the cleaned intermediate file
    df_clean.to_csv(cleaned_csv_path, index=False)
    return df, df_clean


# --- STEP 2: Train and Evaluate the Model ---
def train_model(df_clean, model_save_path, metrics_save_path):
    print("2. Training Random Forest model...")
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']

    # Split data while maintaining the churn ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model with balanced weights to handle class imbalance
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)

    y_pred = rf_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, target_names=['Stayed', 'Churned'])
    accuracy = accuracy_score(y_test, y_pred)

    print("\n--- Model Performance ---")
    print(classification_report(y_test, y_pred, target_names=['Stayed', 'Churned']))

    # Persist the real metrics so the dashboard can show truthful numbers
    metrics = {
        'accuracy': round(accuracy * 100, 1),
        'total_support': int(len(y_test)),
        'classes': {
            'stayed': {
                'precision': round(report['Stayed']['precision'], 2),
                'recall': round(report['Stayed']['recall'], 2),
                'f1': round(report['Stayed']['f1-score'], 2),
                'support': int(report['Stayed']['support']),
            },
            'churned': {
                'precision': round(report['Churned']['precision'], 2),
                'recall': round(report['Churned']['recall'], 2),
                'f1': round(report['Churned']['f1-score'], 2),
                'support': int(report['Churned']['support']),
            },
        },
    }
    with open(metrics_save_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to '{metrics_save_path}'")

    # Save the trained model to disk
    joblib.dump(rf_model, model_save_path)
    print(f"Model successfully saved to '{model_save_path}'")
    return rf_model, X


# --- STEP 3: Generate Business Output ---
def generate_output(df_raw, model, X_features, output_csv_path):
    print("3. Generating business predictions...")
    predictions = model.predict(X_features)
    probabilities = model.predict_proba(X_features)

    df_raw = df_raw.copy()
    df_raw['Prediction'] = np.where(predictions == 1, 'CUSTOMER WILL CHURN', 'CUSTOMER WILL STAY')
    df_raw['Churn Probability (%)'] = np.round(probabilities[:, 1] * 100, 1)

    # Assign Risk Levels based on probability
    df_raw['Risk Level'] = pd.cut(
        df_raw['Churn Probability (%)'],
        bins=[-1, 30, 70, 100],
        labels=['LOW', 'MEDIUM', 'HIGH']
    )

    # Assign actionable business rules
    df_raw['Recommended action'] = np.select(
        [df_raw['Risk Level'] == 'HIGH', df_raw['Risk Level'] == 'MEDIUM'],
        [
            'Urgent: contact the customer within 24 hours and offer a tailored retention plan.',
            'Monitor closely and consider a proactive check-in call.',
        ],
        default='Maintain engagement and monitor future usage.',
    )

    df_raw.to_csv(output_csv_path, index=False)
    print(f"\nSuccess! Final business predictions saved to '{output_csv_path}'.")


if __name__ == "__main__":
    print("=== Starting Customer Churn Analysis and Prediction System ===\n")

    df_raw, df_clean = clean_data(RAW_DATA_PATH, CLEANED_DATA_PATH)
    model, X_features = train_model(df_clean, MODEL_PATH, METRICS_PATH)
    generate_output(df_raw, model, X_features, FINAL_PREDICTIONS_PATH)
