import joblib
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("1. Loading model...")
model = joblib.load('models/random_forest_model.pkl')

print("2. Setting up dummy data...")
# Get the exact columns the model expects
expected_cols = list(model.feature_names_in_)

# Create a blank dataframe with zeros
df_test = pd.DataFrame(0, index=[0], columns=expected_cols)

print("3. Attempting prediction...")
prediction = model.predict(df_test)

print(f"SUCCESS! The model predicted: {prediction}")
