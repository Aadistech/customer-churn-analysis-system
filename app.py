from flask import Flask, render_template, request
import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

MODEL_PATH = 'models/random_forest_model.pkl'
CLEANED_DATA_PATH = 'output/cleaned_customer_churn.csv'
RAW_DATA_PATH = 'data/customer_churn.csv'

def load_model_and_stats():
    os.makedirs('models', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            if hasattr(m, 'feature_names_in_'):
                return m, list(m.feature_names_in_)
        except Exception:
            pass
            
    df = pd.read_csv(RAW_DATA_PATH) if os.path.exists(RAW_DATA_PATH) else pd.read_csv('cleaned_customer_churn.csv')
    if 'Churn' in df.columns and df['Churn'].dtype == 'O':
        df['Churn'] = df['Churn'].map({True: 1, False: 0, 'True': 1, 'False': 0})
    
    X = df.drop(columns=['Churn'], errors='ignore')
    y = df['Churn'].astype(int) if 'Churn' in df.columns else pd.Series([0]*len(df))
    
    m = RandomForestClassifier(n_estimators=50, random_state=42)
    m.fit(X, y)
    joblib.dump(m, MODEL_PATH)
    return m, list(X.columns)

model, EXPECTED_COLS = load_model_and_stats()

def get_kpi_metrics():
    if os.path.exists(RAW_DATA_PATH):
        df = pd.read_csv(RAW_DATA_PATH)
        total = len(df)
        if df['Churn'].dtype == 'O':
            churned = int((df['Churn'] == True).sum() | (df['Churn'] == 'True').sum())
        else:
            churned = int(df['Churn'].sum())
        retained = total - churned
        rate = round((churned / total) * 100, 2)
        avg_tenure = round(df['Account length'].mean(), 1)
        avg_day_mins = round(df['Total day minutes'].mean(), 1)
        return total, churned, retained, rate, avg_tenure, avg_day_mins
    return 667, 95, 572, 14.24, 102.8, 180.9

@app.route('/')
def home():
    return render_template('index.html', active_page='predict')

@app.route('/dashboard')
def dashboard():
    total, churned, retained, rate, avg_tenure, avg_day_mins = get_kpi_metrics()
    return render_template('dashboard.html', 
                           total=total, 
                           churned=churned, 
                           retained=retained, 
                           rate=rate, 
                           avg_tenure=avg_tenure, 
                           avg_day_mins=avg_day_mins, 
                           active_page='dashboard')

@app.route('/evaluation')
def evaluation():
    total, churned, retained, rate, avg_tenure, avg_day_mins = get_kpi_metrics()
    return render_template('evaluation.html', active_page='evaluation')

@app.route('/batch')
def batch():
    total, churned, retained, rate, avg_tenure, avg_day_mins = get_kpi_metrics()
    return render_template('batch.html', active_page='batch')

@app.route('/about')
def about():
    total, churned, retained, rate, avg_tenure, avg_day_mins = get_kpi_metrics()
    return render_template('about.html', active_page='about')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form.to_dict()
        df_input = pd.DataFrame(0, index=[0], columns=EXPECTED_COLS)
        
        if 'Account length' in df_input.columns:
            df_input['Account length'] = df_input['Account length'].astype(float)
            df_input.at[0, 'Account length'] = float(user_input.get('account_length', 0))
            
        if 'Total day minutes' in df_input.columns:
            df_input['Total day minutes'] = df_input['Total day minutes'].astype(float)
            df_input.at[0, 'Total day minutes'] = float(user_input.get('total_day_minutes', 0))
            
        if 'Customer service calls' in df_input.columns:
            df_input['Customer service calls'] = df_input['Customer service calls'].astype(float)
            df_input.at[0, 'Customer service calls'] = float(user_input.get('customer_service_calls', 0))
            
        if 'International plan' in df_input.columns:
            df_input['International plan'] = df_input['International plan'].astype(float)
            df_input.at[0, 'International plan'] = int(user_input.get('intl_plan', 0))
        
        defaults = {'Area code': 415, 'Total day calls': 100, 'Total eve minutes': 200, 'Total night minutes': 200}
        for col, val in defaults.items():
            if col in df_input.columns:
                df_input[col] = df_input[col].astype(float)
                df_input.at[0, col] = float(val)

        prediction = model.predict(df_input)[0]
        probabilities = model.predict_proba(df_input)[0]
        churn_prob = round(probabilities[1] * 100, 1)
        
        if prediction == 1:
            result_text = f"HIGH RISK: This customer is likely to CHURN. (Probability: {churn_prob}%)"
        else:
            result_text = f"LOW RISK: This customer is likely to STAY. (Probability: {churn_prob}%)"
            
        return render_template('index.html', prediction_text=result_text, active_page='predict')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error during prediction: {str(e)}", active_page='predict')

if __name__ == "__main__":
    app.run(debug=True)