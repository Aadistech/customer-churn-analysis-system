from flask import Flask, render_template, request, send_from_directory
import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

MODEL_PATH = 'models/random_forest_model.pkl'
CLEANED_DATA_PATH = 'output/cleaned_customer_churn.csv'
RAW_DATA_PATH = 'data/customer_churn.csv'
BATCH_OUTPUT_PATH = 'output/bulk_prediction_results.csv'


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

    X = pd.get_dummies(df.drop(columns=['Churn'], errors='ignore'), columns=['State', 'Area code'])
    X['International plan'] = X['International plan'].map({'yes': 1, 'no': 0}).fillna(0)
    X['Voice mail plan'] = X['Voice mail plan'].map({'yes': 1, 'no': 0}).fillna(0)
    y = df['Churn'].astype(int) if 'Churn' in df.columns else pd.Series([0] * len(df))

    m = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
    m.fit(X, y)
    joblib.dump(m, MODEL_PATH)
    return m, list(X.columns)


model, EXPECTED_COLS = load_model_and_stats()


def build_feature_vector(record):
    """Turn a raw customer record (plain column names, yes/no plan flags) into
    the one-hot encoded row the trained model expects."""
    row = pd.DataFrame(0.0, index=[0], columns=EXPECTED_COLS)

    numeric_direct = [
        'Account length', 'Number vmail messages', 'Total day minutes', 'Total day calls',
        'Total day charge', 'Total eve minutes', 'Total eve calls', 'Total eve charge',
        'Total night minutes', 'Total night calls', 'Total night charge', 'Total intl minutes',
        'Total intl calls', 'Total intl charge', 'Customer service calls',
    ]
    for col in numeric_direct:
        if col in row.columns:
            try:
                row.at[0, col] = float(record.get(col, 0) or 0)
            except (TypeError, ValueError):
                row.at[0, col] = 0.0

    def truthy(val):
        return str(val).strip().lower() in ('yes', '1', 'true')

    if 'International plan' in row.columns:
        row.at[0, 'International plan'] = 1.0 if truthy(record.get('International plan')) else 0.0
    if 'Voice mail plan' in row.columns:
        row.at[0, 'Voice mail plan'] = 1.0 if truthy(record.get('Voice mail plan')) else 0.0

    state_col = 'State_' + str(record.get('State', '')).strip().upper()
    if state_col in row.columns:
        row.at[0, state_col] = 1.0

    area_col = 'Area code_' + str(record.get('Area code', '')).strip()
    if area_col in row.columns:
        row.at[0, area_col] = 1.0

    return row


def score(record):
    vec = build_feature_vector(record)
    pred = int(model.predict(vec)[0])
    prob = round(float(model.predict_proba(vec)[0][1]) * 100, 1)
    return pred, prob


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
    return render_template('evaluation.html', active_page='evaluation')


@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'GET':
        return render_template('batch.html', active_page='batch')

    file = request.files.get('dataset')
    if not file or file.filename == '':
        return render_template('batch.html', batch_error='Please choose a CSV file to upload.', active_page='batch')

    try:
        df = pd.read_csv(file)
        rows = []
        for _, r in df.iterrows():
            record = r.to_dict()
            pred, prob = score(record)
            rows.append({
                'State': record.get('State', ''),
                'risk': 'HIGH' if pred == 1 else 'LOW',
                'churn_probability': prob,
            })

        out_df = pd.DataFrame(rows)
        os.makedirs('output', exist_ok=True)
        out_df.to_csv(BATCH_OUTPUT_PATH, index=False)

        summary = {
            'total': len(out_df),
            'high_risk': int((out_df['risk'] == 'HIGH').sum()),
            'low_risk': int((out_df['risk'] == 'LOW').sum()),
        }
        return render_template('batch.html',
                                batch_summary=summary,
                                batch_rows=out_df.head(15).to_dict('records'),
                                active_page='batch')
    except Exception as e:
        return render_template('batch.html', batch_error=f'Could not process this file: {e}', active_page='batch')


@app.route('/download-batch-results')
def download_batch():
    directory = os.path.join(os.getcwd(), 'output')
    return send_from_directory(directory, 'bulk_prediction_results.csv', as_attachment=True)


@app.route('/about')
def about():
    return render_template('about.html', active_page='about')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        f = request.form
        record = {
            'State': f.get('state'),
            'Account length': f.get('account_length'),
            'Area code': f.get('area_code'),
            'International plan': f.get('international_plan'),
            'Voice mail plan': f.get('voice_mail_plan'),
            'Number vmail messages': f.get('number_vmail_messages'),
            'Total day minutes': f.get('total_day_minutes'),
            'Total day calls': f.get('total_day_calls'),
            'Total day charge': f.get('total_day_charge'),
            'Total eve minutes': f.get('total_eve_minutes'),
            'Total eve calls': f.get('total_eve_calls'),
            'Total eve charge': f.get('total_eve_charge'),
            'Total night minutes': f.get('total_night_minutes'),
            'Total night calls': f.get('total_night_calls'),
            'Total night charge': f.get('total_night_charge'),
            'Total intl minutes': f.get('total_intl_minutes'),
            'Total intl calls': f.get('total_intl_calls'),
            'Total intl charge': f.get('total_intl_charge'),
            'Customer service calls': f.get('customer_service_calls'),
        }

        prediction, churn_prob = score(record)
        risk_level = 'high' if prediction == 1 else 'low'

        if risk_level == 'high':
            result_text = f"This customer is likely to CHURN, with a predicted probability of {churn_prob}%."
        else:
            result_text = f"This customer is likely to STAY, with a predicted churn probability of only {churn_prob}%."

        return render_template('index.html',
                                prediction_text=result_text,
                                churn_probability=churn_prob,
                                risk_level=risk_level,
                                active_page='predict')

    except Exception as e:
        return render_template('index.html', prediction_error=str(e), active_page='predict')


if __name__ == "__main__":
    app.run(debug=True)
