# 🏢 Aadibi.ai — Enterprise Customer Churn Analysis & Prediction System

An end-to-end, corporate-grade Data Science, Business Intelligence (BI), and Machine Learning platform engineered from scratch to analyze historical telecom telemetry, uncover behavioral churn drivers, and predict customer migration risks in real-time.

---

## 🚀 Executive Summary
In subscription-driven industries, customer retention is paramount. **Aadibi.ai** bridges the gap between raw analytical data and real-time enterprise software. This system converts historical telemetry datasets into executive-level key performance indicators (KPIs), relational database queries, high-resolution BI visualizations, and a robust Scikit-Learn classification pipeline wrapped inside a modern SaaS web application.

---

## 🛠️ Technology Stack & Architecture

* **Programming Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Database Management:** SQL / MySQL (via SQLite storage layer for structured querying)
* **Machine Learning:** Scikit-learn (Random Forest Classifier with class balancing & Joblib serialization)
* **Backend Framework:** Flask
* **Frontend Interface:** HTML5, CSS3 (Modern Corporate SaaS Dark Header Theme), JavaScript
* **Version Control:** Git & GitHub (`Aadistech/customer-churn-analysis-system-final-build`)
* **Development Environment:** VS Code

---

## 📂 Project Structure

```text
customer-churn-analysis-system-final-build/
├── data/
│   └── customer_churn.csv        # Raw enterprise telemetry dataset
├── models/
│   └── random_forest_model.pkl   # Serialized ML classification model
├── output/
│   └── cleaned_customer_churn.csv # Processed dataset records
├── static/
│   ├── chart_churn_dist.png      # Churn distribution doughnut chart
│   ├── chart_service_calls.png   # Service calls impact bar chart
│   ├── chart_intl_plan.png       # International plan risk analysis chart
│   ├── style.css                 # Enterprise corporate styling
│   └── script.js                 # Client-side behavioral scripts
├── templates/
│   ├── header.html               # Reusable Amazon-style corporate top navigation component
│   ├── index.html                # Individual Customer Churn Predictor UI
│   ├── dashboard.html            # Executive BI Analytics Dashboard (KPIs & Visuals)
│   ├── evaluation.html           # Model Evaluation & Classification Report
│   ├── batch.html                # Bulk CSV Dataset Processing Module
│   └── about.html                # System Architecture & Documentation
├── app.py                        # Flask backend server & routing controller
├── database_setup.py             # SQL relational database ingestion & query script
├── generate_charts.py            # Automated BI chart generation pipeline
└── README.md                     # Comprehensive project documentation

📊 Key Features & Modules
Executive BI Analytics Dashboard:

Features 6 comprehensive KPI cards: Total Customer Base, Retained Customers, Churned Customers, Overall Churn Rate, Average Tenure, and Average Daily Usage Minutes.

Multi-chart analytical grid highlighting key behavioral friction points (Customer Service Thresholds, International Subscription Risks, and Usage Patterns).

Real-Time Individual Predictor:

Allows business analysts to input customer telemetry (Account Length, Day Minutes, Customer Service Calls, International Plan) and receive immediate risk categorization (HIGH RISK vs. LOW RISK) paired with precise probability scoring.

Machine Learning Model Evaluation:

Rigorously trained Random Forest Classifier evaluated using stratified cross-validation, achieving robust Precision, Recall, and F1-Score metrics designed to handle class imbalances.

Batch CSV Processing Module:

Scalable architecture supporting bulk dataset risk scoring and automated output logging.

SQL Database Layer:

Automated table ingestion and structured querying for reporting and filtering customer segments.

⚙️ Installation & Local Setup
Clone the Repository:

Bash
git clone [https://github.com/Aadistech/customer-churn-analysis-system-final-build.git](https://github.com/Aadistech/customer-churn-analysis-system-final-build.git)
cd customer-churn-analysis-system-final-build
Install Dependencies:

Bash
pip install pandas numpy matplotlib seaborn scikit-learn flask joblib
Initialize the Database & Charts:

Bash
python database_setup.py
python generate_charts.py
Run the Flask Web Application:

Bash
python app.py
Access the Platform:
Open your browser and navigate to: http://127.0.0.1:5000/

🎓 Academic & Industry Relevance
Developed as a 5th-semester mini project by Aadibi, this system demonstrates a practical mastery of full-stack data science—successfully transforming raw business data into structured relational tables, predictive machine learning models, and executive-ready software tools