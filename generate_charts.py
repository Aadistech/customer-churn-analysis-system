import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RAW_DATA_PATH = 'data/customer_churn.csv'


def generate_dashboard_visualizations():
    print("--- Generating Analytical Visualizations ---")
    os.makedirs('static', exist_ok=True)

    df = pd.read_csv(RAW_DATA_PATH)
    sns.set_theme(style="whitegrid")

    # 1. Churn ratio breakdown (donut chart)
    plt.figure(figsize=(6, 5), dpi=300)
    churn_bool = df['Churn'].astype(str).isin(['True', '1', 'True '])
    counts = [(~churn_bool).sum(), churn_bool.sum()]  # [Retained, Churned]
    plt.pie(
        counts, labels=['Retained', 'Churned'],
        colors=['#0D9488', '#DC2626'],
        autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'),
    )
    plt.title('Overall Customer Churn Ratio', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('static/chart_churn_dist.png', dpi=300, transparent=True)
    plt.close()

    # 2. Customer service calls impact
    plt.figure(figsize=(7, 4.5), dpi=300)
    sns.countplot(x='Customer service calls', hue='Churn', data=df, palette=['#0D9488', '#DC2626'])
    plt.title('Impact of Customer Service Calls on Churn', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('Number of Customer Service Calls', fontweight='semibold')
    plt.ylabel('Customer Count', fontweight='semibold')
    plt.legend(title='Status', labels=['Retained', 'Churned'])
    sns.despine(left=True, top=True)
    plt.tight_layout()
    plt.savefig('static/chart_service_calls.png', dpi=300, transparent=True)
    plt.close()

    # 3. International plan churn
    plt.figure(figsize=(7, 4.5), dpi=300)
    sns.countplot(x='International plan', hue='Churn', data=df, palette=['#0D9488', '#DC2626'])
    plt.title('Churn Rate by International Plan Subscription', fontsize=13, fontweight='bold', pad=15)
    plt.xlabel('International Plan', fontweight='semibold')
    plt.ylabel('Customer Count', fontweight='semibold')
    plt.legend(title='Status', labels=['Retained', 'Churned'])
    sns.despine(left=True, top=True)
    plt.tight_layout()
    plt.savefig('static/chart_intl_plan.png', dpi=300, transparent=True)
    plt.close()

    print("Success: chart_churn_dist.png, chart_service_calls.png, chart_intl_plan.png saved to 'static/'.")


if __name__ == "__main__":
    generate_dashboard_visualizations()
