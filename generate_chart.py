import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_dashboard_visualizations():
    print("--- Generating Analytical Visualizations ---")
    os.makedirs('static', exist_ok=True)
    
    df = pd.read_csv('data/customer_churn.csv')
    
    # Set plot styling
    sns.set_theme(style="whitegrid")
    
    # Chart 1: Churn Distribution Pie/Bar Chart
    plt.figure(figsize=(6, 4))
    sns.countplot(x='Churn', data=df, palette='Set2')
    plt.title('Customer Churn Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Churn Status (0 = Stayed, 1 = Churned)')
    plt.ylabel('Number of Customers')
    plt.tight_layout()
    plt.savefig('static/churn_distribution.png', dpi=300)
    plt.close()
    
    # Chart 2: Churn by Customer Service Calls
    plt.figure(figsize=(7, 4))
    sns.countplot(x='Customer service calls', hue='Churn', data=df, palette='coolwarm')
    plt.title('Churn Impact by Customer Service Calls', fontsize=12, fontweight='bold')
    plt.xlabel('Customer Service Calls')
    plt.ylabel('Customer Count')
    plt.legend(title='Churn', labels=['Stayed', 'Churned'])
    plt.tight_layout()
    plt.savefig('static/churn_by_service_calls.png', dpi=300)
    plt.close()
    
    print("Success: Dashboard charts saved to 'static/' folder.")

if __name__ == "__main__":
    generate_dashboard_visualizations()