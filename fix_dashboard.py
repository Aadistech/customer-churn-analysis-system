import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure static folder exists
os.makedirs('static', exist_ok=True)
print("1. Verified 'static/' folder exists.")

# Load dataset
df = pd.read_csv('data/customer_churn.csv')

# Set theme
sns.set_theme(style="whitegrid")

# 1. Generate Churn Ratio Chart
plt.figure(figsize=(6, 5), dpi=300)
churn_counts = df['Churn'].value_counts()
plt.pie(churn_counts, labels=['Retained', 'Churned'], colors=['#10b981', '#ef4444'], 
        autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.4, edgecolor='w'))
plt.title('Overall Customer Churn Ratio', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('static/chart_churn_dist.png', dpi=300, transparent=True)
plt.close()

# 2. Generate Service Calls Chart
plt.figure(figsize=(7, 4.5), dpi=300)
sns.countplot(x='Customer service calls', hue='Churn', data=df, palette=['#3b82f6', '#ef4444'])
plt.title('Impact of Customer Service Calls on Churn', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('Number of Customer Service Calls', fontweight='semibold')
plt.ylabel('Customer Count', fontweight='semibold')
plt.legend(title='Status', labels=['Retained', 'Churned'])
sns.despine(left=True, top=True)
plt.tight_layout()
plt.savefig('static/chart_service_calls.png', dpi=300, transparent=True)
plt.close()

print("2. High-end charts successfully generated and saved to 'static/'.")