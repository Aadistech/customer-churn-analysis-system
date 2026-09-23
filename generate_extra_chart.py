import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs('static', exist_ok=True)
df = pd.read_csv('data/customer_churn.csv')

plt.figure(figsize=(7, 4.5), dpi=300)
sns.countplot(x='International plan', hue='Churn', data=df, palette=['#10b981', '#ef4444'])
plt.title('Churn Rate by International Plan Subscription', fontsize=13, fontweight='bold', pad=15)
plt.xlabel('International Plan', fontweight='semibold')
plt.ylabel('Customer Count', fontweight='semibold')
plt.legend(title='Status', labels=['Retained', 'Churned'])
sns.despine(left=True, top=True)
plt.tight_layout()
plt.savefig('static/chart_intl_plan.png', dpi=300, transparent=True)
plt.close()
print("Extra chart generated successfully.")