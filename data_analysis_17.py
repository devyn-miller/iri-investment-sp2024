import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.power import TTestIndPower

# Load the datasets
df_baseline = pd.read_csv('investment_2023-12-11.csv')
df_treatment = pd.read_csv('investment_S2_2023-12-13.csv')

# Filter out the rows where 'Payment Received' is NaN or empty
df_baseline = df_baseline[pd.to_numeric(df_baseline['Payment Received'], errors='coerce').notnull()]
df_treatment = df_treatment[pd.to_numeric(df_treatment['Payment Received'], errors='coerce').notnull()]

# Summary statistics for baseline
summary_baseline = df_baseline['Payment Received'].describe()

# Summary statistics for treatment
summary_treatment = df_treatment['Payment Received'].describe()

print("Baseline Summary:\n", summary_baseline)
print("\nTreatment Summary:\n", summary_treatment)

# Calculate average payment received per participant for both sessions
avg_payment_baseline = df_baseline.groupby('SubjectID')['Payment Received'].mean()
avg_payment_treatment = df_treatment.groupby('SubjectID')['Payment Received'].mean()

# Calculate improvement
improvement = avg_payment_treatment - avg_payment_baseline

# Display improvement
print(improvement)

effect_size = 0.5  # Medium effect size
alpha = 0.05  # Significance level
power = 0.8  # Power

analysis = TTestIndPower()
sample_size = analysis.solve_power(effect_size=effect_size, power=power, alpha=alpha, ratio=1.0)
print(f"Required sample size: {sample_size}")

# Histograms of payments received
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(df_baseline['Payment Received'], kde=True, color='blue', label='Baseline')
plt.legend()
plt.subplot(1, 2, 2)
sns.histplot(df_treatment['Payment Received'], kde=True, color='green', label='Treatment')
plt.legend()
plt.show()

# Boxplot of improvement
plt.figure(figsize=(6, 4))
sns.boxplot(data=improvement, orient='h', color='orange')
plt.title('Improvement in Payment Received')
plt.show()
