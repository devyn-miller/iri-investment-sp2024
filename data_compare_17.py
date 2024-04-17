import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind

# Load the datasets
df1 = pd.read_csv('investment_2023-12-11.csv')
df2 = pd.read_csv('investment_S2_2023-12-13.csv')

# Convert 'Parts Completed' and 'Current Payment' to numeric, handling missing values
df1['Parts Completed'] = pd.to_numeric(df1['Parts Completed'], errors='coerce')
df1['Current Payment'] = pd.to_numeric(df1['Current Payment'], errors='coerce')
df2['Parts Completed'] = pd.to_numeric(df2['Parts Completed'], errors='coerce')
df2['Current Payment'] = pd.to_numeric(df2['Current Payment'], errors='coerce')

# Assuming 'Soft Deadline Remaining', 'Hard Deadline Remaining', and 'n_workers' columns exist in your datasets

# Convert relevant columns to numeric
df1['Soft Deadline Remaining'] = pd.to_numeric(df1['Soft Deadline Remaining'], errors='coerce')
df1['Hard Deadline Remaining'] = pd.to_numeric(df1['Hard Deadline Remaining'], errors='coerce')
df2['Soft Deadline Remaining'] = pd.to_numeric(df2['Soft Deadline Remaining'], errors='coerce')
df2['Hard Deadline Remaining'] = pd.to_numeric(df2['Hard Deadline Remaining'], errors='coerce')

# Group by 'Day' and calculate mean for 'Parts Completed', 'Current Payment', and additional metrics
df1_grouped = df1.groupby('Day')[['Parts Completed', 'Current Payment', 'Soft Deadline Remaining', 'Hard Deadline Remaining', 'n_workers']].mean().reset_index()
df2_grouped = df2.groupby('Day')[['Parts Completed', 'Current Payment', 'Soft Deadline Remaining', 'Hard Deadline Remaining', 'n_workers']].mean().reset_index()

# Merge the grouped data for comparison
merged_df = pd.merge(df1_grouped, df2_grouped, on='Day', suffixes=('_df1', '_df2'))

# Perform a T-test for 'Parts Completed' and 'Current Payment' between the two datasets
ttest_parts = ttest_ind(df1['Parts Completed'].dropna(), df2['Parts Completed'].dropna())
ttest_payment = ttest_ind(df1['Current Payment'].dropna(), df2['Current Payment'].dropna())

print(f"T-test for Parts Completed: statistic={ttest_parts.statistic}, p-value={ttest_parts.pvalue}")
print(f"T-test for Current Payment: statistic={ttest_payment.statistic}, p-value={ttest_payment.pvalue}")

# Plotting
# Plot for 'Parts Completed'
fig_parts = px.line(merged_df, x='Day', y=['Parts Completed_df1', 'Parts Completed_df2'], title='Parts Completed Comparison')
fig_parts.show()

# Plot for 'Current Payment'
fig_payment = px.line(merged_df, x='Day', y=['Current Payment_df1', 'Current Payment_df2'], title='Current Payment Comparison')
fig_payment.show()
# Plotting additional metrics
# Example: Plot for 'Soft Deadline Remaining'
fig_soft_deadline = px.line(merged_df, x='Day', y=['Soft Deadline Remaining_df1', 'Soft Deadline Remaining_df2'], title='Soft Deadline Remaining Comparison')
fig_soft_deadline.show()

# Repeat plotting for 'Hard Deadline Remaining' and 'n_workers'
fig_hard_deadline = px.line(merged_df, x='Day', y=['Hard Deadline Remaining_df1', 'Hard Deadline Remaining_df2'], title='Hard Deadline Remaining Comparison')
fig_hard_deadline.show()

fig_n_workers = px.line(merged_df, x='Day', y=['n_workers_df1', 'n_workers_df2'], title='Number of Workers Comparison')
fig_n_workers.show()
