# ===============================
#  PAYROLL DATA ANALYSIS PROJECT
# ===============================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Style
plt.style.use("default")
sns.set_context("notebook")

# ===============================
# 1. LOAD DATA
# ===============================
df = pd.read_csv("MTA_Employee_Payroll__Beginning_2025.csv")

print("Shape:", df.shape)
print(df.info())
print(df.describe(include='all'))

# ===============================
# 2. DATA CLEANING
# ===============================

# Missing values
print("\nMissing Values:\n", df.isnull().sum())

# Drop duplicates
df = df.drop_duplicates()

# Convert numeric columns
cols = ['Regular Pay','Overtime Pay','Cash Outs','Retro Pay','Other Pay','Total Earnings']
for col in cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Remove outliers using IQR on Total Earnings
Q1 = df['Total Earnings'].quantile(0.25)
Q3 = df['Total Earnings'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df = df[(df['Total Earnings'] >= lower) & (df['Total Earnings'] <= upper)]

print("Cleaned Shape:", df.shape)

# ===============================
# 3. OUTLIER VISUALIZATION
# ===============================
plt.figure()
sns.boxplot(x=df['Total Earnings'])
plt.title("Total Earnings Outliers")
plt.show()

# ===============================
# 4. EARNINGS DISTRIBUTION
# ===============================
plt.figure()
sns.histplot(df['Total Earnings'], bins=30, kde=True)
plt.title("Total Earnings Distribution")
plt.xlabel("Earnings")
plt.ylabel("Count")
plt.show()

# ===============================
# 5. DEPARTMENT ANALYSIS
# ===============================
top_dept = df.groupby('Department')['Total Earnings'].mean().sort_values(ascending=False).head(10)

plt.figure()
top_dept.plot(kind='bar')
plt.title("Top Departments by Avg Earnings")
plt.xticks(rotation=45)
plt.show()

# ===============================
# 6. JOB TITLE ANALYSIS
# ===============================
top_titles = df.groupby('Title')['Total Earnings'].mean().sort_values(ascending=False).head(10)

plt.figure()
top_titles.plot(kind='bar')
plt.title("Top Job Titles by Earnings")
plt.xticks(rotation=45)
plt.show()

# ===============================
# 7. PAY COMPONENT ANALYSIS
# ===============================
pay_components = df[['Regular Pay','Overtime Pay','Other Pay']].sum()

plt.figure()
pay_components.plot(kind='bar')
plt.title("Pay Components Comparison")
plt.ylabel("Total Amount")
plt.show()

# ===============================
# 8. OVERTIME IMPACT
# ===============================
plt.figure()
sns.scatterplot(x='Overtime Pay', y='Total Earnings', data=df)
sns.regplot(x='Overtime Pay', y='Total Earnings', data=df, scatter=False)
plt.title("Overtime Pay vs Total Earnings")
plt.show()

# ===============================
# 9. PAY BASIS ANALYSIS
# ===============================
plt.figure()
sns.boxplot(x='Pay Basis', y='Total Earnings', data=df)
plt.title("Pay Basis vs Earnings")
plt.show()

# ===============================
# 10. CORRELATION HEATMAP
# ===============================
plt.figure()
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Features (Independent variables)
X = df[['Regular Pay', 'Overtime Pay', 'Other Pay', 'Cash Outs', 'Retro Pay']]

# Target (Dependent variable)
y = df['Total Earnings']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# ===============================
# MODEL EVALUATION
# ===============================
print("MODEL PERFORMANCE")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
print("R2 Score:", r2_score(y_test, y_pred))

# ===============================
# COEFFICIENTS
# ===============================
coeff_df = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n Feature Importance:\n", coeff_df)

# ===============================
# ACTUAL vs PREDICTED PLOT
# ===============================
plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Earnings")
plt.ylabel("Predicted Earnings")
plt.title("Actual vs Predicted Earnings")

# Perfect prediction line
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.show()
# ===============================
# 11. MINI DASHBOARD
# ===============================
plt.figure(figsize=(10,8))

# Distribution
plt.subplot(2,2,1)
sns.histplot(df['Total Earnings'], kde=True)
plt.title("Earnings Distribution")

# Overtime vs Earnings
plt.subplot(2,2,2)
sns.scatterplot(x='Overtime Pay', y='Total Earnings', data=df)
plt.title("Overtime Impact")

# Departments
plt.subplot(2,2,3)
top_dept.plot(kind='bar')
plt.title("Top Departments")

# Job Titles
plt.subplot(2,2,4)
top_titles.sort_values().plot(kind='barh')
plt.title("Top Job Titles")

plt.tight_layout()
plt.show()


