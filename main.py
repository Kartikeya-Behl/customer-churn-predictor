import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv('./E-Commerce_Churn_Data.csv')

print("Dataset Shape:", df.shape)

print(df.head())

# Summary of data types and memory usage
df.info()

# Check for duplicate rows
print("Duplicate Rows:", df.duplicated().sum())

# List column names to identify unique IDs or unwanted features
print("Columns:", df.columns.tolist())
