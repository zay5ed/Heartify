
import pandas as pd

# Load the downloaded dataset
data = pd.read_csv("data/raw/heart_disease.csv")

# 1. Dataset dimensions
print("\n--- DATASET SHAPE ---")
print(data.shape)

# 2. Column names
print("\n--- COLUMN NAMES ---")
print(data.columns.tolist())

# 3. Data types
print("\n--- DATA TYPES ---")
print(data.dtypes)

# 4. Missing values
print("\n--- MISSING VALUES ---")
print(data.isnull().sum())

# 5. Target distribution
print("\n--- TARGET DISTRIBUTION ---")
print(data["num"].value_counts().sort_index())

# 6. Duplicate records
print("\n--- DUPLICATE RECORDS ---")
print(data.duplicated().sum())

# 7. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(data.describe())
