
from ucimlrepo import fetch_ucirepo
import pandas as pd
from pathlib import Path

# Fetch the UCI Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Separate input features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine features and target into one dataset
data = pd.concat([X, y], axis=1)

# Create the raw data directory if it doesn't exist
Path("data/raw").mkdir(parents=True, exist_ok=True)

# Save the original dataset
data.to_csv("data/raw/heart_disease.csv", index=False)

# Display basic dataset information
print("Dataset downloaded successfully!")
print("Dataset shape:", data.shape)

print("\nFirst five records:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())
