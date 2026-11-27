
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "raw" / "heart_disease.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]


# ==================================================
# 2. DATA LOADING
# ==================================================

def load_data():
    """Load and return the raw heart disease dataset."""

    df = pd.read_csv(DATA_PATH)

    return df


# ==================================================
# 3. FEATURE AND TARGET PREPARATION
# ==================================================

def prepare_features(df):
    """Create the binary target and separate X and y."""

    df = df.copy()

    # Validate original target values
    assert df["num"].isin([0, 1, 2, 3, 4]).all()

    # Convert original target into binary target
    df["target"] = (df["num"] > 0).astype(int)

    # Separate features and target
    X = df.drop(columns=["num", "target"])
    y = df["target"]

    # Validate feature classification
    all_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    assert len(all_features) == len(set(all_features))
    assert set(all_features) == set(X.columns)

    return X, y


# ==================================================
# 4. TRAIN-TEST SPLITTING
# ==================================================

def split_data(X, y):
    """Perform a reproducible stratified train-test split."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # Validate record separation and target alignment
    assert set(X_train.index).isdisjoint(X_test.index)
    assert X_train.index.equals(y_train.index)
    assert X_test.index.equals(y_test.index)

    return X_train, X_test, y_train, y_test


# ==================================================
# 5. PREPROCESSING PIPELINE
# ==================================================

def build_preprocessor():
    """Construct and return an unfitted preprocessor."""

    # Numerical preprocessing
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    # Combine both pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, NUMERICAL_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder="drop"
    )

    return preprocessor


# ==================================================
# 6. PREPROCESSING VERIFICATION
# ==================================================

def main():
    """Execute and verify the preprocessing workflow."""

    print("=" * 50)
    print("HEARTIFY - DATA PREPROCESSING")
    print("=" * 50)

    # Load dataset
    df = load_data()
    print("\nDataset loaded successfully!")
    print("Dataset shape:", df.shape)

    # Prepare features and target
    X, y = prepare_features(df)
    print("\nFeature matrix shape:", X.shape)
    print("Target vector shape:", y.shape)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining shape:", X_train.shape)
    print("Testing shape:", X_test.shape)

    # Build preprocessing pipeline
    preprocessor = build_preprocessor()

    # Fit only on training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform testing data
    X_test_processed = preprocessor.transform(X_test)

    # Retrieve transformed feature names
    feature_names = preprocessor.get_feature_names_out()

    print("\nProcessed training shape:", X_train_processed.shape)
    print("Processed testing shape:", X_test_processed.shape)
    print("Total processed features:", len(feature_names))

    # Validate processed datasets
    assert X_train_processed.shape[0] == len(y_train)
    assert X_test_processed.shape[0] == len(y_test)

    assert X_train_processed.shape[1] == X_test_processed.shape[1]

    assert np.isfinite(X_train_processed).all()
    assert np.isfinite(X_test_processed).all()

    print("\nMissing values in training data:",
          np.isnan(X_train_processed).sum())

    print("Missing values in testing data:",
          np.isnan(X_test_processed).sum())

    print("\nAll preprocessing validations passed!")
    print("Preprocessing completed successfully!")

    print("=" * 50)


# ==================================================
# 7. ENTRY POINT
# ==================================================

if __name__ == "__main__":
    main()
