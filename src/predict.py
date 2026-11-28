
import joblib
import pandas as pd

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


def load_model(model_name):
    """Load a saved machine learning pipeline."""

    model_path = MODEL_DIR / f"{model_name}_model.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return joblib.load(model_path)


def predict_patient(model, patient_data):
    """Generate predictions for a patient record."""

    patient = pd.DataFrame([patient_data])

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0, 1])

    return {
        "prediction": prediction,
        "probability": probability,
        "label": (
            "Disease Present"
            if prediction == 1
            else "Disease Absent"
        )
    }


def main():

    print("\nHEARTIFY - MULTI-MODEL PREDICTION")
    print("=" * 65)

    patient_data = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 2,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 3,
        "ca": 0,
        "thal": 6
    }

    models = {
        "Logistic Regression": "logistic",
        "Random Forest": "random_forest"
    }

    results = []

    for display_name, model_name in models.items():

        model = load_model(model_name)

        result = predict_patient(model, patient_data)

        results.append({
            "Model": display_name,
            "Class": result["prediction"],
            "Prediction": result["label"],
            "Probability": f"{result['probability']:.2%}"
        })

    results_df = pd.DataFrame(results)

    print("\nPATIENT INPUT")
    print("-" * 65)

    print(pd.Series(patient_data).to_string())

    print("\nMODEL PREDICTION COMPARISON")
    print("-" * 65)

    print(results_df.to_string(index=False))

    if results[0]["Class"] == results[1]["Class"]:
        print("\nBoth models agree on the predicted class.")
    else:
        print("\nThe models disagree on the predicted class.")

    print("\nAcademic prototype - not a medical diagnosis.")
    print("=" * 65)


if __name__ == "__main__":
    main()
