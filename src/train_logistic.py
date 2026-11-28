
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)

from preprocessing import (
    load_data,
    prepare_features,
    split_data,
    build_preprocessor
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 60)
    print("HEARTIFY - LOGISTIC REGRESSION")
    print("=" * 60)

    # ==================================================
    # 2. LOAD AND PREPARE DATA
    # ==================================================

    df = load_data()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))


    # ==================================================
    # 3. BUILD COMPLETE ML PIPELINE
    # ==================================================

    model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
            C=1.0
        ))
    ])


    # ==================================================
    # 4. STRATIFIED 5-FOLD CROSS-VALIDATION
    # ==================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc"
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    print("\n5-FOLD CROSS-VALIDATION RESULTS")
    print("-" * 60)

    for metric in scoring:
        scores = cv_results[f"test_{metric}"]

        print(
            f"{metric.upper():12s}: "
            f"{scores.mean():.4f} "
            f"(+/- {scores.std():.4f})"
        )


    # ==================================================
    # 5. TRAIN THE MODEL
    # ==================================================

    model.fit(X_train, y_train)

    print("\nModel trained successfully!")


    # ==================================================
    # 6. MAKE PREDICTIONS
    # ==================================================

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # ==================================================
    # 7. EVALUATE MODEL PERFORMANCE
    # ==================================================

    print("\nHOLDOUT TEST RESULTS")
    print("-" * 60)

    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")

    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")

    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")

    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

    print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        y_pred,
        target_names=[
            "Disease Absent",
            "Disease Present"
        ],
        zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix:")
    print(cm)


    # ==================================================
    # 8. GENERATE EVALUATION PLOTS
    # ==================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Absent", "Present"]
    ).plot(ax=ax, cmap="Blues", colorbar=False)

    ax.set_title("Heartify - Logistic Regression")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "logistic_confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)


    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        name="Logistic Regression",
        ax=ax
    )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")

    ax.set_title("Heartify - ROC Curve")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "logistic_roc_curve.png",
        dpi=300
    )

    plt.close(fig)

    print("\nEvaluation plots saved to reports/")


    # ==================================================
    # 9. INSPECT LEARNED COEFFICIENTS
    # ==================================================

    classifier = model.named_steps["classifier"]

    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    coefficients = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": classifier.coef_[0]
    })

    coefficients["Magnitude"] = (
        coefficients["Coefficient"].abs()
    )

    coefficients = coefficients.sort_values(
        by="Magnitude",
        ascending=False
    )

    print("\nTop 10 coefficients by absolute magnitude:")

    print(coefficients.head(10).to_string(index=False))

    print("\nModel intercept:", classifier.intercept_[0])

    coefficients.to_csv(
        REPORT_DIR / "logistic_coefficients.csv",
        index=False
    )


    # ==================================================
    # 10. DEMONSTRATE AN UNSEEN PATIENT PREDICTION
    # ==================================================

    sample_patient = X_test.iloc[[0]]

    actual = y_test.iloc[0]

    prediction = model.predict(sample_patient)[0]

    probability = model.predict_proba(sample_patient)[0, 1]

    print("\nSAMPLE PATIENT PREDICTION")
    print("-" * 60)

    print(sample_patient.T.to_string(header=False))

    print("\nActual class:", actual)

    print("Predicted class:", prediction)

    print(f"Estimated disease probability: {probability:.2%}")

    print(
        "Prediction:",
        "Disease Present" if prediction == 1
        else "Disease Absent"
    )


    # ==================================================
    # 11. SAVE THE COMPLETE TRAINED PIPELINE
    # ==================================================

    model_path = MODEL_DIR / "logistic_model.joblib"

    joblib.dump(model, model_path)

    print("\nModel saved successfully!")

    print("Model path:", model_path)

    print("\nAcademic prototype - not a clinical diagnostic tool.")

    print("=" * 60)


if __name__ == "__main__":
    main()
