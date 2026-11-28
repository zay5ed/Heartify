
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate
)

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
    print("HEARTIFY - RANDOM FOREST")
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
    # 3. BUILD RANDOM FOREST PIPELINE
    # ==================================================

    model = Pipeline(steps=[
        ("preprocessor", build_preprocessor()),

        ("classifier", RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            bootstrap=True,
            random_state=42,
            n_jobs=-1
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

    print("\nRunning cross-validation...")

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring
    )

    print("\n5-FOLD CROSS-VALIDATION RESULTS")
    print("-" * 60)

    cv_summary = []

    for metric in scoring:

        scores = cv_results[f"test_{metric}"]

        print(
            f"{metric.upper():12s}: "
            f"{scores.mean():.4f} "
            f"(+/- {scores.std():.4f})"
        )

        cv_summary.append({
            "Metric": metric,
            "Mean": scores.mean(),
            "Std": scores.std()
        })

    pd.DataFrame(cv_summary).to_csv(
        REPORT_DIR / "random_forest_cv_results.csv",
        index=False
    )


    # ==================================================
    # 5. TRAIN RANDOM FOREST
    # ==================================================

    print("\nTraining Random Forest...")

    model.fit(X_train, y_train)

    print("Model trained successfully!")


    # ==================================================
    # 6. GENERATE PREDICTIONS
    # ==================================================

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]


    # ==================================================
    # 7. EVALUATE MODEL
    # ==================================================

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),

        "precision": precision_score(
            y_test, y_pred, zero_division=0
        ),

        "recall": recall_score(
            y_test, y_pred, zero_division=0
        ),

        "f1": f1_score(
            y_test, y_pred, zero_division=0
        ),

        "roc_auc": roc_auc_score(y_test, y_prob)
    }

    print("\nPRELIMINARY HOLDOUT TEST RESULTS")
    print("-" * 60)

    for metric, value in metrics.items():
        print(f"{metric.upper():12s}: {value:.4f}")

    pd.DataFrame([metrics]).to_csv(
        REPORT_DIR / "random_forest_test_results.csv",
        index=False
    )

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
    # 8. CONFUSION MATRIX PLOT
    # ==================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Absent", "Present"]
    ).plot(
        ax=ax,
        cmap="Greens",
        colorbar=False
    )

    ax.set_title("Heartify - Random Forest")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "random_forest_confusion_matrix.png",
        dpi=300
    )

    plt.close(fig)


    # ==================================================
    # 9. ROC CURVE
    # ==================================================

    fig, ax = plt.subplots(figsize=(6, 5))

    RocCurveDisplay.from_predictions(
        y_test,
        y_prob,
        name="Random Forest",
        ax=ax
    )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray"
    )

    ax.set_title("Heartify - Random Forest ROC Curve")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "random_forest_roc_curve.png",
        dpi=300
    )

    plt.close(fig)


    # ==================================================
    # 10. FEATURE IMPORTANCE
    # ==================================================

    classifier = model.named_steps["classifier"]

    preprocessor = model.named_steps["preprocessor"]

    feature_names = preprocessor.get_feature_names_out()

    importance = pd.DataFrame({
        "Feature": feature_names,
        "Importance": classifier.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTOP 10 IMPORTANT FEATURES")
    print("-" * 60)

    print(importance.head(10).to_string(index=False))

    importance.to_csv(
        REPORT_DIR / "random_forest_feature_importance.csv",
        index=False
    )

    top_features = importance.head(10).sort_values(
        by="Importance"
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        top_features["Feature"],
        top_features["Importance"],
        color="seagreen"
    )

    ax.set_xlabel("Impurity-based Feature Importance")

    ax.set_title("Heartify - Random Forest Feature Importance")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "random_forest_feature_importance.png",
        dpi=300
    )

    plt.close(fig)


    # ==================================================
    # 11. SAVE TRAINED PIPELINE
    # ==================================================

    model_path = MODEL_DIR / "random_forest_model.joblib"

    joblib.dump(model, model_path)

    print("\nModel saved successfully!")

    print("Model path:", model_path)

    print("Evaluation reports saved to reports/")

    print("\nAcademic prototype - not a medical diagnosis.")

    print("=" * 60)


if __name__ == "__main__":
    main()
