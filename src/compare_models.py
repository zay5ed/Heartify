
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from pathlib import Path

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
    make_scorer,
    RocCurveDisplay
)

from preprocessing import (
    load_data,
    prepare_features,
    split_data
)


# ==================================================
# 1. PROJECT CONFIGURATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(exist_ok=True)


def main():

    print("=" * 65)
    print("HEARTIFY - MODEL COMPARISON")
    print("=" * 65)

    # ==================================================
    # 2. LOAD DATA
    # ==================================================

    df = load_data()

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("\nTraining records:", len(X_train))
    print("Testing records:", len(X_test))


    # ==================================================
    # 3. REGISTER AVAILABLE MODELS
    # ==================================================

    model_files = {
        "Logistic Regression": "logistic_model.joblib",
        "Random Forest": "random_forest_model.joblib"
    }

    models = {}

    for name, filename in model_files.items():

        models[name] = joblib.load(
            MODEL_DIR / filename
        )

        print(f"Loaded: {name}")


    # ==================================================
    # 4. CROSS-VALIDATION CONFIGURATION
    # ==================================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "Accuracy": "accuracy",
        "Precision": "precision",
        "Recall": "recall",
        "F1": "f1",
        "ROC-AUC": "roc_auc",
        "Specificity": make_scorer(
            recall_score,
            pos_label=0
        )
    }

    cv_summary = []
    test_summary = []
    fold_summary = []

    fig, ax = plt.subplots(figsize=(8, 6))


    # ==================================================
    # 5. EVALUATE EACH MODEL
    # ==================================================

    for name, model in models.items():

        print(f"\nEvaluating {name}...")

        cv_results = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            error_score="raise"
        )

        cv_row = {"Model": name}

        for metric in scoring:

            scores = cv_results[f"test_{metric}"]

            cv_row[metric] = scores.mean()

            fold_summary.append({
                "Model": name,
                "Metric": metric,
                "Mean": scores.mean(),
                "Std": scores.std()
            })

        cv_summary.append(cv_row)


        # Preliminary holdout evaluation

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(X_test)[:, 1]

        test_summary.append({
            "Model": name,

            "Accuracy": accuracy_score(
                y_test, y_pred
            ),

            "Precision": precision_score(
                y_test, y_pred,
                zero_division=0
            ),

            "Recall": recall_score(
                y_test, y_pred,
                zero_division=0
            ),

            "F1": f1_score(
                y_test, y_pred,
                zero_division=0
            ),

            "ROC-AUC": roc_auc_score(
                y_test, y_prob
            ),

            "Specificity": recall_score(
                y_test, y_pred,
                pos_label=0,
                zero_division=0
            )
        })


        # Add model to combined ROC curve

        RocCurveDisplay.from_predictions(
            y_test,
            y_prob,
            name=name,
            ax=ax
        )


    # ==================================================
    # 6. GENERATE COMPARISON TABLES
    # ==================================================

    cv_df = pd.DataFrame(cv_summary).set_index("Model")

    test_df = pd.DataFrame(test_summary).set_index("Model")

    print("\nCROSS-VALIDATION COMPARISON")
    print("-" * 65)

    print(cv_df.round(4).to_string())

    print("\nPRELIMINARY HOLDOUT COMPARISON")
    print("-" * 65)

    print(test_df.round(4).to_string())


    # ==================================================
    # 7. SAVE COMPARISON REPORTS
    # ==================================================

    cv_df.to_csv(
        REPORT_DIR / "model_comparison_cv.csv"
    )

    test_df.to_csv(
        REPORT_DIR / "model_comparison_test.csv"
    )

    pd.DataFrame(fold_summary).to_csv(
        REPORT_DIR / "model_comparison_cv_statistics.csv",
        index=False
    )


    # ==================================================
    # 8. SAVE COMBINED ROC CURVE
    # ==================================================

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    ax.set_title(
        "Heartify - Preliminary Holdout ROC Comparison"
    )

    ax.legend(loc="lower right")

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "combined_roc_curve.png",
        dpi=300
    )

    plt.close(fig)


    # ==================================================
    # 9. PERFORMANCE COMPARISON BAR CHART
    # ==================================================

    ax = cv_df.T.plot(
        kind="bar",
        figsize=(11, 6),
        width=0.75,
        color=["steelblue", "seagreen"]
    )

    ax.set_title(
        "Heartify - 5-Fold Cross-Validation Comparison"
    )

    ax.set_ylabel("Performance Score")
    ax.set_xlabel("Evaluation Metric")
    ax.set_ylim(0, 1.12)

    ax.tick_params(axis="x", rotation=0)

    ax.legend(title="Model")

    for container in ax.containers:

        ax.bar_label(
            container,
            fmt="%.3f",
            fontsize=8,
            padding=3
        )

    fig = ax.get_figure()

    fig.tight_layout()

    fig.savefig(
        REPORT_DIR / "model_comparison.png",
        dpi=300
    )

    plt.close(fig)


    # ==================================================
    # 10. IDENTIFY METRIC LEADERS
    # ==================================================

    print("\nCROSS-VALIDATION METRIC LEADERS")
    print("-" * 65)

    for metric in cv_df.columns:

        best_score = cv_df[metric].max()

        leaders = cv_df.index[
            cv_df[metric] == best_score
        ]

        print(
            f"{metric:12s}: "
            f"{', '.join(leaders)} "
            f"({best_score:.4f})"
        )

    print("\nAll comparison reports saved successfully!")

    print("Academic prototype - not a clinical diagnostic tool.")

    print("=" * 65)


if __name__ == "__main__":
    main()
