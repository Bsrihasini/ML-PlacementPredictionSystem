import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


CHART_DIR = os.path.join(
    os.path.dirname(__file__),
    "static",
    "charts"
)


def create_model_visualizations(actual, predictions):

    os.makedirs(CHART_DIR, exist_ok=True)

    # Actual vs Predicted
    plt.figure(figsize=(8, 6))

    sns.scatterplot(
        x=actual,
        y=predictions
    )

    plt.xlabel("Actual Salary")
    plt.ylabel("Predicted Salary")
    plt.title("Actual vs Predicted Salary")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "actual_vs_predicted.png"
        )
    )

    plt.close()

    # Residuals
    residuals = actual - predictions

    plt.figure(figsize=(8, 6))

    sns.histplot(
        residuals,
        kde=True
    )

    plt.xlabel("Residual")
    plt.title("Residual Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            CHART_DIR,
            "residual_distribution.png"
        )
    )

    plt.close()

    return [
        "actual_vs_predicted.png",
        "residual_distribution.png"
    ]