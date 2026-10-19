import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import numpy as np


MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "linear_regression_model.pkl"
)


def train_linear_regression(preprocessed):

    os.makedirs(MODEL_DIR, exist_ok=True)

    model = LinearRegression()

    model.fit(
        preprocessed["X_train"],
        preprocessed["y_train"]
    )

    predictions = model.predict(
        preprocessed["X_test"]
    )

    mae = mean_absolute_error(
        preprocessed["y_test"],
        predictions
    )

    mse = mean_squared_error(
        preprocessed["y_test"],
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        preprocessed["y_test"],
        predictions
    )

    joblib.dump(
        {
            "model": model,
            "preprocessor": preprocessed["preprocessor"]
        },
        MODEL_FILE
    )

    return {
        "model": model,
        "predictions": predictions,
        "actual": preprocessed["y_test"],
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2
    }


def load_model():

    if not os.path.exists(MODEL_FILE):
        return None

    return joblib.load(MODEL_FILE)