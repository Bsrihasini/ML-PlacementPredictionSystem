import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def get_regression_targets(df):
    """
    Return columns suitable for Linear Regression.

    Detects:
    - Already numeric columns
    - Columns containing numeric values stored as strings

    Excludes PlacementStatus because it is a classification target.
    """

    targets = []

    for column in df.columns:

        # Explicitly exclude classification outcome
        if column.lower() in [
            "placementstatus",
            "placement_status",
            "placement"
        ]:
            continue

        # Check if already numeric
        if pd.api.types.is_numeric_dtype(df[column]):
            targets.append(column)
            continue

        # Try converting string/object columns to numeric
        converted = pd.to_numeric(
            df[column],
            errors="coerce"
        )

        # If most values are numeric, allow as regression target
        valid_ratio = converted.notna().sum() / len(df)

        if valid_ratio >= 0.8:
            targets.append(column)

    return targets


def train_linear_regression(df, target=None, target_column=None):
    """
    Train Linear Regression model.

    Supports both:
        train_linear_regression(df, target="CGPA")

    and:
        train_linear_regression(df, target_column="CGPA")
    """

    try:

        # ==========================================
        # GET TARGET
        # ==========================================

        if target is None:
            target = target_column

        if target is None or target == "":
            return {
                "success": False,
                "error": "Please select a target variable."
            }

        if target not in df.columns:
            return {
                "success": False,
                "error": f"Target column '{target}' does not exist."
            }

        # ==========================================
        # COPY DATA
        # ==========================================

        data = df.copy()

        # ==========================================
        # CONVERT TARGET TO NUMERIC
        # ==========================================

        data[target] = pd.to_numeric(
            data[target],
            errors="coerce"
        )

        # Remove rows where target is missing
        data = data.dropna(subset=[target])

        # ==========================================
        # PREPARE FEATURES
        # ==========================================

        X = data.drop(columns=[target])
        y = data[target]

        # Convert categorical columns using One-Hot Encoding
        X = pd.get_dummies(
            X,
            drop_first=True
        )

        # Convert all values to numeric
        X = X.apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace infinity
        X = X.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Fill missing values with median
        for column in X.columns:

            median_value = X[column].median()

            if pd.isna(median_value):
                median_value = 0

            X[column] = X[column].fillna(
                median_value
            )

        # ==========================================
        # VALIDATION
        # ==========================================

        if X.shape[1] == 0:
            return {
                "success": False,
                "error": "No feature columns available."
            }

        if len(X) < 10:
            return {
                "success": False,
                "error": "Not enough data available for training."
            }

        # ==========================================
        # TRAIN TEST SPLIT
        # ==========================================

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ==========================================
        # CREATE MODEL
        # ==========================================

        model = LinearRegression()

        # ==========================================
        # TRAIN
        # ==========================================

        model.fit(
            X_train,
            y_train
        )

        # ==========================================
        # PREDICTIONS
        # ==========================================

        predictions = model.predict(X_test)

        # ==========================================
        # METRICS
        # ==========================================

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        mse = mean_squared_error(
            y_test,
            predictions
        )

        rmse = np.sqrt(mse)

        r2 = r2_score(
            y_test,
            predictions
        )

        # ==========================================
        # COEFFICIENTS
        # ==========================================

        coefficients = {}

        for feature, coefficient in zip(
            X.columns,
            model.coef_
        ):

            coefficients[feature] = round(
                float(coefficient),
                6
            )

        # Convert arrays to normal Python lists
        actual_values = [
            round(float(value), 4)
            for value in y_test.tolist()
        ]

        predicted_values = [
            round(float(value), 4)
            for value in predictions.tolist()
        ]

        # ==========================================
        # RETURN RESULTS
        # ==========================================

        return {
            "success": True,

            "target": target,

            # Metrics
            "MAE": round(float(mae), 4),
            "MSE": round(float(mse), 4),
            "RMSE": round(float(rmse), 4),
            "R2": round(float(r2), 4),
            "R2 Score": round(float(r2), 4),

            # Model details
            "Intercept": round(
                float(model.intercept_),
                6
            ),

            "Coefficients": coefficients,

            "features_used": X.columns.tolist(),

            "total_features": len(X.columns),

            "training_samples": len(X_train),

            "testing_samples": len(X_test),

            # Actual values
            "actual": actual_values,

            # Multiple aliases for compatibility
            "actual_values": actual_values,

            # Predictions
            "predictions": predicted_values,
            "predicted": predicted_values,
            "predicted_values": predicted_values
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }