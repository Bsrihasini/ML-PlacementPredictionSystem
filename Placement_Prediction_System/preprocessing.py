import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


TARGET = "Salary Package"


def preprocess_data(data):

    df = data.copy()

    # Remove duplicate rows
    duplicates_before = df.duplicated().sum()
    df = df.drop_duplicates()

    # Remove StudentID because it is only an identifier
    if "StudentID" in df.columns:
        df = df.drop(columns=["StudentID"])

    # Remove anomaly flag from model features
    if "IsAnomaly" in df.columns:
        df = df.drop(columns=["IsAnomaly"])

    # Target column
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in dataset."
        )

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    numeric_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        ))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return {
        "X_train": X_train_processed,
        "X_test": X_test_processed,
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_count": X_train_processed.shape[1],
        "duplicates_removed": int(duplicates_before),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "cleaned_data": df
    }