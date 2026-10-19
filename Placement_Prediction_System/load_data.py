import os
import pandas as pd


# CSV file path
DATA_FILE = os.path.join(
    os.path.dirname(__file__),
    "placement_predict_50k Dataset (3)(in).csv"
)


def load_data():
    """
    Load the placement dataset.
    """
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE)

    return data


def get_data_summary(data):
    """
    Create summary information required by index.html.
    """

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),

        # List of column names
        "columns": list(data.columns),

        # Data types of each column
        "dtypes": data.dtypes.astype(str).to_dict(),

        # Missing values in each column
        "missing_counts": data.isnull().sum().to_dict(),

        # First 10 rows
        "preview": data.head(10).to_dict(orient="records")
    }