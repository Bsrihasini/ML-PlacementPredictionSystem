import pandas as pd


def run_correlation_analysis(df, target_column=None):
    """
    Performs correlation analysis on numeric columns.
    """

    # Select only numeric columns
    numeric_df = df.select_dtypes(include=["number"])

    # Calculate correlation matrix
    correlation_matrix = numeric_df.corr()

    result = {
        "correlation_matrix": correlation_matrix,
        "strong_positive": [],
        "strong_negative": [],
        "target_correlations": {}
    }

    # Find strong correlations
    for column in correlation_matrix.columns:
        for row in correlation_matrix.index:
            value = correlation_matrix.loc[row, column]

            if column != row:
                if value >= 0.7:
                    pair = {
                        "feature1": row,
                        "feature2": column,
                        "correlation": round(float(value), 3)
                    }

                    if pair not in result["strong_positive"]:
                        result["strong_positive"].append(pair)

                elif value <= -0.7:
                    pair = {
                        "feature1": row,
                        "feature2": column,
                        "correlation": round(float(value), 3)
                    }

                    if pair not in result["strong_negative"]:
                        result["strong_negative"].append(pair)

    # Correlation with target column
    if target_column and target_column in correlation_matrix.columns:

        target_corr = correlation_matrix[target_column].sort_values(
            ascending=False
        )

        result["target_correlations"] = {
            column: round(float(value), 3)
            for column, value in target_corr.items()
            if column != target_column
        }

    return result