from flask import Flask, render_template, request, redirect, url_for
import os
import pandas as pd
import numpy as np

from load_data import load_data, get_data_summary
from placement_eda import run_eda
from preprocessing import preprocess_data
from visualization import create_model_visualizations
from analysis.correlation import run_correlation_analysis
from models.linear_regression import train_linear_regression, get_regression_targets
from models.model_training import MODEL_TYPES, get_target_options, train_model, STORAGE_DIR

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def dashboard_stats(df):
    rows = len(df)
    columns = len(df.columns)
    missing = int(df.isnull().sum().sum())
    duplicates = int(df.duplicated().sum())

    placed = 0
    if "PlacementStatus" in df.columns:
        status = df["PlacementStatus"].astype(str).str.strip().str.lower()
        placed = int(status.isin(["placed", "yes", "true", "1"]).sum())

    not_placed = max(rows - placed, 0)

    return {
        "rows": rows,
        "columns": columns,
        "missing": missing,
        "duplicates": duplicates,
        "placed": placed,
        "not_placed": not_placed,
    }


@app.route("/")
def index():
    try:
        df = load_data()
        stats = dashboard_stats(df)
        return render_template("dashboard.html", active="dashboard", **stats, error=None)
    except Exception as e:
        return render_template(
            "dashboard.html",
            active="dashboard",
            rows=0,
            columns=0,
            missing=0,
            duplicates=0,
            placed=0,
            not_placed=0,
            error=str(e),
        )


@app.route("/data-loading")
def data_loading():
    error = None
    summary = None
    try:
        df = load_data()
        summary = get_data_summary(df)
    except FileNotFoundError as e:
        error = str(e)
    except Exception as e:
        error = f"Unexpected error: {e}"

    return render_template(
        "index.html",
        active="data-loading",
        summary=summary,
        error=error,
        uploaded=False,
    )


@app.route("/upload", methods=["POST"])
def upload_dataset():
    if "dataset" not in request.files:
        return redirect(url_for("data_loading"))

    file = request.files["dataset"]

    if not file.filename:
        return redirect(url_for("data_loading"))

    if not allowed_file(file.filename):
        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            uploaded=False,
            error="Only CSV, XLSX and XLS files are allowed.",
        )

    filename = os.path.basename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)

    try:
        file.save(filepath)

        if filename.lower().endswith(".csv"):
            df = pd.read_csv(filepath)
        else:
            df = pd.read_excel(filepath)

        summary = get_data_summary(df)

        return render_template(
            "index.html",
            active="data-loading",
            summary=summary,
            uploaded=True,
            uploaded_filename=filename,
            error=None,
        )
    except Exception as e:
        return render_template(
            "index.html",
            active="data-loading",
            summary=None,
            uploaded=False,
            error=f"Error uploading dataset: {e}",
        )


@app.route("/eda")
def eda_page():
    try:
        df = load_data()
        results = run_eda()

        missing = df.isnull().sum()
        missing = missing[missing > 0]

        numeric_columns = df.select_dtypes(include="number").columns.tolist()
        categorical_columns = df.select_dtypes(exclude="number").columns.tolist()

        target_counts = df["PlacementStatus"].astype(str).value_counts() if "PlacementStatus" in df.columns else pd.Series(dtype=int)

        cgpa_values = pd.to_numeric(df["CGPA"], errors="coerce").dropna() if "CGPA" in df.columns else pd.Series(dtype=float)

        return render_template(
            "eda.html",
            active="eda",
            results=results,
            rows=len(df),
            columns=len(df.columns),
            duplicates=int(df.duplicated().sum()),
            missing_total=int(df.isnull().sum().sum()),
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            placement_labels=target_counts.index.astype(str).tolist(),
            placement_values=target_counts.values.astype(int).tolist(),
            cgpa_labels=[],
            cgpa_values=[],
            missing_labels=missing.index.tolist(),
            missing_values=missing.values.astype(int).tolist(),
            cgpa_data=cgpa_values.head(1000).tolist(),
            error=None,
        )
    except Exception as e:
        return render_template(
            "eda.html",
            active="eda",
            results=None,
            rows=0,
            columns=0,
            duplicates=0,
            missing_total=0,
            numeric_columns=[],
            categorical_columns=[],
            placement_labels=[],
            placement_values=[],
            cgpa_labels=[],
            cgpa_values=[],
            missing_labels=[],
            missing_values=[],
            cgpa_data=[],
            error=str(e),
        )


@app.route("/preprocessing")
def preprocessing_page():
    try:
        df = load_data()
        results = preprocess_data(df)
        results["feature_names"] = []

        try:
            names = results["preprocessor"].get_feature_names_out()
            results["feature_names"] = names.tolist()
        except Exception:
            pass

        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=results,
            error=None,
        )
    except Exception as e:
        return render_template(
            "preprocessing.html",
            active="preprocessing",
            results=None,
            error=str(e),
        )


@app.route("/visualization")
def visualization_page():
    try:
        df = load_data()
        total_students = len(df)

        placement_labels, placement_values = [], []
        if "PlacementStatus" in df.columns:
            counts = df["PlacementStatus"].astype(str).value_counts()
            placement_labels = counts.index.tolist()
            placement_values = counts.values.astype(int).tolist()

        placed_students = 0
        if "PlacementStatus" in df.columns:
            status = df["PlacementStatus"].astype(str).str.strip().str.lower()
            placed_students = int(status.isin(["placed", "yes", "true", "1"]).sum())

        placement_rate = round((placed_students / total_students) * 100, 2) if total_students else 0

        average_cgpa = 0
        if "CGPA" in df.columns:
            cgpa = pd.to_numeric(df["CGPA"], errors="coerce")
            if not cgpa.dropna().empty:
                average_cgpa = round(float(cgpa.mean()), 2)

        cgpa_labels, cgpa_values = [], []
        if "PlacementStatus" in df.columns and "CGPA" in df.columns:
            temp = df[["PlacementStatus", "CGPA"]].copy()
            temp["CGPA"] = pd.to_numeric(temp["CGPA"], errors="coerce")
            group = temp.groupby("PlacementStatus")["CGPA"].mean().dropna()
            cgpa_labels = group.index.astype(str).tolist()
            cgpa_values = [round(float(x), 2) for x in group.values]

        salary_labels, salary_values = [], []
        if "Salary Package" in df.columns:
            salary = pd.to_numeric(df["Salary Package"], errors="coerce").dropna()
            if not salary.empty:
                bins = [-0.001, 3, 5, 7, 10, 15, 20, float("inf")]
                labels = ["0-3 LPA", "3-5 LPA", "5-7 LPA", "7-10 LPA", "10-15 LPA", "15-20 LPA", "20+ LPA"]
                grouped = pd.cut(salary, bins=bins, labels=labels, include_lowest=True).value_counts().sort_index()
                salary_labels = grouped.index.astype(str).tolist()
                salary_values = grouped.values.astype(int).tolist()

        city_labels, city_values = [], []
        if "City" in df.columns:
            cities = df["City"].astype(str).value_counts().head(10)
            city_labels = cities.index.tolist()
            city_values = cities.values.astype(int).tolist()

        scatter_x, scatter_y = [], []
        if "AttendancePercent" in df.columns and "CGPA" in df.columns:
            temp = df[["AttendancePercent", "CGPA"]].copy()
            temp["AttendancePercent"] = pd.to_numeric(temp["AttendancePercent"], errors="coerce")
            temp["CGPA"] = pd.to_numeric(temp["CGPA"], errors="coerce")
            temp = temp.dropna().head(1000)
            scatter_x = temp["AttendancePercent"].tolist()
            scatter_y = temp["CGPA"].tolist()

        coding_x, coding_y = [], []
        if "CodingTestScore" in df.columns and "CGPA" in df.columns:
            temp = df[["CodingTestScore", "CGPA"]].copy()
            temp["CodingTestScore"] = pd.to_numeric(temp["CodingTestScore"], errors="coerce")
            temp["CGPA"] = pd.to_numeric(temp["CGPA"], errors="coerce")
            temp = temp.dropna().head(1000)
            coding_x = temp["CodingTestScore"].tolist()
            coding_y = temp["CGPA"].tolist()

        softskills_labels, softskills_values = [], []
        if "SoftSkillsRating" in df.columns:
            values = pd.to_numeric(df["SoftSkillsRating"], errors="coerce").dropna().value_counts().sort_index()
            softskills_labels = [str(x) for x in values.index]
            softskills_values = values.values.astype(int).tolist()

        return render_template(
            "visualization.html",
            active="visualization",
            error=None,
            total_students=total_students,
            placed_students=placed_students,
            placement_rate=placement_rate,
            average_cgpa=average_cgpa,
            placement_labels=placement_labels,
            placement_values=placement_values,
            cgpa_labels=cgpa_labels,
            cgpa_values=cgpa_values,
            salary_labels=salary_labels,
            salary_values=salary_values,
            city_labels=city_labels,
            city_values=city_values,
            scatter_x=scatter_x,
            scatter_y=scatter_y,
            coding_x=coding_x,
            coding_y=coding_y,
            softskills_labels=softskills_labels,
            softskills_values=softskills_values,
        )
    except Exception as e:
        return render_template(
            "visualization.html",
            active="visualization",
            error=f"Visualization Error: {e}",
            total_students=0,
            placed_students=0,
            placement_rate=0,
            average_cgpa=0,
            placement_labels=[],
            placement_values=[],
            cgpa_labels=[],
            cgpa_values=[],
            salary_labels=[],
            salary_values=[],
            city_labels=[],
            city_values=[],
            scatter_x=[],
            scatter_y=[],
            coding_x=[],
            coding_y=[],
            softskills_labels=[],
            softskills_values=[],
        )


@app.route("/correlation")
def correlation_page():
    try:
        df = load_data()
        results = run_correlation_analysis(df)

        return render_template(
            "correlation.html",
            active="correlation",
            results=results,
            error=None,
        )
    except Exception as e:
        return render_template(
            "correlation.html",
            active="correlation",
            results=None,
            error=f"Correlation analysis error: {e}",
        )
@app.route("/regression", methods=["GET", "POST"])
def regression_page():

    results = None
    error = None
    target_options = []
    selected_target = "Salary Package"

    try:
        # Load dataset
        df = load_data()

        # Get numeric target columns
        target_options = get_regression_targets(df)

        if not target_options:
            raise ValueError(
                "No continuous numeric target is available for Linear Regression."
            )

        # Get selected target
        selected_target = request.form.get(
            "target",
            "Salary Package"
        )

        # Set default target
        if selected_target not in target_options:
            if "Salary Package" in target_options:
                selected_target = "Salary Package"
            else:
                selected_target = target_options[0]

        # Train only when button is clicked
        if request.method == "POST":

            results = train_linear_regression(
                df,
                target=selected_target
            )

            # Check training success
            if not results.get("success", False):
                raise ValueError(
                    results.get(
                        "error",
                        "Linear Regression training failed."
                    )
                )

            # Convert Python lists to NumPy arrays
            # This fixes: unsupported operand type(s) for -: 'list' and 'list'
            actual = np.array(
                results["actual"],
                dtype=float
            )

            predictions = np.array(
                results["predictions"],
                dtype=float
            )

            # Create regression visualizations
            create_model_visualizations(
                actual,
                predictions
            )

    except Exception as e:
        error = f"Regression error: {e}"

    return render_template(
        "regression.html",
        active="regression",
        results=results,
        target_options=target_options,
        selected_target=selected_target,
        error=error,
    )
@app.route("/models", methods=["GET", "POST"])
@app.route("/model-training", methods=["GET", "POST"])
def model_page():
    """Configure and train the ML models shown on the Models page."""
    selected_model = request.form.get("model_name", "") if request.method == "POST" else ""
    selected_target = request.form.get("target", "") if request.method == "POST" else ""
    selected_features = request.form.getlist("features") if request.method == "POST" else []
    results = None
    error = None
    columns = []

    try:
        df = load_data()
        columns = [c for c in df.columns if c != "StudentID"]

        if request.method == "POST":
            if not selected_model:
                raise ValueError("Please select a machine learning model.")

            # Unsupervised models do not need a target variable.
            unsupervised = selected_model in {
                "K-Means", "Hierarchical Clustering",
                "Agglomerative Clustering", "DBSCAN",
                "PCA - Principal Component Analysis"
            }

            if not unsupervised and not selected_target:
                # Use a sensible automatic default instead of blocking the run.
                if selected_model in {"Linear Regression", "Multiple Linear Regression", "Ridge Regression", "Lasso Regression"}:
                    numeric_targets = [c for c in df.select_dtypes(include=np.number).columns
                                       if c not in {"StudentID", "PlacementStatus"}]
                    selected_target = "Salary Package" if "Salary Package" in numeric_targets else (numeric_targets[0] if numeric_targets else "")
                elif "PlacementStatus" in df.columns:
                    selected_target = "PlacementStatus"

            if not unsupervised and not selected_target:
                raise ValueError("Please select a target variable.")

            if not selected_features:
                selected_features = [c for c in columns if c != selected_target]

            results = train_model(
                df=df,
                model_name=selected_model,
                target_column=selected_target if not unsupervised else None,
                feature_columns=selected_features,
            )

            if not results.get("success", False):
                raise ValueError(results.get("error", "Model training failed."))

    except Exception as e:
        error = str(e)

    return render_template(
        "model.html",
        active="models",
        model_options=MODEL_TYPES,
        columns=columns,
        selected_model=selected_model,
        selected_target=selected_target,
        selected_features=selected_features,
        results=results,
        error=error,
    )


@app.route("/model-storage")
def model_storage_page():
    files = []
    try:
        for name in sorted(os.listdir(STORAGE_DIR), reverse=True):
            if name.endswith(".pkl"):
                files.append(name)
    except Exception:
        pass
    return render_template(
        "model_storage.html", active="storage", files=files
    )


if __name__ == "__main__":
    app.run(debug=True)
