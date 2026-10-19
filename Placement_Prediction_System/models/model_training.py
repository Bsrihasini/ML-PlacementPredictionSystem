import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    confusion_matrix
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STORAGE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "trained_models")
CHART_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "charts", "models")
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

MODEL_TYPES = [
    "Logistic Regression", "Linear Regression", "Multiple Linear Regression", "Ridge Regression", "Lasso Regression",
    "Decision Tree - ID3", "Decision Tree - C4.5", "Decision Tree - CART", "Bagging",
    "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM",
    "K-Means", "Hierarchical Clustering", "Agglomerative Clustering", "DBSCAN", "PCA - Principal Component Analysis",
]

REGRESSION_MODELS = {"Linear Regression", "Multiple Linear Regression", "Ridge Regression", "Lasso Regression"}
CLASSIFICATION_MODELS = {"Logistic Regression", "Decision Tree - ID3", "Decision Tree - C4.5", "Decision Tree - CART", "Bagging", "AdaBoost", "Gradient Boosting", "XGBoost", "LightGBM"}
UNSUPERVISED_MODELS = {"K-Means", "Hierarchical Clustering", "Agglomerative Clustering", "DBSCAN", "PCA - Principal Component Analysis"}

MODEL_ATTRIBUTES = {
    "Linear Regression": ["fit_intercept", "copy_X", "n_jobs", "coefficients", "intercept"],
    "Multiple Linear Regression": ["fit_intercept", "copy_X", "n_jobs", "multiple predictors", "coefficients"],
    "Ridge Regression": ["alpha", "fit_intercept", "solver", "max_iter", "coefficients"],
    "Lasso Regression": ["alpha", "fit_intercept", "max_iter", "tol", "selection"],
    "Logistic Regression": ["C", "solver", "max_iter", "class_weight", "intercept"],
    "Decision Tree - ID3": ["criterion=entropy", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"],
    "Decision Tree - C4.5": ["criterion=entropy", "splitter=best", "max_depth", "min_samples_split", "min_samples_leaf"],
    "Decision Tree - CART": ["criterion=gini", "max_depth", "min_samples_split", "min_samples_leaf", "random_state"],
    "Bagging": ["n_estimators", "max_samples", "max_features", "bootstrap", "random_state"],
    "AdaBoost": ["n_estimators", "learning_rate", "estimator", "random_state"],
    "Gradient Boosting": ["n_estimators", "learning_rate", "max_depth", "subsample", "random_state"],
    "XGBoost": ["n_estimators", "learning_rate", "max_depth", "subsample", "colsample_bytree"],
    "LightGBM": ["n_estimators", "learning_rate", "num_leaves", "max_depth", "subsample"],
    "K-Means": ["n_clusters", "init", "n_init", "max_iter", "random_state"],
    "Hierarchical Clustering": ["n_clusters", "linkage", "metric", "distance_threshold"],
    "Agglomerative Clustering": ["n_clusters", "linkage", "metric", "compute_full_tree"],
    "DBSCAN": ["eps", "min_samples", "metric", "algorithm"],
    "PCA - Principal Component Analysis": ["n_components", "whiten", "svd_solver", "tol", "random_state"],
}


def get_target_options(df, model_name=None):
    if model_name in UNSUPERVISED_MODELS:
        return []
    if model_name in CLASSIFICATION_MODELS:
        return [c for c in df.columns if c not in {"StudentID"}]
    return [c for c in df.select_dtypes(include=np.number).columns if c not in {"StudentID", "PlacementStatus"}]


def _prepare_features(df, target=None):
    cols = [c for c in df.columns if c not in {"StudentID", target}]
    X = pd.get_dummies(df[cols], dummy_na=True)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True)).fillna(0)
    return X


def _plot_regression(y_test, pred, slug):
    path = os.path.join(CHART_DIR, slug + "_actual_predicted.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(y_test, pred, s=10, alpha=.45)
    lo, hi = float(min(y_test.min(), pred.min())), float(max(y_test.max(), pred.max()))
    plt.plot([lo, hi], [lo, hi], linewidth=2)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    return path


def _plot_classification(y_test, pred, slug):
    path = os.path.join(CHART_DIR, slug + "_confusion_matrix.png")
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    return path


def _plot_clusters(X2, labels, slug, title):
    path = os.path.join(CHART_DIR, slug + "_clusters.png")
    plt.figure(figsize=(6, 4))
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=8, alpha=.55)
    plt.xlabel("Component 1"); plt.ylabel("Component 2"); plt.title(title)
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    return path


def _plot_pca(explained, slug):
    path = os.path.join(CHART_DIR, slug + "_variance.png")
    plt.figure(figsize=(6, 4))
    x = np.arange(1, len(explained) + 1)
    plt.bar(x, explained)
    plt.xlabel("Principal Component"); plt.ylabel("Explained Variance Ratio")
    plt.title("PCA Explained Variance")
    plt.tight_layout(); plt.savefig(path, dpi=130); plt.close()
    return path


def _classifier(model_name):
    if model_name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=42)
    if model_name == "Decision Tree - ID3":
        return DecisionTreeClassifier(criterion="entropy", random_state=42)
    if model_name == "Decision Tree - C4.5":
        return DecisionTreeClassifier(criterion="entropy", splitter="best", random_state=42)
    if model_name == "Decision Tree - CART":
        return DecisionTreeClassifier(criterion="gini", random_state=42)
    if model_name == "Bagging":
        return BaggingClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    if model_name == "AdaBoost":
        return AdaBoostClassifier(n_estimators=100, learning_rate=0.5, random_state=42)
    if model_name == "Gradient Boosting":
        return GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    if model_name == "XGBoost":
        if XGBClassifier is None: raise ValueError("XGBoost is not installed. Run: py -m pip install xgboost")
        return XGBClassifier(n_estimators=100, max_depth=4, learning_rate=.08, subsample=.9, colsample_bytree=.9, eval_metric="logloss", random_state=42)
    if model_name == "LightGBM":
        if LGBMClassifier is None: raise ValueError("LightGBM is not installed. Run: py -m pip install lightgbm")
        return LGBMClassifier(n_estimators=100, learning_rate=.08, num_leaves=31, max_depth=-1, random_state=42, verbosity=-1)
    raise ValueError("Unsupported classification model")


def train_model(df, model_name, target_column=None, feature_columns=None, **kwargs):
    try:
        if model_name not in MODEL_TYPES:
            raise ValueError("Please select a valid machine learning model.")
        safe_name = "".join(ch.lower() if ch.isalnum() else "_" for ch in model_name).strip("_")
        chart = None
        metrics = {}
        calculations = []
        category = "Unsupervised" if model_name in UNSUPERVISED_MODELS else ("Regression" if model_name in REGRESSION_MODELS else "Classification")

        if category == "Unsupervised":
            X = _prepare_features(df)
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            if len(Xs) > 8000:
                rng = np.random.RandomState(42); idx = rng.choice(len(Xs), 8000, replace=False)
                work = Xs[idx]
            else:
                work = Xs
            pca2 = PCA(n_components=2, random_state=42)
            X2 = pca2.fit_transform(work)

            if model_name == "PCA - Principal Component Analysis":
                pca = PCA(n_components=min(10, Xs.shape[1]), random_state=42)
                pca.fit(Xs)
                explained = pca.explained_variance_ratio_
                metrics = {"Components": int(len(explained)), "Variance Explained": round(float(explained.sum() * 100), 2) }
                calculations = [f"Variance explained = Σ component ratios × 100 = {explained.sum()*100:.2f}%"]
                chart = _plot_pca(explained, safe_name)
                target = "Not required"
                result = {"success": True, "model": model_name, "category": category, "target": target, "features": X.shape[1], "training_samples": len(X), "testing_samples": 0, "metrics": metrics, "calculations": calculations, "attributes": MODEL_ATTRIBUTES[model_name], "chart": os.path.basename(chart) if chart else None, "chart_type": "pca"}
            else:
                if model_name == "K-Means":
                    model = KMeans(n_clusters=3, n_init=10, max_iter=300, random_state=42)
                    labels = model.fit_predict(work)
                    metrics["Silhouette Score"] = round(float(silhouette_score(work, labels)), 4)
                    metrics["Inertia"] = round(float(model.inertia_), 2)
                    calculations = [f"Silhouette = mean((b-a)/max(a,b)) = {metrics['Silhouette Score']}", f"Inertia = Σ squared distance to centroid = {metrics['Inertia']}"]
                elif model_name in {"Hierarchical Clustering", "Agglomerative Clustering"}:
                    model = AgglomerativeClustering(n_clusters=3, linkage="ward")
                    labels = model.fit_predict(work)
                    metrics["Silhouette Score"] = round(float(silhouette_score(work, labels)), 4)
                    metrics["Davies-Bouldin"] = round(float(davies_bouldin_score(work, labels)), 4)
                    calculations = [f"Silhouette = {metrics['Silhouette Score']}", f"Davies-Bouldin = {metrics['Davies-Bouldin']} (computed from cluster compactness/separation)"]
                else:
                    model = DBSCAN(eps=0.8, min_samples=8)
                    labels = model.fit_predict(work)
                    valid = labels != -1
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    metrics["Clusters"] = int(max(n_clusters, 0))
                    metrics["Noise Points"] = int((labels == -1).sum())
                    if n_clusters >= 2 and valid.sum() > n_clusters:
                        metrics["Silhouette Score"] = round(float(silhouette_score(work[valid], labels[valid])), 4)
                        calculations = [f"Clusters = unique labels excluding noise = {n_clusters}", f"Noise points = label -1 count = {metrics['Noise Points']}", f"Silhouette = {metrics['Silhouette Score']}"]
                    else:
                        calculations = [f"Clusters = {n_clusters}", f"Noise points = {metrics['Noise Points']}"]
                chart = _plot_clusters(X2, labels, safe_name, model_name)
                joblib.dump({"model": model, "features": X.columns.tolist()}, os.path.join(STORAGE_DIR, safe_name + ".pkl"))
                result = {"success": True, "model": model_name, "category": category, "target": "Not required", "features": X.shape[1], "training_samples": len(X), "testing_samples": 0, "metrics": metrics, "calculations": calculations, "attributes": MODEL_ATTRIBUTES[model_name], "chart": os.path.basename(chart) if chart else None, "chart_type": "cluster"}
            return result

        if not target_column:
            target_column = "PlacementStatus" if category == "Classification" and "PlacementStatus" in df.columns else "Salary Package"
        if target_column not in df.columns:
            raise ValueError("Selected target column does not exist.")

        data = df.dropna(subset=[target_column]).copy()
        X = _prepare_features(data, target_column)
        y = data[target_column].copy()
        if category == "Regression":
            y = pd.to_numeric(y, errors="coerce")
            valid = y.notna(); X, y = X.loc[valid], y.loc[valid]
        else:
            y = y.astype(str).fillna("Unknown")
            y = pd.Series(LabelEncoder().fit_transform(y), index=X.index)
        if len(X) < 10 or y.nunique() < 2:
            raise ValueError("Not enough valid data for this model.")
        stratify = y if category == "Classification" and y.value_counts().min() >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42, stratify=stratify)

        if category == "Regression":
            if model_name == "Ridge Regression": model = Ridge(alpha=1.0)
            elif model_name == "Lasso Regression": model = Lasso(alpha=.01, max_iter=5000)
            else: model = LinearRegression()
        else:
            model = _classifier(model_name)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        if category == "Regression":
            mse = mean_squared_error(y_test, pred); rmse = np.sqrt(mse)
            metrics = {"MAE": round(float(mean_absolute_error(y_test, pred)), 2), "MSE": round(float(mse), 2), "RMSE": round(float(rmse), 2), "R²": round(float(r2_score(y_test, pred)), 4)}
            calculations = [f"MAE = Σ|actual − predicted| / n = {metrics['MAE']}", f"RMSE = √(Σ(actual − predicted)² / n) = {metrics['RMSE']}", f"R² = 1 − SSE/SST = {metrics['R²']}"]
            chart = _plot_regression(y_test, pred, safe_name)
            chart_type = "regression"
        else:
            acc = accuracy_score(y_test, pred) * 100
            metrics = {"Accuracy": round(float(acc), 2), "Correct Predictions": int((y_test.to_numpy() == pred).sum()), "Test Samples": int(len(y_test))}
            calculations = [f"Accuracy = correct predictions / test samples × 100 = {metrics['Accuracy']}%", f"Correct = {metrics['Correct Predictions']} / {metrics['Test Samples']}"]
            chart = _plot_classification(y_test, pred, safe_name)
            chart_type = "classification"

        model_path = os.path.join(STORAGE_DIR, safe_name + ".pkl")
        joblib.dump({"model": model, "features": X.columns.tolist()}, model_path)
        return {"success": True, "model": model_name, "category": category, "target": target_column, "features": X.shape[1], "training_samples": len(X_train), "testing_samples": len(X_test), "metrics": metrics, "calculations": calculations, "attributes": MODEL_ATTRIBUTES[model_name], "chart": os.path.basename(chart) if chart else None, "chart_type": chart_type, "model_path": model_path}
    except Exception as e:
        return {"success": False, "error": str(e)}
