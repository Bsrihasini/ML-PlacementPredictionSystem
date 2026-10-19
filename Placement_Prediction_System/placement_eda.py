import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
#
import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

CHARTS_DIR = os.path.join(os.path.dirname(__file__), "static", "charts")


def _chart_path(filename):
    os.makedirs(CHARTS_DIR, exist_ok=True)
    return os.path.join(CHARTS_DIR, filename)


def run_eda():

    data = load_data()

    charts = []

    sns.set_style("whitegrid")

    # Missing Values
    missing = data.isnull().sum()
    missing = missing[missing > 0]

    if not missing.empty:

        plt.figure(figsize=(10,5))
        sns.barplot(x=missing.index, y=missing.values)

        plt.xticks(rotation=45)
        plt.title("Missing Values")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_values.png"))
        plt.close()

        charts.append("missing_values.png")

        plt.figure(figsize=(12,6))
        sns.heatmap(data.isnull(), cbar=False)

        plt.title("Missing Value Heatmap")

        plt.tight_layout()
        plt.savefig(_chart_path("missing_heatmap.png"))
        plt.close()

        charts.append("missing_heatmap.png")

    duplicates = int(data.duplicated().sum())

    target_counts = data["PlacementStatus"].value_counts().to_dict()

    plt.figure(figsize=(6,5))

    sns.countplot(
        x="PlacementStatus",
        data=data
    )

    plt.title("Placement Status Distribution")

    plt.tight_layout()

    plt.savefig(_chart_path("placement_status.png"))
    plt.close()

    charts.append("placement_status.png")




    hist_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore"
    ]

    hist_cols = [c for c in hist_cols if c in data.columns]

    if hist_cols:

        data[hist_cols].hist(figsize=(14,10), bins=20)

        plt.tight_layout()
        plt.savefig(_chart_path("numeric_distributions.png"))
        plt.close()

        charts.append("numeric_distributions.png")

    if "CGPA" in data.columns:

        plt.figure(figsize=(8,5))

        sns.histplot(data["CGPA"], kde=True)

        plt.axvline(
            data["CGPA"].mean(),
            color="green",
            linestyle="--",
            label="Mean"
        )

        plt.legend()
        plt.title("CGPA Distribution")

        plt.tight_layout()
        plt.savefig(_chart_path("cgpa_distribution.png"))
        plt.close()

        charts.append("cgpa_distribution.png")




    box_cols = [
        "CGPA",
        "AttendancePercent",
        "AptitudeTestScore",
        "SoftSkillsRating",
        "CodingTestScore",
        "MockInterviewScore",
        "Salary Package"
    ]

    box_cols = [c for c in box_cols if c in data.columns]

    for col in box_cols:

        plt.figure(figsize=(8,4))

        sns.boxplot(x=data[col], color="skyblue")

        plt.title(f"Boxplot - {col}")

        plt.tight_layout()

        filename = f"boxplot_{col.replace(' ','_')}.png"

        plt.savefig(_chart_path(filename))
        plt.close()

        charts.append(filename)




    corr = data.select_dtypes(include="number").corr()

    plt.figure(figsize=(14,10))

    sns.heatmap(
        corr,
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    plt.tight_layout()
    plt.savefig(_chart_path("correlation_heatmap.png"))
    plt.close()

    charts.append("correlation_heatmap.png")




    if "CGPA" in data.columns and "Salary Package" in data.columns:

        plt.figure(figsize=(8,5))

        sns.regplot(
            x="CGPA",
            y="Salary Package",
            data=data,
            color="red"
        )

        plt.title("CGPA vs Salary Package")

        plt.tight_layout()
        plt.savefig(_chart_path("cgpa_salary.png"))
        plt.close()

        charts.append("cgpa_salary.png")


    if "CodingTestScore" in data.columns and "AptitudeTestScore" in data.columns:

        plt.figure(figsize=(8,5))

        sns.regplot(
            x="CodingTestScore",
            y="AptitudeTestScore",
            data=data,
            color="red"
        )

        plt.title("Coding Test Score vs Aptitude Test Score")

        plt.tight_layout()
        plt.savefig(_chart_path("coding_aptitude.png"))
        plt.close()

        charts.append("coding_aptitude.png")



    cat_cols = [
        "Gender",
        "City",
        "CollegeTier",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs",
        "CGPA_Tier"
    ]

    cat_cols = [c for c in cat_cols if c in data.columns]

    for col in cat_cols:

        plt.figure(figsize=(10,5))

        order = data[col].value_counts().index

        sns.countplot(
            x=col,
            data=data,
            order=order
        )

        plt.xticks(rotation=45)

        plt.title(f"{col} Count")

        plt.tight_layout()

        filename = f"{col.lower()}_count.png"

        plt.savefig(_chart_path(filename))
        plt.close()

        charts.append(filename)



    if "Gender" in data.columns and "PlacementStatus" in data.columns:

        plt.figure(figsize=(7,5))

        sns.countplot(
            x="Gender",
            hue="PlacementStatus",
            data=data
        )

        plt.title("Placement Status by Gender")

        plt.tight_layout()
        plt.savefig(_chart_path("gender_vs_placement.png"))
        plt.close()

        charts.append("gender_vs_placement.png")



    if "CollegeTier" in data.columns and "PlacementStatus" in data.columns:

        plt.figure(figsize=(7,5))

        sns.countplot(
            x="CollegeTier",
            hue="PlacementStatus",
            data=data
        )

        plt.title("Placement Status by College Tier")

        plt.tight_layout()
        plt.savefig(_chart_path("college_tier_vs_placement.png"))
        plt.close()

        charts.append("college_tier_vs_placement.png")



    sgpa_cols = [
        f"SGPA_Sem{i}"
        for i in range(1,9)
        if f"SGPA_Sem{i}" in data.columns
    ]

    if sgpa_cols:

        avg_sgpa = data[sgpa_cols].mean()

        plt.figure(figsize=(8,5))

        plt.plot(
            avg_sgpa.index,
            avg_sgpa.values,
            marker="o"
        )

        plt.title("Average SGPA Across Semesters")
        plt.xlabel("Semester")
        plt.ylabel("Average SGPA")

        plt.tight_layout()
        plt.savefig(_chart_path("sgpa_trend.png"))
        plt.close()

        charts.append("sgpa_trend.png")


    if "Salary Package" in data.columns:

        plt.figure(figsize=(7,5))

        sns.histplot(
            data["Salary Package"],
            kde=True
        )

        plt.title("Salary Package Distribution")

        plt.tight_layout()
        plt.savefig(_chart_path("salary_distribution.png"))
        plt.close()

        charts.append("salary_distribution.png")


        plt.figure(figsize=(7,5))

        sns.boxplot(
            x=data["Salary Package"]
        )

        plt.title("Salary Package Boxplot")

        plt.tight_layout()
        plt.savefig(_chart_path("salary_boxplot.png"))
        plt.close()

        charts.append("salary_boxplot.png")


    pair_cols = [
        "CGPA",
        "AttendancePercent",
        "PlacementStatus"
    ]

    pair_cols = [c for c in pair_cols if c in data.columns]

    if len(pair_cols) >= 3:

        sample_data = data[pair_cols].sample(
            min(500, len(data)),
            random_state=42
        )

        g = sns.pairplot(
            sample_data,
            hue="PlacementStatus"
        )

        g.savefig(_chart_path("pairplot.png"))
        plt.close("all")

        charts.append("pairplot.png")
    return {
        "rows": len(data),
        "columns": len(data.columns),
        "duplicates": duplicates,
        "missing": missing.to_dict(),
        "target_counts": target_counts,
        "charts": charts
    }