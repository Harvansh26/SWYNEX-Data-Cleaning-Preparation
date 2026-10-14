
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. PROJECT PATHS AND SETTINGS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DATA_FILE = BASE_DIR / "data" / "food_waste_cleaned.csv"
CHART_DIR = BASE_DIR / "charts"
REPORT_DIR = BASE_DIR / "reports"

CHART_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CHART_COLOR = "#5278A8"
ACCENT_COLOR = "#D97757"
FIGURE_DPI = 200

sns.set_theme(
    style="whitegrid",
    context="notebook",
    font_scale=1.0
)


# --------------------------------------------------
# 2. HELPER FUNCTION FOR SAVING CHARTS
# --------------------------------------------------

def save_chart(fig, filename):
    """Save a chart and close its figure safely."""

    output_path = CHART_DIR / filename
    fallback_path = output_path.with_name(
        f"{output_path.stem}_updated.png"
    )

    try:
        fig.tight_layout()

        try:
            fig.savefig(
                str(output_path.resolve()),
                format="png",
                dpi=FIGURE_DPI,
                bbox_inches="tight"
            )
            print(f"Chart saved: {output_path}")

        except OSError as error:
            # If the existing image is locked, save a new copy.
            if fallback_path.exists():
                fallback_path = CHART_DIR / (
                    f"{output_path.stem}_updated_"
                    f"{pd.Timestamp.now().strftime('%H%M%S')}.png"
                )

            fig.savefig(
                str(fallback_path.resolve()),
                format="png",
                dpi=FIGURE_DPI,
                bbox_inches="tight"
            )

            print(f"Original chart could not be overwritten: {error}")
            print(f"Updated chart saved instead: {fallback_path}")

    finally:
        plt.close(fig)


# --------------------------------------------------
# 3. LOAD AND VALIDATE DATA
# --------------------------------------------------

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Cleaned dataset not found: {DATA_FILE}\n"
        "Check that food_waste_cleaned.csv is inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

required_columns = [
    "ID",
    "date",
    "meals_served",
    "kitchen_staff",
    "temperature_C",
    "humidity_percent",
    "day_of_week",
    "special_event",
    "past_waste_kg",
    "staff_experience",
    "waste_category",
    "food_waste_kg"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns are missing: {missing_columns}"
    )

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)

print("=" * 65)
print("FOOD WASTE EXPLORATORY DATA ANALYSIS")
print("=" * 65)

print("\n1. DATASET OVERVIEW")
print("Rows and columns:", df.shape)

print("\nColumn types:")
print(df.dtypes.to_string())

print("\n2. NUMERICAL SUMMARY")
print(df.describe().round(2).to_string())

print("\n3. MISSING VALUES")
print(df.isna().sum().to_string())

print("\nExact duplicate rows:", df.duplicated().sum())


# --------------------------------------------------
# 4. GROUPED STATISTICS
# --------------------------------------------------

category_stats = (
    df.groupby("waste_category")["food_waste_kg"]
    .agg(["count", "mean", "median", "std"])
    .sort_values("mean", ascending=False)
    .round(2)
)

category_stats.to_csv(
    REPORT_DIR / "food_waste_by_category.csv"
)

experience_stats = (
    df.groupby("staff_experience")["food_waste_kg"]
    .agg(["count", "mean", "median"])
    .sort_values("mean", ascending=False)
    .round(2)
)

experience_stats.to_csv(
    REPORT_DIR / "food_waste_by_experience.csv"
)

day_stats = (
    df.groupby("day_of_week")["food_waste_kg"]
    .agg(["count", "mean", "median"])
    .sort_index()
    .round(2)
)

day_stats.to_csv(
    REPORT_DIR / "food_waste_by_day.csv"
)

print("\n4. FOOD WASTE BY CATEGORY (kg)")
print(category_stats.to_string())

print("\n5. FOOD WASTE BY STAFF EXPERIENCE (kg)")
print(experience_stats.to_string())

print("\n6. FOOD WASTE BY DAY CODE (kg)")
print(day_stats.to_string())


# --------------------------------------------------
# 5. CORRELATION ANALYSIS
# --------------------------------------------------

meal_corr = df["meals_served"].corr(
    df["food_waste_kg"]
)

temp_corr = df["temperature_C"].corr(
    df["food_waste_kg"]
)

average_waste = df["food_waste_kg"].mean()

print("\n7. CORRELATIONS")
print(f"Meals served vs food waste: {meal_corr:.3f}")
print(f"Temperature vs food waste: {temp_corr:.3f}")
print(f"Average food waste: {average_waste:.2f} kg")


# --------------------------------------------------
# 6. CHART 1: AVERAGE WASTE BY CATEGORY
# --------------------------------------------------

category_order = category_stats.index.tolist()

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=df,
    x="waste_category",
    y="food_waste_kg",
    order=category_order,
    estimator="mean",
    errorbar=None,
    color=CHART_COLOR,
    ax=ax
)

ax.bar_label(
    ax.containers[0],
    fmt="%.2f kg",
    padding=5,
    fontsize=10,
    fontweight="bold"
)

ax.set_title(
    "Average Food Waste by Category",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Food Category", fontsize=11)
ax.set_ylabel("Average Food Waste (kg)", fontsize=11)

ax.set_ylim(
    0,
    category_stats["mean"].max() * 1.20
)

sns.despine(ax=ax)

save_chart(fig, "01_waste_by_category.png")


# --------------------------------------------------
# 7. CHART 2: MEALS SERVED VS FOOD WASTE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="meals_served",
    y="food_waste_kg",
    alpha=0.65,
    s=45,
    color=CHART_COLOR,
    edgecolor="white",
    linewidth=0.3,
    ax=ax
)

ax.set_title(
    f"Meals Served vs Food Waste (r = {meal_corr:.3f})",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Meals Served", fontsize=11)
ax.set_ylabel("Food Waste (kg)", fontsize=11)

sns.despine(ax=ax)

save_chart(fig, "02_meals_vs_waste.png")


# --------------------------------------------------
# 8. CHART 2B: FOCUSED VIEW OF MEAL COUNTS
# --------------------------------------------------

# This supplementary chart excludes records above 1,000 meals.
# The full-dataset chart above is retained.

focused_df = df[
    df["meals_served"] <= 1000
].copy()

if not focused_df.empty:

    focused_corr = focused_df[
        "meals_served"
    ].corr(
        focused_df["food_waste_kg"]
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.regplot(
        data=focused_df,
        x="meals_served",
        y="food_waste_kg",
        ci=None,
        scatter_kws={
            "alpha": 0.60,
            "s": 35,
            "color": CHART_COLOR
        },
        line_kws={
            "color": ACCENT_COLOR,
            "linewidth": 2
        },
        ax=ax
    )

    ax.set_title(
        "Meals Served vs Food Waste: Up to 1,000 Meals "
        f"(r = {focused_corr:.3f})",
        fontsize=14,
        fontweight="bold",
        pad=15
    )

    ax.set_xlabel("Meals Served", fontsize=11)
    ax.set_ylabel("Food Waste (kg)", fontsize=11)

    sns.despine(ax=ax)

    save_chart(fig, "02b_meals_vs_waste_focused.png")


# --------------------------------------------------
# 9. CHART 3: FOOD WASTE BY STAFF EXPERIENCE
# --------------------------------------------------

experience_order = experience_stats.index.tolist()

fig, ax = plt.subplots(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="staff_experience",
    y="food_waste_kg",
    order=experience_order,
    color=CHART_COLOR,
    width=0.60,
    showfliers=True,
    ax=ax
)

group_counts = df["staff_experience"].value_counts()

labels = [
    f"{group}\n(n={group_counts[group]})"
    for group in experience_order
]

ax.set_xticks(range(len(experience_order)))
ax.set_xticklabels(labels)

ax.set_title(
    "Food Waste by Staff Experience",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Staff Experience Group", fontsize=11)
ax.set_ylabel("Food Waste (kg)", fontsize=11)

sns.despine(ax=ax)

save_chart(fig, "03_waste_by_experience.png")


# --------------------------------------------------
# 10. CHART 4: TEMPERATURE VS FOOD WASTE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 6))

sns.scatterplot(
    data=df,
    x="temperature_C",
    y="food_waste_kg",
    alpha=0.65,
    s=45,
    color=CHART_COLOR,
    edgecolor="white",
    linewidth=0.3,
    ax=ax
)

ax.set_title(
    f"Temperature vs Food Waste (r = {temp_corr:.3f})",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Temperature (°C)", fontsize=11)
ax.set_ylabel("Food Waste (kg)", fontsize=11)

ax.axhline(
    average_waste,
    color=ACCENT_COLOR,
    linestyle="--",
    linewidth=1.8,
    label=f"Average waste: {average_waste:.2f} kg"
)

ax.legend(
    frameon=False,
    loc="upper right"
)

sns.despine(ax=ax)

save_chart(fig, "04_temperature_vs_waste.png")


# --------------------------------------------------
# 11. CHART 5: AVERAGE WASTE BY DAY CODE
# --------------------------------------------------

day_plot = day_stats.reset_index()

fig, ax = plt.subplots(figsize=(10, 6))

sns.barplot(
    data=day_plot,
    x="day_of_week",
    y="mean",
    order=sorted(day_plot["day_of_week"].unique()),
    errorbar=None,
    color=CHART_COLOR,
    ax=ax
)

# Add average waste labels above each bar.
for bar in ax.patches:
    height = bar.get_height()

    if pd.notna(height):
        ax.annotate(
            f"{height:.2f} kg",
            (
                bar.get_x() + bar.get_width() / 2,
                height
            ),
            xytext=(0, 7),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold"
        )

# Show the overall average as a reference line.
ax.axhline(
    average_waste,
    color=ACCENT_COLOR,
    linestyle="--",
    linewidth=2,
    label=f"Overall average: {average_waste:.2f} kg"
)

ax.set_title(
    "Average Food Waste by Day Code",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Day-of-Week Code", fontsize=11)
ax.set_ylabel("Average Food Waste (kg)", fontsize=11)

ax.set_xticks(sorted(day_plot["day_of_week"].unique()))

# Add space above the tallest bar for its label.
highest_day_mean = day_plot["mean"].max()

ax.set_ylim(
    0,
    max(highest_day_mean, average_waste) * 1.20
)

ax.legend(
    frameon=False,
    loc="upper right"
)

ax.grid(axis="x", visible=False)

sns.despine(ax=ax)

save_chart(fig, "05_waste_by_day.png")


# --------------------------------------------------
# 12. OUTLIER SCREENING USING THE IQR RULE
# --------------------------------------------------

q1 = df["food_waste_kg"].quantile(0.25)
q3 = df["food_waste_kg"].quantile(0.75)

iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = df[
    (df["food_waste_kg"] < lower)
    | (df["food_waste_kg"] > upper)
].copy()

outliers.to_csv(
    REPORT_DIR / "food_waste_outlier_review.csv",
    index=False
)

print("\n8. OUTLIER REVIEW")
print(f"Lower IQR boundary: {lower:.2f} kg")
print(f"Upper IQR boundary: {upper:.2f} kg")
print(f"Records flagged for review: {len(outliers)}")


# --------------------------------------------------
# 13. GENERATE FIVE DATA-BASED INSIGHTS
# --------------------------------------------------

highest_category = category_stats["mean"].idxmax()
lowest_category = category_stats["mean"].idxmin()

highest_experience = experience_stats["mean"].idxmax()

highest_day = day_stats["mean"].idxmax()
lowest_day = day_stats["mean"].idxmin()

insights = [
    (
        f"1. Category comparison: {highest_category} has the "
        f"highest average food waste "
        f"({category_stats.loc[highest_category, 'mean']:.2f} kg), "
        f"while {lowest_category} has the lowest "
        f"({category_stats.loc[lowest_category, 'mean']:.2f} kg)."
    ),
    (
        f"2. Meals and waste: the Pearson correlation is "
        f"{meal_corr:.3f}. This describes a linear association "
        "and does not establish causation."
    ),
    (
        f"3. Staff experience comparison: {highest_experience} "
        f"has the highest observed average food waste "
        f"({experience_stats.loc[highest_experience, 'mean']:.2f} kg). "
        "This is an association, not proof that experience causes "
        "the difference."
    ),
    (
        f"4. Day-code comparison: code {int(highest_day)} has "
        f"the highest average waste "
        f"({day_stats.loc[highest_day, 'mean']:.2f} kg); "
        f"code {int(lowest_day)} has the lowest "
        f"({day_stats.loc[lowest_day, 'mean']:.2f} kg). "
        "The code-to-weekday mapping has not been assumed."
    ),
    (
        f"5. Outlier screening: the IQR rule flags "
        f"{len(outliers)} records with food waste below "
        f"{lower:.2f} kg or above {upper:.2f} kg. "
        "These are candidates for investigation, not automatically errors."
    )
]

insight_text = "\n\n".join(insights)

(REPORT_DIR / "eda_insights.txt").write_text(
    insight_text,
    encoding="utf-8"
)

print("\n9. FIVE INITIAL INSIGHTS")
print(insight_text)


# --------------------------------------------------
# 14. SAVE A RUN SUMMARY
# --------------------------------------------------

summary = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "missing_cells",
        "exact_duplicate_rows",
        "meals_waste_correlation",
        "temperature_waste_correlation",
        "iqr_outlier_records"
    ],
    "value": [
        len(df),
        len(df.columns),
        int(df.isna().sum().sum()),
        int(df.duplicated().sum()),
        round(meal_corr, 3),
        round(temp_corr, 3),
        len(outliers)
    ]
})

summary.to_csv(
    REPORT_DIR / "cleaning_summary.csv",
    index=False
)


# --------------------------------------------------
# 15. COMPLETION
# --------------------------------------------------

print("\nCharts saved in:", CHART_DIR.resolve())
print("Reports saved in:", REPORT_DIR.resolve())
print("\nEDA completed successfully.")
