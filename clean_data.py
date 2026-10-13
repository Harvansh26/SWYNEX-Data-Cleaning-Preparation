from pathlib import Path
import pandas as pd
import numpy as np

# ==========================================
# 1. PATHS AND DATA LOADING
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
RAW_FILE = BASE_DIR / "data" / "raw" / "train.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Dataset not found: {RAW_FILE}")

df = pd.read_csv(RAW_FILE)
df.columns = df.columns.str.strip()

required_columns = {
    "ID", "date", "meals_served", "kitchen_staff",
    "temperature_C", "humidity_percent", "day_of_week",
    "special_event", "past_waste_kg", "staff_experience",
    "waste_category", "food_waste_kg"
}

missing_columns = required_columns - set(df.columns)
if missing_columns:
    raise ValueError(f"Missing expected columns: {missing_columns}")

original = df.copy()

# ==========================================
# 2. BASELINE QUALITY METRICS
# ==========================================
before = {
    "rows": len(df),
    "columns": len(df.columns),
    "missing_cells": int(df.isna().sum().sum()),
    "duplicate_rows": int(df.duplicated().sum()),
    "duplicate_ids": int(df["ID"].duplicated().sum())
}

# ==========================================
# 3. STANDARDIZE TEXT
# ==========================================
for col in ["staff_experience", "waste_category"]:
    df[col] = df[col].map(
        lambda x: x.strip() if isinstance(x, str) else x
    )

# Standardize capitalization
df["staff_experience"] = df["staff_experience"].str.title()
df["waste_category"] = df["waste_category"].str.title()

# ==========================================
# 4. HANDLE MISSING VALUES
# ==========================================
# Unknown experience is explicitly labelled rather than guessed.
df["staff_experience"] = (
    df["staff_experience"].fillna("Unknown")
)

# Preserve any other missing values for investigation.
# Do not invent measurements or dates.

# ==========================================
# 5. CORRECT DATA TYPES
# ==========================================
df["date"] = pd.to_datetime(df["date"], errors="coerce")

numeric_columns = [
    "ID", "meals_served", "kitchen_staff",
    "temperature_C", "humidity_percent", "day_of_week",
    "special_event", "past_waste_kg", "food_waste_kg"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ==========================================
# 6. VALIDATE IDENTIFIERS AND CATEGORIES
# ==========================================
# Report duplicate IDs; don't silently discard records.
duplicate_ids = df[df["ID"].duplicated(keep=False)].copy()
duplicate_ids.to_csv(
    REPORTS_DIR / "duplicate_ids_review.csv", index=False
)

# Exact duplicate rows are safe to remove.
exact_duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates()

# Known valid category values
valid_experience = {
    "Beginner", "Intermediate", "Expert", "Unknown"
}
valid_waste_categories = {
    "Meat", "Dairy", "Vegetables", "Grains"
}

unexpected_experience = df[
    ~df["staff_experience"].isin(valid_experience)
]
unexpected_categories = df[
    ~df["waste_category"].isin(valid_waste_categories)
]

unexpected_experience.to_csv(
    REPORTS_DIR / "unexpected_experience_values.csv", index=False
)
unexpected_categories.to_csv(
    REPORTS_DIR / "unexpected_categories.csv", index=False
)

# ==========================================
# 7. FLAG INVALID OR SUSPICIOUS VALUES
# ==========================================
# These are review rules, not proof that a value is wrong.
df["review_meals_served"] = df["meals_served"] <= 0
df["review_kitchen_staff"] = df["kitchen_staff"] <= 0
df["review_humidity"] = ~df["humidity_percent"].between(0, 100)
df["review_special_event"] = ~df["special_event"].isin([0, 1])
df["review_food_waste"] = df["food_waste_kg"] < 0
df["review_past_waste"] = df["past_waste_kg"] < 0
df["review_temperature"] = (
    df["temperature_C"].lt(-50)
    | df["temperature_C"].gt(60)
)

# IQR-based outlier flags; observations are retained.
for col in ["meals_served", "food_waste_kg", "past_waste_kg"]:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    df[f"outlier_{col}"] = (
        (df[col] < q1 - 1.5 * iqr)
        | (df[col] > q3 + 1.5 * iqr)
    )

# Export review records separately.
review_columns = [
    col for col in df.columns
    if col.startswith("review_") or col.startswith("outlier_")
]

df.loc[df[review_columns].any(axis=1)].to_csv(
    REPORTS_DIR / "records_for_review.csv", index=False
)

# ==========================================
# 8. EXPORT CLEANED DATA
# ==========================================
# Preserve the original measurements and IDs.
# Remove temporary review flags from the main dataset.
cleaned = df.drop(columns=review_columns)

cleaned.to_csv(
    PROCESSED_DIR / "food_waste_cleaned.csv",
    index=False,
    date_format="%Y-%m-%d"
)

# ==========================================
# 9. GENERATE BEFORE-AND-AFTER REPORT
# ==========================================
after = {
    "rows": len(cleaned),
    "columns": len(cleaned.columns),
    "missing_cells": int(cleaned.isna().sum().sum()),
    "duplicate_rows": int(cleaned.duplicated().sum()),
    "duplicate_ids": int(cleaned["ID"].duplicated().sum())
}

summary = pd.DataFrame([
    {"metric": key, "before": value, "after": after[key]}
    for key, value in before.items()
])

summary.to_csv(REPORTS_DIR / "cleaning_summary.csv", index=False)

missing_report = pd.DataFrame({
    "column": cleaned.columns,
    "missing_values": cleaned.isna().sum().values
})
missing_report.to_csv(
    REPORTS_DIR / "missing_values_after.csv", index=False
)

category_report = pd.DataFrame({
    "staff_experience": cleaned["staff_experience"].value_counts(
        dropna=False
    ).index.astype(str)
})
category_report.to_csv(
    REPORTS_DIR / "staff_experience_categories.csv", index=False
)

# ==========================================
# 10. PRINT RESULTS
# ==========================================
print("\n===== FOOD WASTE DATA CLEANING COMPLETE =====")
print("\nBEFORE:")
print(pd.Series(before).to_string())

print("\nAFTER:")
print(pd.Series(after).to_string())

print("\nStandardized waste categories:")
print(cleaned["waste_category"].value_counts(dropna=False).to_string())

print("\nStaff experience categories:")
print(cleaned["staff_experience"].value_counts(dropna=False).to_string())

print("\nReview records:", int(
    df[review_columns].any(axis=1).sum()
))
print("\nCleaned dataset:", PROCESSED_DIR / "food_waste_cleaned.csv")
print("Reports directory:", REPORTS_DIR)
