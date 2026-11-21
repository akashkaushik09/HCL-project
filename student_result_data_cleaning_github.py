import pandas as pd
import numpy as np

print("===== STUDENT RESULT DATA CLEANING =====")

# 1. Load original dataset
df = pd.read_csv("Student_result_calc(1).csv")

print("\nOriginal Dataset:")
print(df.head())

# 2. Create unclean dataset
unclean = df.copy()

# Introduce missing values
unclean.loc[7, "weekly_self_study_hours"] = np.nan
unclean.loc[15, "attendance_percentage"] = np.nan
unclean.loc[32, "class_participation"] = np.nan
unclean.loc[61, "total_score"] = np.nan

# Introduce duplicate records
unclean = pd.concat(
    [unclean, unclean.iloc[[4, 20, 100]]],
    ignore_index=True
)

# Introduce invalid values
unclean.loc[35, "weekly_self_study_hours"] = -5
unclean.loc[90, "attendance_percentage"] = 115
unclean.loc[160, "class_participation"] = 15
unclean.loc[275, "total_score"] = -10

# Inconsistent grade formatting
unclean["grade"] = unclean["grade"].astype("string").str.strip()
unclean.loc[10, "grade"] = " a "

# Save unclean dataset
unclean.to_csv("Student_Result_Unclean.csv", index=False)

print("\nUnclean Dataset Created Successfully!")

# 3. Data Cleaning
clean = unclean.copy()

# Remove duplicate records
clean.drop_duplicates(inplace=True)

# Standardize grade values
clean["grade"] = (
    clean["grade"].astype("string").str.strip().str.upper()
)

# Convert numeric columns
numeric_columns = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score"
]

for col in numeric_columns:
    clean[col] = pd.to_numeric(clean[col], errors="coerce")

# 4. Handle invalid values
clean.loc[
    ~clean["weekly_self_study_hours"].between(0, 168),
    "weekly_self_study_hours"
] = np.nan

clean.loc[
    ~clean["attendance_percentage"].between(0, 100),
    "attendance_percentage"
] = np.nan

clean.loc[
    ~clean["class_participation"].between(0, 10),
    "class_participation"
] = np.nan

clean.loc[
    ~clean["total_score"].between(0, 100),
    "total_score"
] = np.nan

# 5. Handle missing values using median
for col in numeric_columns:
    clean[col] = clean[col].fillna(clean[col].median())

# 6. Save clean dataset
clean.to_csv("Student_Result_Clean.csv", index=False)

# 7. Display final results
print("\n===== DATA CLEANING SUMMARY =====")

print("\nOriginal Dataset Shape:", df.shape)
print("Unclean Dataset Shape:", unclean.shape)
print("Clean Dataset Shape:", clean.shape)

print("\nMissing Values Before Cleaning:")
print(unclean.isnull().sum())

print("\nMissing Values After Cleaning:")
print(clean.isnull().sum())

print("\nDuplicate Records After Cleaning:")
print(clean.duplicated().sum())

print("\nCleaned Dataset:")
print(clean.head())

print("\nClean Dataset Saved Successfully!")
