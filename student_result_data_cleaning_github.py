"""Clean the supplied student CSV and save a separate cleaned data file."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().with_name("Student_Result_Unclean.csv")
OUTPUT_FILE = Path(__file__).resolve().with_name("Student_Result_Clean.csv")

NUMERIC_RANGES = {
    "weekly_self_study_hours": (0, 168),
    "attendance_percentage": (0, 100),
    "class_participation": (0, 10),
    "total_score": (0, 100),
}
REQUIRED_COLUMNS = {
    "student_id",
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
    "total_score",
    "grade",
}


def clean_student_data(input_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load, validate, and clean the source data without inventing missing marks."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV was not found: {input_path}")

    data = pd.read_csv(input_path)
    data.columns = (
        data.columns.astype("string")
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Input CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    input_rows = len(data)
    data["student_id"] = data["student_id"].astype("string").str.strip()
    data["student_id"] = data["student_id"].replace("", pd.NA)
    data["grade"] = data["grade"].astype("string").str.strip().str.upper()
    data["grade"] = data["grade"].replace("", pd.NA)

    duplicate_ids = data["student_id"].notna() & data["student_id"].duplicated(
        keep="first"
    )
    duplicate_rows = int(duplicate_ids.sum())
    data = data.loc[~duplicate_ids].copy()

    for column, (minimum, maximum) in NUMERIC_RANGES.items():
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data.loc[~data[column].between(minimum, maximum), column] = np.nan

    imputed_values: dict[str, int] = {}
    for column in NUMERIC_RANGES:
        if column == "total_score":
            continue
        missing_count = int(data[column].isna().sum())
        median = data[column].median()
        if pd.isna(median) and missing_count:
            raise ValueError(
                f"Cannot impute {column}: no valid values are available."
            )
        data[column] = data[column].fillna(median)
        imputed_values[column] = missing_count

    data["data_quality_status"] = np.where(
        data["student_id"].isna(),
        "invalid: missing student_id",
        np.where(
            data["total_score"].isna(),
            "invalid: missing or invalid total_score",
            "valid",
        ),
    )

    stats = {
        "input_rows": input_rows,
        "duplicate_rows_removed": duplicate_rows,
        "rows_missing_student_id": int(data["student_id"].isna().sum()),
        "rows_missing_or_invalid_total_score": int(data["total_score"].isna().sum()),
        "rows_with_imputed_other_values": sum(
            count > 0 for count in imputed_values.values()
        ),
        "imputed_values": sum(imputed_values.values()),
    }
    return data, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and validate the supplied student records."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=INPUT_FILE,
        help=f"source CSV (default: {INPUT_FILE.name})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=OUTPUT_FILE,
        help=f"destination CSV (default: {OUTPUT_FILE.name})",
    )
    args = parser.parse_args()

    input_path = args.input_csv.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("Output CSV must be different from the source CSV.")

    cleaned, stats = clean_student_data(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output_path, index=False)

    print("===== STUDENT DATA CLEANING SUMMARY =====")
    print(f"Input rows: {stats['input_rows']}")
    print(f"Duplicate student IDs removed (kept first): {stats['duplicate_rows_removed']}")
    print(f"Rows missing student ID: {stats['rows_missing_student_id']}")
    print(
        "Rows missing or with invalid total score: "
        f"{stats['rows_missing_or_invalid_total_score']}"
    )
    print(f"Other missing/invalid values imputed: {stats['imputed_values']}")
    print(f"Rows exported: {len(cleaned)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
