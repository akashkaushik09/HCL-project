"""Calculate percentage, grade, and pass/fail from the cleaned student CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


INPUT_FILE = Path(__file__).resolve().with_name("Student_Result_Clean.csv")
OUTPUT_FILE = Path(__file__).resolve().with_name("Student_Result_Calculated.csv")
PASS_MARK = 33
REQUIRED_COLUMNS = {"student_id", "total_score", "data_quality_status"}


def calculate_results(input_path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Calculate results only for rows with valid IDs and total scores."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Cleaned CSV was not found: {input_path}")

    data = pd.read_csv(input_path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(
            "Cleaned CSV is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["student_id"] = data["student_id"].astype("string").str.strip()
    data["student_id"] = data["student_id"].replace("", pd.NA)
    data["total_score"] = pd.to_numeric(data["total_score"], errors="coerce")

    valid = (
        data["student_id"].notna()
        & data["total_score"].between(0, 100)
        & data["data_quality_status"].eq("valid")
    )
    results = data.copy()
    results["percentage"] = np.nan
    results["calculated_grade"] = pd.Series(pd.NA, index=results.index, dtype="string")
    results["result_status"] = pd.Series(pd.NA, index=results.index, dtype="string")

    scores = results.loc[valid, "total_score"]
    results.loc[valid, "percentage"] = scores
    results.loc[valid, "calculated_grade"] = np.select(
        [
            scores >= 90,
            scores >= 80,
            scores >= 70,
            scores >= 60,
            scores >= 50,
        ],
        ["A+", "A", "B", "C", "D"],
        default="E",
    )
    results.loc[valid, "result_status"] = np.where(
        scores >= PASS_MARK, "PASS", "FAIL"
    )

    passed = int(results["result_status"].eq("PASS").sum())
    failed = int(results["result_status"].eq("FAIL").sum())
    stats = {
        "input_rows": len(data),
        "calculated_rows": int(valid.sum()),
        "excluded_rows": int(len(data) - valid.sum()),
        "passed_rows": passed,
        "failed_rows": failed,
    }
    return results, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate student results from a cleaned CSV file."
    )
    parser.add_argument(
        "input_csv",
        nargs="?",
        type=Path,
        default=INPUT_FILE,
        help=f"cleaned source CSV (default: {INPUT_FILE.name})",
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
        raise ValueError("Output CSV must be different from the cleaned source CSV.")

    results, stats = calculate_results(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)

    print("===== STUDENT RESULT CALCULATION SUMMARY =====")
    print(f"Input rows: {stats['input_rows']}")
    print(f"Rows with calculated results: {stats['calculated_rows']}")
    print(f"Rows excluded from calculation: {stats['excluded_rows']}")
    print(f"Pass: {stats['passed_rows']}")
    print(f"Fail: {stats['failed_rows']}")
    print(f"Rows exported: {len(results)}")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    main()
