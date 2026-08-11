"""Data loading and validation for the attrition analysis."""

from pathlib import Path

import pandas as pd


TARGET_COLUMN = "Attrition"
TARGET_BINARY_COLUMN = "AttritionBinary"
IDENTIFIER_COLUMN = "EmployeeNumber"

EXPECTED_COLUMNS = [
    "Age",
    "Attrition",
    "BusinessTravel",
    "DailyRate",
    "Department",
    "DistanceFromHome",
    "Education",
    "EducationField",
    "EmployeeCount",
    "EmployeeNumber",
    "EnvironmentSatisfaction",
    "Gender",
    "HourlyRate",
    "JobInvolvement",
    "JobLevel",
    "JobRole",
    "JobSatisfaction",
    "MaritalStatus",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "Over18",
    "OverTime",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StandardHours",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]
AMBIGUOUS_RATE_COLUMNS = ["DailyRate", "HourlyRate", "MonthlyRate"]
SENSITIVE_AUDIT_COLUMNS = ["Age", "Gender", "MaritalStatus"]

AGE_BINS = [18, 25, 35, 45, 55, float("inf")]
AGE_BAND_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def validate_source_data(data: pd.DataFrame) -> None:
    """Raise a clear error when the source no longer matches the analysis contract."""
    missing_columns = sorted(set(EXPECTED_COLUMNS) - set(data.columns))
    unexpected_columns = sorted(set(data.columns) - set(EXPECTED_COLUMNS))
    if missing_columns or unexpected_columns:
        raise ValueError(
            "Unexpected source schema. "
            f"Missing columns: {missing_columns}; "
            f"unexpected columns: {unexpected_columns}."
        )

    if set(data[TARGET_COLUMN].dropna().unique()) != {"No", "Yes"}:
        raise ValueError("Attrition must contain exactly the levels 'No' and 'Yes'.")
    if not data[IDENTIFIER_COLUMN].is_unique:
        raise ValueError("EmployeeNumber must uniquely identify every source row.")
    if data.isna().any().any():
        raise ValueError("The frozen analysis expects a source with no missing cells.")

    detected_constants = data.columns[data.nunique(dropna=False).eq(1)].tolist()
    if set(detected_constants) != set(CONSTANT_COLUMNS):
        raise ValueError(
            "Detected constant columns differ from the frozen specification: "
            f"{detected_constants}."
        )


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load, validate, and return a copy of the raw IBM synthetic dataset."""
    data_path = Path(filepath)
    if not data_path.is_file():
        raise FileNotFoundError(f"Expected data file was not found: {data_path}")

    data = pd.read_csv(data_path)
    validate_source_data(data)
    return data


def add_analysis_fields(data: pd.DataFrame) -> pd.DataFrame:
    """Add the encoded target and audit-only age bands without dropping source fields."""
    analysis_data = data.copy()
    analysis_data[TARGET_BINARY_COLUMN] = (
        analysis_data[TARGET_COLUMN] == "Yes"
    ).astype(int)
    analysis_data["AgeBand"] = pd.cut(
        analysis_data["Age"],
        bins=AGE_BINS,
        labels=AGE_BAND_LABELS,
        right=False,
        include_lowest=True,
    )
    if analysis_data["AgeBand"].isna().any():
        raise ValueError("At least one employee was not assigned to an age band.")
    return analysis_data
