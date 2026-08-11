"""Frozen, leakage-safe logistic model used by the report and Streamlit app."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing import IDENTIFIER_COLUMN, TARGET_BINARY_COLUMN


RANDOM_SEED = 42
TEST_SIZE = 0.20
REPORTING_THRESHOLD = 0.32

ATTITUDE_INDICATORS = [
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobSatisfaction",
    "RelationshipSatisfaction",
    "WorkLifeBalance",
]

BEHAVIORAL_PREDICTORS = [
    *ATTITUDE_INDICATORS,
    "BusinessTravel",
    "DistanceFromHome",
    "OverTime",
]

CAREER_PREDICTORS = [
    "NumCompaniesWorked",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager",
]

CONTEXTUAL_PREDICTORS = [
    *CAREER_PREDICTORS,
    "Department",
    "Education",
    "EducationField",
    "JobLevel",
    "JobRole",
    "MonthlyIncome",
    "PercentSalaryHike",
    "PerformanceRating",
    "StockOptionLevel",
]

COMBINED_PREDICTORS = [*BEHAVIORAL_PREDICTORS, *CONTEXTUAL_PREDICTORS]

NOMINAL_PREDICTORS = {
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "OverTime",
}

SENSITIVE_AUDIT_VARIABLES = ["Age", "Gender", "MaritalStatus"]

DEVELOPMENT_REFERENCE_METRICS = {
    "Average Precision": 0.640,
    "ROC-AUC": 0.835,
    "Brier score": 0.092,
    "Log loss": 0.324,
}

MODEL_COMPARISON_RESULTS = pd.DataFrame(
    [
        {
            "Model": "Prevalence baseline",
            "Average Precision": 0.162,
            "ROC-AUC": 0.500,
            "Brier score": 0.135,
            "Log loss": 0.442,
        },
        {
            "Model": "Behavioral logistic",
            "Average Precision": 0.411,
            "ROC-AUC": 0.747,
            "Brier score": 0.119,
            "Log loss": 0.388,
        },
        {
            "Model": "Contextual logistic",
            "Average Precision": 0.397,
            "ROC-AUC": 0.736,
            "Brier score": 0.120,
            "Log loss": 0.396,
        },
        {
            "Model": "Combined logistic",
            "Average Precision": 0.637,
            "ROC-AUC": 0.834,
            "Brier score": 0.093,
            "Log loss": 0.327,
        },
        {
            "Model": "Nested XGBoost",
            "Average Precision": 0.607,
            "ROC-AUC": 0.825,
            "Brier score": 0.097,
            "Log loss": 0.330,
        },
    ]
)


@dataclass
class FrozenModelResult:
    """Artifacts from the one frozen development/test evaluation."""

    model: Pipeline
    development_data: pd.DataFrame
    test_data: pd.DataFrame
    test_probabilities: np.ndarray
    test_predictions: np.ndarray
    probability_metrics: dict[str, float]
    threshold_metrics: dict[str, float | int]


def build_logistic_pipeline() -> Pipeline:
    """Build the exact C=1 combined-logistic pipeline from the Quarto analysis."""
    numeric_predictors = [
        predictor
        for predictor in COMBINED_PREDICTORS
        if predictor not in NOMINAL_PREDICTORS
    ]
    nominal_predictors = [
        predictor
        for predictor in COMBINED_PREDICTORS
        if predictor in NOMINAL_PREDICTORS
    ]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one_hot",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_predictors),
            ("nominal", nominal_pipeline, nominal_predictors),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2_000,
        class_weight=None,
        random_state=RANDOM_SEED,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def stable_development_test_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the identifier-stable stratified split frozen in the report."""
    required_columns = {
        IDENTIFIER_COLUMN,
        TARGET_BINARY_COLUMN,
        *COMBINED_PREDICTORS,
    }
    missing_columns = sorted(required_columns - set(data.columns))
    if missing_columns:
        raise ValueError(f"Modeling data is missing columns: {missing_columns}")

    split_source = (
        data[[IDENTIFIER_COLUMN, TARGET_BINARY_COLUMN]]
        .sort_values(IDENTIFIER_COLUMN)
        .reset_index(drop=True)
    )
    development_ids, test_ids = train_test_split(
        split_source[IDENTIFIER_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=split_source[TARGET_BINARY_COLUMN],
    )
    development_id_set = set(development_ids)
    test_id_set = set(test_ids)

    if development_id_set & test_id_set:
        raise ValueError("Employee overlap detected between development and test.")
    if development_id_set | test_id_set != set(data[IDENTIFIER_COLUMN]):
        raise ValueError("The frozen split does not cover every employee exactly once.")

    development_data = data.loc[
        data[IDENTIFIER_COLUMN].isin(development_id_set)
    ].copy()
    test_data = data.loc[data[IDENTIFIER_COLUMN].isin(test_id_set)].copy()
    return development_data, test_data


def calculate_threshold_metrics(
    observed: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = REPORTING_THRESHOLD,
) -> dict[str, float | int]:
    """Calculate the frozen-threshold confusion matrix and derived metrics."""
    predicted = probabilities >= threshold
    true_positive = int(((predicted == 1) & (observed == 1)).sum())
    false_positive = int(((predicted == 1) & (observed == 0)).sum())
    true_negative = int(((predicted == 0) & (observed == 0)).sum())
    false_negative = int(((predicted == 0) & (observed == 1)).sum())

    def safe_ratio(numerator: float, denominator: float) -> float:
        return numerator / denominator if denominator else float("nan")

    precision = safe_ratio(true_positive, true_positive + false_positive)
    recall = safe_ratio(true_positive, true_positive + false_negative)
    specificity = safe_ratio(true_negative, true_negative + false_positive)
    f1 = safe_ratio(2 * precision * recall, precision + recall)

    return {
        "threshold": threshold,
        "flagged_employees": int(predicted.sum()),
        "flagged_rate": float(predicted.mean()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_negative": true_negative,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "false_positive_rate": 1 - specificity,
        "f1": f1,
        "balanced_accuracy": float(np.mean([recall, specificity])),
    }


def fit_frozen_model(data: pd.DataFrame) -> FrozenModelResult:
    """Fit on frozen development rows and evaluate once on frozen test rows."""
    if set(SENSITIVE_AUDIT_VARIABLES) & set(COMBINED_PREDICTORS):
        raise ValueError("Sensitive audit variables entered the model specification.")
    if IDENTIFIER_COLUMN in COMBINED_PREDICTORS:
        raise ValueError("EmployeeNumber entered the model specification.")

    development_data, test_data = stable_development_test_split(data)
    model = build_logistic_pipeline()
    model.fit(
        development_data[COMBINED_PREDICTORS],
        development_data[TARGET_BINARY_COLUMN],
    )

    test_observed = test_data[TARGET_BINARY_COLUMN].to_numpy()
    test_probabilities = model.predict_proba(
        test_data[COMBINED_PREDICTORS]
    )[:, 1]
    test_predictions = (test_probabilities >= REPORTING_THRESHOLD).astype(int)

    probability_metrics = {
        "Average Precision": average_precision_score(
            test_observed, test_probabilities
        ),
        "ROC-AUC": roc_auc_score(test_observed, test_probabilities),
        "Brier score": brier_score_loss(test_observed, test_probabilities),
        "Log loss": log_loss(
            test_observed, test_probabilities, labels=[0, 1]
        ),
        "Observed attrition rate": float(test_observed.mean()),
        "Mean predicted probability": float(test_probabilities.mean()),
    }
    threshold_metrics = calculate_threshold_metrics(
        test_observed,
        test_probabilities,
    )

    return FrozenModelResult(
        model=model,
        development_data=development_data,
        test_data=test_data,
        test_probabilities=test_probabilities,
        test_predictions=test_predictions,
        probability_metrics=probability_metrics,
        threshold_metrics=threshold_metrics,
    )
