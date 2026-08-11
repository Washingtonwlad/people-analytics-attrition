"""Regression tests for the frozen attrition analysis contract."""

from pathlib import Path

import pytest

from src.modeling import (
    COMBINED_PREDICTORS,
    IDENTIFIER_COLUMN,
    SENSITIVE_AUDIT_VARIABLES,
    fit_frozen_model,
    stable_development_test_split,
)
from src.preprocessing import (
    EXPECTED_COLUMNS,
    TARGET_BINARY_COLUMN,
    add_analysis_fields,
    load_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "HR-Employee-Attrition.csv"


@pytest.fixture(scope="module")
def analysis_data():
    """Load the validated synthetic source once for this test module."""
    return add_analysis_fields(load_data(DATA_PATH))


@pytest.fixture(scope="module")
def frozen_result(analysis_data):
    """Fit the frozen pipeline once for metric regression tests."""
    return fit_frozen_model(analysis_data)


def test_source_contract(analysis_data):
    """The portfolio analysis depends on the frozen 1,470 by 35 source."""
    assert len(analysis_data) == 1_470
    assert set(EXPECTED_COLUMNS).issubset(analysis_data.columns)
    assert int(analysis_data[TARGET_BINARY_COLUMN].sum()) == 237
    assert analysis_data[TARGET_BINARY_COLUMN].mean() == pytest.approx(
        0.1612244898
    )


def test_split_is_complete_disjoint_and_leakage_safe(analysis_data):
    """The identifier stabilizes the split but never enters the model."""
    development, test = stable_development_test_split(analysis_data)
    development_ids = set(development[IDENTIFIER_COLUMN])
    test_ids = set(test[IDENTIFIER_COLUMN])

    assert len(development) == 1_176
    assert len(test) == 294
    assert development_ids.isdisjoint(test_ids)
    assert development_ids | test_ids == set(analysis_data[IDENTIFIER_COLUMN])
    assert IDENTIFIER_COLUMN not in COMBINED_PREDICTORS
    assert set(SENSITIVE_AUDIT_VARIABLES).isdisjoint(COMBINED_PREDICTORS)


def test_frozen_held_out_probability_metrics(frozen_result):
    """Guard against silent changes to final held-out performance."""
    expected = {
        "Average Precision": 0.582770,
        "ROC-AUC": 0.799208,
        "Brier score": 0.098650,
        "Log loss": 0.350868,
        "Observed attrition rate": 0.159864,
        "Mean predicted probability": 0.149904,
    }
    for metric, estimate in expected.items():
        assert frozen_result.probability_metrics[metric] == pytest.approx(
            estimate,
            abs=1e-6,
        )


def test_frozen_threshold_confusion_matrix(frozen_result):
    """Guard the descriptive 0.32 threshold and its confusion counts."""
    metrics = frozen_result.threshold_metrics
    assert metrics["threshold"] == pytest.approx(0.32)
    assert metrics["flagged_employees"] == 45
    assert metrics["true_positive"] == 26
    assert metrics["false_positive"] == 19
    assert metrics["true_negative"] == 228
    assert metrics["false_negative"] == 21
