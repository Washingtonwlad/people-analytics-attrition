"""Streamlit companion for the frozen People Analytics Quarto analysis."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modeling import (  # noqa: E402
    ATTITUDE_INDICATORS,
    BEHAVIORAL_PREDICTORS,
    COMBINED_PREDICTORS,
    CONTEXTUAL_PREDICTORS,
    DEVELOPMENT_REFERENCE_METRICS,
    MODEL_COMPARISON_RESULTS,
    REPORTING_THRESHOLD,
    SENSITIVE_AUDIT_VARIABLES,
    calculate_threshold_metrics,
    fit_frozen_model,
)
from src.preprocessing import (  # noqa: E402
    AMBIGUOUS_RATE_COLUMNS,
    CONSTANT_COLUMNS,
    IDENTIFIER_COLUMN,
    TARGET_BINARY_COLUMN,
    add_analysis_fields,
    load_data,
)


DATA_PATH = PROJECT_ROOT / "data" / "raw" / "HR-Employee-Attrition.csv"
REPORT_PATH = PROJECT_ROOT / "analysis" / "people_analytics_attrition.html"
RETENTION_COLOR = "#2E86AB"
ATTRITION_COLOR = "#E84855"
CONTEXT_COLOR = "#7D3C98"

st.set_page_config(
    page_title="People Analytics — Attrition Evidence",
    page_icon="📊",
    layout="wide",
)
sns.set_theme(style="whitegrid", context="notebook")
plt.close("all")


@st.cache_data
def load_analysis_data() -> pd.DataFrame:
    """Load the validated source and add audit-only analysis fields."""
    return add_analysis_fields(load_data(DATA_PATH))


@st.cache_resource
def train_frozen_result(data: pd.DataFrame):
    """Reproduce the development fit and one frozen held-out evaluation."""
    return fit_frozen_model(data)


def category_attrition_summary(
    data: pd.DataFrame,
    category: str,
) -> pd.DataFrame:
    """Return bounded descriptive attrition rates for a categorical field."""
    return (
        data.groupby(category, observed=True)
        .agg(
            employees=(TARGET_BINARY_COLUMN, "size"),
            attrition_cases=(TARGET_BINARY_COLUMN, "sum"),
            attrition_rate=(TARGET_BINARY_COLUMN, "mean"),
        )
        .reset_index()
        .sort_values("attrition_rate", ascending=False)
    )


def subgroup_performance(
    audit_data: pd.DataFrame,
    attribute: str,
) -> pd.DataFrame:
    """Calculate held-out performance by an audit-only attribute."""
    rows = []
    for subgroup, group in audit_data.groupby(attribute, observed=True):
        observed = group[TARGET_BINARY_COLUMN].to_numpy()
        probabilities = group["PredictedProbability"].to_numpy()
        threshold_metrics = calculate_threshold_metrics(
            observed,
            probabilities,
            REPORTING_THRESHOLD,
        )
        positive_cases = int(observed.sum())
        negative_cases = int(len(observed) - positive_cases)
        has_both_classes = np.unique(observed).size == 2

        if len(observed) < 50 or min(positive_cases, negative_cases) < 5:
            reliability = "Very limited"
        elif min(positive_cases, negative_cases) < 20:
            reliability = "Limited"
        else:
            reliability = "More stable"

        rows.append(
            {
                "Subgroup": str(subgroup),
                "Employees": len(observed),
                "Attrition cases": positive_cases,
                "Observed rate": observed.mean(),
                "Mean predicted": probabilities.mean(),
                "Calibration gap": probabilities.mean() - observed.mean(),
                "Average Precision": (
                    average_precision_score(observed, probabilities)
                    if has_both_classes
                    else np.nan
                ),
                "ROC-AUC": (
                    roc_auc_score(observed, probabilities)
                    if has_both_classes
                    else np.nan
                ),
                "Brier score": brier_score_loss(observed, probabilities),
                "Flag rate": threshold_metrics["flagged_rate"],
                "Precision": threshold_metrics["precision"],
                "Recall": threshold_metrics["recall"],
                "False-positive rate": threshold_metrics["false_positive_rate"],
                "Reliability": reliability,
            }
        )
    return pd.DataFrame(rows)


data = load_analysis_data()
frozen_result = train_frozen_result(data)

st.title("People Analytics: Behavioral Indicators and Attrition")
st.caption(
    "Interactive companion to a reproducible behavioral data science analysis"
)
st.info(
    "This dashboard uses a synthetic educational dataset. It describes associations "
    "and frozen-model validation; it does not identify causes, recommend employment "
    "actions, or provide employee-level risk scores."
)

section = st.sidebar.radio(
    "Explore",
    [
        "Evidence overview",
        "Behavioral evidence",
        "Model validation",
        "Responsible-use audit",
        "Methods and limitations",
    ],
)
st.sidebar.caption("Model: combined L2 logistic regression · C=1")
st.sidebar.caption(f"Descriptive reporting threshold: {REPORTING_THRESHOLD:.2f}")


if section == "Evidence overview":
    st.header("Evidence overview")
    total_employees = len(data)
    attrition_cases = int(data[TARGET_BINARY_COLUMN].sum())
    attrition_rate = data[TARGET_BINARY_COLUMN].mean()
    test_metrics = frozen_result.probability_metrics

    metric_columns = st.columns(4)
    metric_columns[0].metric("Employees", f"{total_employees:,}")
    metric_columns[1].metric(
        "Observed attrition",
        f"{attrition_rate:.1%}",
        help=f"{attrition_cases} observed cases in the complete synthetic sample.",
    )
    metric_columns[2].metric(
        "Held-out Average Precision",
        f"{test_metrics['Average Precision']:.3f}",
        help="Primary threshold-free metric; test prevalence is 0.160.",
    )
    metric_columns[3].metric(
        "Held-out ROC-AUC",
        f"{test_metrics['ROC-AUC']:.3f}",
        help="Secondary ranking metric from the one frozen test evaluation.",
    )

    st.subheader("Observed attrition varies across job-demand conditions")
    overtime_summary = category_attrition_summary(data, "OverTime")
    travel_summary = category_attrition_summary(data, "BusinessTravel")

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(
        overtime_summary["OverTime"],
        overtime_summary["attrition_rate"] * 100,
        color=[ATTRITION_COLOR, RETENTION_COLOR],
    )
    axes[0].set_title("Overtime")
    axes[0].set_ylabel("Observed attrition rate (%)")
    axes[0].set_ylim(0, 35)
    for position, value in enumerate(overtime_summary["attrition_rate"] * 100):
        axes[0].text(position, value + 0.7, f"{value:.1f}%", ha="center")

    travel_plot = travel_summary.sort_values("attrition_rate")
    axes[1].barh(
        travel_plot["BusinessTravel"].str.replace("_", " "),
        travel_plot["attrition_rate"] * 100,
        color=[RETENTION_COLOR, "#F5A05A", ATTRITION_COLOR],
    )
    axes[1].set_title("Business travel")
    axes[1].set_xlabel("Observed attrition rate (%)")
    axes[1].set_xlim(0, 30)
    for position, value in enumerate(travel_plot["attrition_rate"] * 100):
        axes[1].text(value + 0.5, position, f"{value:.1f}%", va="center")

    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)
    st.caption(
        "These are bivariate observed rates. Job role, compensation, tenure, and other "
        "conditions may confound the comparisons."
    )

    st.subheader("What the analysis supports")
    st.markdown(
        """
        - Behavioral and contextual predictors each contain out-of-sample signal.
        - Their combined logistic model consistently outperforms either block alone.
        - Nested XGBoost does not improve discrimination or probability quality.
        - Held-out performance is promising but uncertain and does not establish causal effects.
        """
    )


elif section == "Behavioral evidence":
    st.header("Behavioral and organizational evidence")
    st.warning(
        "The five attitude variables are single four-level indicators, not validated "
        "psychometric scales."
    )

    selected_indicator = st.selectbox(
        "Attitude indicator",
        ATTITUDE_INDICATORS,
    )
    distribution = (
        data.groupby([selected_indicator, "Attrition"], observed=True)
        .size()
        .groupby(level=1)
        .transform(lambda values: values / values.sum() * 100)
        .rename("Percentage")
        .reset_index()
    )

    left_column, right_column = st.columns([1.35, 1])
    with left_column:
        figure, axis = plt.subplots(figsize=(7, 4.5))
        sns.barplot(
            data=distribution,
            x=selected_indicator,
            y="Percentage",
            hue="Attrition",
            hue_order=["No", "Yes"],
            palette=[RETENTION_COLOR, ATTRITION_COLOR],
            ax=axis,
        )
        axis.set_xlabel("Response level")
        axis.set_ylabel("Employees within outcome group (%)")
        axis.set_title(f"{selected_indicator} distribution by observed attrition")
        axis.legend(title="Observed attrition")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with right_column:
        indicator_summary = (
            data.groupby("Attrition")[selected_indicator]
            .agg(["count", "mean", "median", "std"])
            .rename_axis("Observed attrition")
        )
        st.subheader("Group summary")
        st.dataframe(indicator_summary.round(3), width="stretch")
        st.markdown(
            "Employees who left report lower means across all five indicators, but "
            "the distributions overlap substantially. No item is an individual diagnostic."
        )

    st.subheader("Organizational categories")
    selected_category = st.selectbox(
        "Category",
        ["JobRole", "Department", "EducationField"],
    )
    category_summary = category_attrition_summary(data, selected_category)
    st.dataframe(
        category_summary.assign(
            attrition_rate=lambda frame: frame["attrition_rate"].map(
                lambda value: f"{value:.1%}"
            )
        ),
        width="stretch",
        hide_index=True,
    )
    st.caption(
        "Small categories produce unstable rates. These comparisons are descriptive "
        "and involve multiple groups."
    )


elif section == "Model validation":
    st.header("Frozen model validation")
    st.markdown(
        "The primary model is a combined 24-predictor L2 logistic regression. "
        "Preprocessing is fitted inside training folds; identifiers and sensitive "
        "attributes are excluded."
    )

    comparison_column, metric_column = st.columns([1.4, 1])
    with comparison_column:
        st.subheader("Development model comparison")
        model_order = MODEL_COMPARISON_RESULTS["Model"].tolist()
        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.barh(
            model_order,
            MODEL_COMPARISON_RESULTS["Average Precision"],
            color=["#AAB7B8", RETENTION_COLOR, CONTEXT_COLOR, ATTRITION_COLOR, "#D68910"],
        )
        axis.axvline(0.162, color="#566573", linestyle="--", linewidth=1)
        axis.set_xlim(0, 0.72)
        axis.set_xlabel("Mean validation Average Precision")
        axis.invert_yaxis()
        for position, value in enumerate(MODEL_COMPARISON_RESULTS["Average Precision"]):
            axis.text(value + 0.01, position, f"{value:.3f}", va="center")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)

    with metric_column:
        st.subheader("Final held-out metrics")
        st.metric("Average Precision", f"{frozen_result.probability_metrics['Average Precision']:.3f}")
        st.metric("ROC-AUC", f"{frozen_result.probability_metrics['ROC-AUC']:.3f}")
        st.metric("Brier score", f"{frozen_result.probability_metrics['Brier score']:.3f}")
        st.metric("Log loss", f"{frozen_result.probability_metrics['Log loss']:.3f}")

    st.subheader("Development-to-test generalization")
    generalization_table = pd.DataFrame(
        [
            {
                "Metric": metric,
                "Development OOF": development_value,
                "Held-out test": frozen_result.probability_metrics[metric],
                "Test minus development": (
                    frozen_result.probability_metrics[metric] - development_value
                ),
            }
            for metric, development_value in DEVELOPMENT_REFERENCE_METRICS.items()
        ]
    )
    st.dataframe(generalization_table.round(3), width="stretch", hide_index=True)

    st.subheader(f"Descriptive threshold transfer at {REPORTING_THRESHOLD:.2f}")
    threshold_metrics = frozen_result.threshold_metrics
    confusion_matrix = np.array(
        [
            [threshold_metrics["true_negative"], threshold_metrics["false_positive"]],
            [threshold_metrics["false_negative"], threshold_metrics["true_positive"]],
        ]
    )
    matrix_column, threshold_column = st.columns([1.1, 1])
    with matrix_column:
        figure, axis = plt.subplots(figsize=(5.5, 4.2))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Predicted stayed", "Flagged"],
            yticklabels=["Observed stayed", "Observed left"],
            ax=axis,
        )
        axis.set_xlabel("")
        axis.set_ylabel("")
        figure.tight_layout()
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    with threshold_column:
        st.metric("Employees flagged", f"{threshold_metrics['flagged_employees']} / 294")
        st.metric("Precision", f"{threshold_metrics['precision']:.3f}")
        st.metric("Recall", f"{threshold_metrics['recall']:.3f}")
        st.metric("Specificity", f"{threshold_metrics['specificity']:.3f}")
        st.caption(
            "The threshold maximized F1 in development. It is included for descriptive "
            "transfer only and is not an operational employment rule."
        )


elif section == "Responsible-use audit":
    st.header("Responsible-use and subgroup audit")
    st.markdown(
        "Age, gender, and marital status were attached only after held-out predictions "
        "were generated. They never entered model fitting."
    )

    audit_data = frozen_result.test_data.copy()
    audit_data["PredictedProbability"] = frozen_result.test_probabilities
    audit_attribute = st.selectbox(
        "Audit attribute",
        ["AgeBand", "Gender", "MaritalStatus"],
    )
    audit_summary = subgroup_performance(audit_data, audit_attribute)

    figure, axis = plt.subplots(figsize=(9, 4.8))
    x_positions = np.arange(len(audit_summary))
    axis.plot(
        x_positions,
        audit_summary["Observed rate"],
        marker="o",
        linewidth=2,
        color=ATTRITION_COLOR,
        label="Observed rate",
    )
    axis.plot(
        x_positions,
        audit_summary["Mean predicted"],
        marker="o",
        linewidth=2,
        color=RETENTION_COLOR,
        label="Mean predicted",
    )
    axis.set_xticks(x_positions, audit_summary["Subgroup"], rotation=20)
    axis.set_ylabel("Probability / observed rate")
    axis.set_title(f"Held-out calibration by {audit_attribute}")
    axis.legend()
    figure.tight_layout()
    st.pyplot(figure, width="stretch")
    plt.close(figure)

    percentage_columns = [
        "Observed rate",
        "Mean predicted",
        "Calibration gap",
        "Flag rate",
        "Precision",
        "Recall",
        "False-positive rate",
    ]
    display_audit = audit_summary.copy()
    display_audit[percentage_columns] = display_audit[percentage_columns].round(3)
    st.dataframe(display_audit, width="stretch", hide_index=True)

    very_limited = audit_summary.loc[
        audit_summary["Reliability"] == "Very limited", "Subgroup"
    ].tolist()
    if very_limited:
        st.warning(
            "Very limited evidence for: " + ", ".join(very_limited) + ". "
            "These groups need more observations and attrition cases before comparison."
        )
    st.caption(
        "This sample cannot establish legal or deployment fairness. Different base "
        "rates also prevent one threshold from equalizing every performance metric."
    )


else:
    st.header("Methods and limitations")
    st.subheader("Frozen predictor specification")
    predictor_columns = st.columns(2)
    with predictor_columns[0]:
        st.markdown(f"**Behavioral block — {len(BEHAVIORAL_PREDICTORS)} predictors**")
        st.code("\n".join(BEHAVIORAL_PREDICTORS), language=None)
    with predictor_columns[1]:
        st.markdown(f"**Contextual block — {len(CONTEXTUAL_PREDICTORS)} predictors**")
        st.code("\n".join(CONTEXTUAL_PREDICTORS), language=None)

    st.subheader("Explicit exclusions")
    exclusion_table = pd.DataFrame(
        {
            "Role": [
                "Identifier",
                "Constant metadata",
                "Ambiguous rate variables",
                "Sensitive audit only",
            ],
            "Variables": [
                IDENTIFIER_COLUMN,
                ", ".join(CONSTANT_COLUMNS),
                ", ".join(AMBIGUOUS_RATE_COLUMNS),
                ", ".join(SENSITIVE_AUDIT_VARIABLES),
            ],
        }
    )
    st.dataframe(exclusion_table, width="stretch", hide_index=True)

    st.subheader("Validation design")
    st.markdown(
        """
        1. Stable 80/20 stratified development/test split using the identifier only for partition integrity.
        2. Five-fold stratified cross-validation repeated five times inside development.
        3. Fold-specific imputation, standardization, and one-hot encoding.
        4. Average Precision as the primary selection metric; probability quality checked with Brier and log loss.
        5. Nested cross-validation for XGBoost and one final frozen test evaluation.
        6. Sensitive attributes reserved for post-prediction auditing.
        """
    )

    st.subheader("Material limitations")
    st.markdown(
        """
        - Synthetic, cross-sectional data do not establish temporal or external validity.
        - Associations and coefficients are not causal effects.
        - Attitude variables are single ordinal items, not validated psychological scales.
        - Several held-out subgroups contain too few attrition cases for stable fairness metrics.
        - The 0.32 threshold has no intervention-cost or capacity justification.
        - No employee-level prediction should be used for hiring, discipline, promotion, or termination.
        """
    )

    if REPORT_PATH.is_file():
        st.download_button(
            "Download the full rendered analysis",
            data=REPORT_PATH.read_bytes(),
            file_name=REPORT_PATH.name,
            mime="text/html",
        )


st.divider()
st.caption(
    "Washington Casamen Nolasco · Quantitative Psychology & Behavioral Data Science · "
    "Synthetic IBM HR Analytics dataset"
)
