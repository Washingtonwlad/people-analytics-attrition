# People Analytics: Behavioral Indicators and Attrition

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![Quarto](https://img.shields.io/badge/Quarto-reproducible_report-39729E?logo=quarto)
![Streamlit](https://img.shields.io/badge/Streamlit-interactive_companion-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

A reproducible behavioral data science case study examining observed employee attrition in the synthetic IBM HR Analytics dataset. The project combines descriptive evidence, leakage-safe model validation, probability assessment, threshold trade-offs, and subgroup auditing.

The central question is deliberately limited:

> How much predictive information do behavioral indicators add when combined with career and organizational context, and how reliably does that information generalize?

This is an educational portfolio project, not an operational HR decision system. It does not infer causes, prescribe interventions, or generate employee-level recommendations.

## Live Report

**Published Quarto report:** [washingtonwlad.github.io/people-analytics-attrition](https://washingtonwlad.github.io/people-analytics-attrition/)

The report contains the full analytical pipeline, including exploratory descriptive evidence, leakage-safe nested cross-validation, probability calibration, threshold trade-off analysis, and post-prediction subgroup auditing across age, gender, and marital status.

## Main results

The analysis uses 1,470 synthetic employee records with 237 observed attrition cases (16.1%). A stratified 20% test set is held out until the model specification and reporting threshold are frozen.

| Result | Estimate |
|---|---:|
| Behavioral logistic CV Average Precision | 0.411 |
| Contextual logistic CV Average Precision | 0.397 |
| Combined logistic CV Average Precision | 0.637 |
| Nested XGBoost CV Average Precision | 0.607 |
| Held-out Average Precision | 0.583 |
| Held-out ROC-AUC | 0.799 |
| Held-out Brier score | 0.099 |
| Held-out log loss | 0.351 |

The combined regularized logistic model outperforms the behavioral-only and contextual-only models on all 25 paired development folds. Nested XGBoost does not improve on it, so the simpler logistic model is retained.

At the development-selected descriptive threshold of 0.32, the held-out test results are:

- 45 of 294 employees flagged (15.3%);
- precision 0.578 and recall 0.553;
- specificity 0.923 and F1 0.565;
- 26 true positives, 19 false positives, 228 true negatives, and 21 false negatives.

This threshold illustrates classification trade-offs only. It has no intervention-cost, legal, or organizational-capacity justification.

## Analytical design

### Behavioral block

- Environment satisfaction
- Job involvement
- Job satisfaction
- Relationship satisfaction
- Work-life balance
- Business travel
- Distance from home
- Overtime

### Contextual block

Career tenure, training, department, education, job level and role, monthly income, salary increase, performance rating, and stock-option level.

Age, gender, and marital status are excluded from model training and reserved for post-prediction auditing. `EmployeeNumber` is used only to make the split stable. Constant metadata and ambiguously defined rate variables are excluded.

### Validation protocol

1. Stable stratified 80/20 development-test partition.
2. Five-fold stratified cross-validation repeated five times inside development.
3. Fold-specific imputation, standardization, and one-hot encoding.
4. Average Precision as the primary selection metric.
5. ROC-AUC, Brier score, and log loss as complementary diagnostics.
6. Nested evaluation of prespecified XGBoost candidates.
7. One final evaluation of the frozen logistic pipeline on the held-out test set.
8. Post-prediction subgroup audit with explicit sample-size warnings.

## Repository structure

```text
people-analytics-attrition/
|-- analysis/
|   |-- people_analytics_attrition.qmd   # Source-of-truth analysis
|   `-- people_analytics_attrition.html  # Rendered report
|-- app/
|   `-- app.py                           # Streamlit companion
|-- data/raw/
|   `-- HR-Employee-Attrition.csv
|-- src/
|   |-- preprocessing.py                 # Data contract and validation
|   `-- modeling.py                      # Frozen model specification
|-- .gitignore
|-- LICENSE
|-- README.md
`-- requirements.txt
```

The Quarto document is the analytical source of truth. The Python modules reproduce its frozen data contract and final logistic pipeline for the Streamlit application.

## Reproduce the project

### 1. Create an environment

```bash
python -m venv .venv
```

On PowerShell:

```powershell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project was last validated with Python 3.14 and the exact package versions in `requirements.txt`.

### 2. Render the analysis

Install [Quarto](https://quarto.org/docs/get-started/) separately, then run:

```bash
quarto render analysis/people_analytics_attrition.qmd
```

The report executes all Python cells from a fresh kernel and writes `analysis/people_analytics_attrition.html`.

### 3. Run the interactive companion

```bash
streamlit run app/app.py
```

The application presents aggregate evidence, behavioral comparisons, frozen-model validation, and subgroup diagnostics. It intentionally does not expose employee-level scores.

## Interpretation limits

- The dataset is synthetic and cross-sectional; external and temporal validity are unknown.
- Observed differences and model coefficients are associations, not causal effects.
- The five attitude indicators are single ordinal items, not validated multi-item psychological scales.
- Several held-out subgroups contain too few attrition cases for stable fairness conclusions.
- Predictive performance does not establish that deployment would be useful, fair, lawful, or ethical.
- The project should not be used for hiring, discipline, promotion, compensation, or termination decisions.

## Data source

[IBM HR Analytics Employee Attrition & Performance](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset), a synthetic dataset commonly used for educational analysis. No real employee data are included.

## Author

Washington Casamen Nolasco

Psychology, Quantitative Methods, and Behavioral Data Science

[GitHub](https://github.com/Washingtonwlad)

## License

Released under the [MIT License](LICENSE). The dataset remains subject to the terms of its original source.
