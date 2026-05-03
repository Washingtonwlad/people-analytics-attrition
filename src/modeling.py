"""
modeling.py
-----------
Interpretable attrition prediction pipeline.
XGBoost classifier with SHAP explanations.
Extracted from 03_predictive_modeling.ipynb — Phase 3.

Author: Washington Casamen Nolasco
"""

import pandas as pd
import numpy as np
import shap
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score
)


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Prepare feature matrix and target vector for modeling.

    Drops non-numeric columns and separates X from y.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain AttritionBinary column and no raw
        categorical variables (use encode_categoricals first).

    Returns
    -------
    tuple
        (X: pd.DataFrame, y: pd.Series)
    """
    cols_to_drop = [c for c in ['AttritionBinary', 'Attrition', 'Profile']
                    if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df['AttritionBinary']
    return X, y


def split_data(X: pd.DataFrame,
               y: pd.Series,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Stratified train/test split preserving class proportions.

    Stratification is mandatory here: without it, the test set
    may not reflect the 16.1% attrition rate of the full dataset,
    producing artificially inflated or deflated evaluation metrics.

    Parameters
    ----------
    X : pd.DataFrame
    y : pd.Series
    test_size : float
    random_state : int

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )


def train_xgboost(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  scale_pos_weight: float,
                  random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train XGBoost classifier with explicit class imbalance handling.

    Hyperparameter rationale:
    - max_depth=4: conservative depth to prevent overfitting on
      1,470 records
    - learning_rate=0.05: slow learning with 300 estimators for
      stable convergence
    - subsample=0.8, colsample_bytree=0.8: stochastic training
      reduces overfitting by randomizing data and feature subsets
    - scale_pos_weight: penalizes false negatives (missed attrition)
      proportionally to class imbalance ratio (~5.2x)

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    scale_pos_weight : float
        Ratio of negative to positive class counts.
    random_state : int

    Returns
    -------
    xgb.XGBClassifier
        Fitted model.
    """
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=random_state,
        verbosity=0
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model: xgb.XGBClassifier,
                   X_test: pd.DataFrame,
                   y_test: pd.Series) -> dict:
    """
    Evaluate model on test set with imbalance-aware metrics.

    Metrics used:
    - AUC-ROC: probability ranking quality (threshold-independent)
    - Average Precision: area under Precision-Recall curve
    - Classification report: precision, recall, F1 per class

    Accuracy is intentionally excluded: a naive classifier
    predicting 'no attrition' for all employees achieves 83.9%
    accuracy — making it a misleading metric here.

    Parameters
    ----------
    model : xgb.XGBClassifier
    X_test : pd.DataFrame
    y_test : pd.Series

    Returns
    -------
    dict
        Evaluation metrics dictionary.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'auc_roc': roc_auc_score(y_test, y_prob),
        'average_precision': average_precision_score(y_test, y_prob),
        'classification_report': classification_report(
            y_test, y_pred,
            target_names=['Stayed', 'Left']
        ),
        'y_pred': y_pred,
        'y_prob': y_prob
    }


def compute_shap_values(model: xgb.XGBClassifier,
                        X: pd.DataFrame) -> tuple:
    """
    Compute SHAP values using TreeExplainer.

    SHAP (SHapley Additive exPlanations) attributes the model's
    prediction to individual features using Shapley values from
    cooperative game theory. Unlike XGBoost's native feature
    importance (which counts tree splits), SHAP measures actual
    impact on the predicted probability with direction and magnitude.

    Parameters
    ----------
    model : xgb.XGBClassifier
    X : pd.DataFrame
        Feature matrix to explain.

    Returns
    -------
    tuple
        (shap_values: np.ndarray, explainer: shap.TreeExplainer)
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def get_shap_importance(shap_values: np.ndarray,
                        feature_names: list,
                        top_n: int = 15) -> pd.DataFrame:
    """
    Compute global feature importance from SHAP values.

    Uses mean absolute SHAP value per feature — a more reliable
    importance measure than XGBoost's native gain or split count.

    Parameters
    ----------
    shap_values : np.ndarray
    feature_names : list
    top_n : int

    Returns
    -------
    pd.DataFrame
        Features ranked by mean |SHAP| value.
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': np.abs(shap_values).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)

    return importance.head(top_n).reset_index(drop=True)
