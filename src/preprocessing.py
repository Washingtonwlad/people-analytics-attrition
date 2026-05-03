"""
preprocessing.py
----------------
Data loading, cleaning, and feature engineering pipeline.
Extracted from 01_exploratory_analysis.ipynb — Phase 1.

Author: Washington Casamen Nolasco
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw IBM HR Analytics dataset and apply initial cleaning.

    Removes three constant columns with no analytical value:
    - EmployeeCount: always 1
    - Over18: always 'Y'
    - StandardHours: always 80

    Parameters
    ----------
    filepath : str
        Path to HR-Employee-Attrition.csv

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with 32 variables.
    """
    df = pd.read_csv(filepath)
    df.drop(columns=['EmployeeCount', 'Over18', 'StandardHours'], inplace=True)
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary encode the Attrition target variable.

    Yes -> 1 (left)
    No  -> 0 (stayed)

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with added AttritionBinary column.
    """
    df = df.copy()
    df['AttritionBinary'] = (df['Attrition'] == 'Yes').astype(int)
    return df


def encode_overtime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary encode the OverTime variable for clustering.

    Yes -> 1
    No  -> 0

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Dataframe with added OverTimeBinary column.
    """
    df = df.copy()
    df['OverTimeBinary'] = (df['OverTime'] == 'Yes').astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode all categorical variables for modeling.
    Uses drop_first=True to avoid multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain AttritionBinary column.

    Returns
    -------
    pd.DataFrame
        Fully numeric dataframe ready for ML pipeline.
    """
    df_encoded = pd.get_dummies(
        df.drop(columns=['Attrition']),
        drop_first=True
    )
    return df_encoded


def get_class_imbalance_ratio(y: pd.Series) -> float:
    """
    Compute class imbalance ratio for XGBoost scale_pos_weight.

    ratio = count(negative class) / count(positive class)

    Parameters
    ----------
    y : pd.Series
        Binary target variable (0/1).

    Returns
    -------
    float
        Recommended scale_pos_weight value.
    """
    return (y == 0).sum() / (y == 1).sum()


def variable_taxonomy() -> dict:
    """
    Return the conceptual taxonomy of dataset variables.
    Used for documentation and analytical framing.

    Returns
    -------
    dict
        Variable groups by analytical dimension.
    """
    return {
        'Target': ['Attrition'],
        'Psychometric — Satisfaction': [
            'JobSatisfaction', 'EnvironmentSatisfaction', 'RelationshipSatisfaction'
        ],
        'Psychometric — Engagement': [
            'JobInvolvement', 'WorkLifeBalance'
        ],
        'Workload & Strain': [
            'OverTime', 'BusinessTravel', 'DistanceFromHome'
        ],
        'Career Development': [
            'YearsSinceLastPromotion', 'TrainingTimesLastYear',
            'PercentSalaryHike', 'StockOptionLevel'
        ],
        'Organizational Tenure': [
            'YearsAtCompany', 'YearsInCurrentRole',
            'YearsWithCurrManager', 'TotalWorkingYears'
        ],
        'Compensation': [
            'MonthlyIncome', 'DailyRate', 'HourlyRate', 'MonthlyRate'
        ],
        'Structural': [
            'Department', 'JobRole', 'JobLevel'
        ],
        'Demographic': [
            'Age', 'Gender', 'MaritalStatus', 'EducationField', 'Education'
        ],
        'Background': [
            'NumCompaniesWorked', 'PerformanceRating'
        ]
    }
