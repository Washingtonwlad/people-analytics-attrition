"""
clustering.py
-------------
Behavioral segmentation pipeline using K-Means clustering.
Extracted from 02_behavioral_segmentation.ipynb — Phase 2.

Author: Washington Casamen Nolasco
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# Behavioral variable block used for clustering
CLUSTERING_VARS = [
    'JobSatisfaction',
    'EnvironmentSatisfaction',
    'RelationshipSatisfaction',
    'JobInvolvement',
    'WorkLifeBalance',
    'OverTimeBinary'
]

# Profile names derived from centroid analysis
PROFILE_NAMES = {
    0: 'Strained but Present',
    1: 'Structurally Stable, Relationally Distant',
    2: 'Overloaded at Risk',
    3: 'Engaged and Balanced'
}


def scale_features(df: pd.DataFrame,
                   variables: list = CLUSTERING_VARS) -> tuple:
    """
    Standardize clustering variables to mean=0, std=1.

    K-Means computes Euclidean distances — scaling ensures all
    variables contribute equally regardless of original range.

    Parameters
    ----------
    df : pd.DataFrame
    variables : list
        Variables to scale. Defaults to CLUSTERING_VARS.

    Returns
    -------
    tuple
        (df_scaled: np.ndarray, scaler: StandardScaler)
    """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[variables])
    return df_scaled, scaler


def evaluate_k_range(df_scaled: np.ndarray,
                     k_range: range = range(2, 11),
                     random_state: int = 42) -> pd.DataFrame:
    """
    Evaluate K-Means for a range of k values using inertia
    and Silhouette Score.

    Parameters
    ----------
    df_scaled : np.ndarray
        Scaled feature matrix.
    k_range : range
        Range of k values to evaluate.
    random_state : int

    Returns
    -------
    pd.DataFrame
        Table with k, inertia, and silhouette_score columns.
    """
    results = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(df_scaled)
        results.append({
            'k': k,
            'inertia': kmeans.inertia_,
            'silhouette_score': silhouette_score(df_scaled, labels)
        })
    return pd.DataFrame(results)


def fit_kmeans(df_scaled: np.ndarray,
               n_clusters: int = 4,
               random_state: int = 42) -> KMeans:
    """
    Fit final K-Means model.

    Decision rationale for k=4:
    - k=2 yields highest silhouette (0.206) but insufficient
      organizational resolution
    - k=4 provides 4 interpretable behavioral archetypes with
      stable silhouette (0.163) and substantially reduced inertia
    - Behavioral profiles are validated by distinct attrition rates
      ranging from 8.5% to 29.7%

    Parameters
    ----------
    df_scaled : np.ndarray
    n_clusters : int
    random_state : int

    Returns
    -------
    KMeans
        Fitted KMeans model.
    """
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    kmeans.fit(df_scaled)
    return kmeans


def assign_profiles(df: pd.DataFrame,
                    kmeans: KMeans,
                    df_scaled: np.ndarray,
                    profile_names: dict = PROFILE_NAMES) -> pd.DataFrame:
    """
    Assign cluster labels and profile names to dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    kmeans : KMeans
        Fitted KMeans model.
    df_scaled : np.ndarray
    profile_names : dict
        Mapping from cluster int to profile name string.

    Returns
    -------
    pd.DataFrame
        Dataframe with Cluster and Profile columns added.
    """
    df = df.copy()
    df['Cluster'] = kmeans.predict(df_scaled)
    df['Profile'] = df['Cluster'].map(profile_names)
    return df


def get_centroid_summary(df: pd.DataFrame,
                         variables: list = CLUSTERING_VARS,
                         scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Compute centroid summary in original scale with attrition rate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Cluster, Profile, and AttritionBinary columns.
    variables : list
    scaler : StandardScaler
        If provided, used to document scaling parameters.

    Returns
    -------
    pd.DataFrame
        Centroid table with attrition rate and cluster size.
    """
    summary = df.groupby('Profile')[variables].mean().round(2)
    summary['attrition_rate_%'] = (
        df.groupby('Profile')['AttritionBinary'].mean() * 100
    ).round(1)
    summary['n_employees'] = df.groupby('Profile').size()
    return summary
