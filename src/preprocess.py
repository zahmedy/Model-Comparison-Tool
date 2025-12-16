from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def _identify_id_like_columns(df: pd.DataFrame, target_col: str) -> List[str]:
    """
    Detect likely identifier columns:
    - column name contains 'id' or 'uuid'
    - or has near-unique values (90%+ unique)
    """
    id_cols: List[str] = []
    n_rows = len(df)
    for col in df.columns:
        if col == target_col:
            continue
        col_lower = col.lower()
        if "id" in col_lower or "uuid" in col_lower:
            id_cols.append(col)
            continue
        if n_rows > 0:
            uniq_ratio = df[col].nunique(dropna=True) / n_rows
            if uniq_ratio > 0.9:
                id_cols.append(col)
    return list(dict.fromkeys(id_cols))


def _prepare_features(
    df: pd.DataFrame,
    target_col: str
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Data-agnostic preprocessing:
    - drop duplicates and rows missing the target
    - remove ID-like columns
    - drop columns with >50% missing values
    - impute numeric (median) and categorical/bool (mode) missing values
    - one-hot encode categoricals so models receive numeric features
    """
    df = df.copy()
    df = df.drop_duplicates()
    df = df.dropna(subset=[target_col])

    inferred_drop_cols = _identify_id_like_columns(df, target_col=target_col)

    feature_df = df.drop(columns=[target_col] + inferred_drop_cols, errors="ignore")

    # Drop columns with excessive missingness
    missing_ratio = feature_df.isna().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.5].index.tolist()
    feature_df = feature_df.drop(columns=cols_to_drop, errors="ignore")

    numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]

    if numeric_cols:
        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(feature_df[numeric_cols].median())

    for col in categorical_cols:
        mode = feature_df[col].mode()
        fill_value = mode.iloc[0] if not mode.empty else ""
        feature_df[col] = feature_df[col].fillna(fill_value)

    feature_df = pd.get_dummies(feature_df, columns=categorical_cols, drop_first=True)
    feature_cols = feature_df.columns.tolist()
    y = df[target_col]
    return feature_df, y, feature_cols


def split_data(
    df: pd.DataFrame,
    target_col: str = "cluster_id",
    test_size: float = 0.2,
    random_state: int = 42
):
    """Preprocess data and split into train and test sets."""

    X, y, feature_cols = _prepare_features(df, target_col=target_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return X_train, X_test, y_train, y_test, feature_cols
