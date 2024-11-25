"""Preprocessing data for training and testing."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple


def load_data(file_path: str) -> pd.DataFrame:
    """Load CSV data from a specified file path."""
    return pd.read_csv(
        file_path, sep=',', skipinitialspace=True, encoding='utf-8'
    )


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Clip outliers in numeric columns."""
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for column in numeric_columns:
        if column != 'Churn':
            lower_bound = df[column].quantile(0.01)
            upper_bound = df[column].quantile(0.99)
            df[column] = np.clip(df[column], lower_bound, upper_bound)
    return df


def check_multicollinearity(
    df: pd.DataFrame, threshold: float = 0.95
) -> pd.DataFrame:
    """Remove columns with high multicollinearity."""
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != 'Churn']
    corr_matrix = df[numeric_columns].corr().abs()

    # Mask upper triangle of the correlation matrix
    upper_triangle_mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    upper = corr_matrix.where(upper_triangle_mask)

    # Identify columns with correlation above the threshold
    to_drop = [
        column for column in upper.columns if any(upper[column] > threshold)
    ]
    return df.drop(columns=to_drop)


def preprocess_data(
    df: pd.DataFrame, is_training: bool = True
) -> pd.DataFrame:
    """Preprocess the data by handling outliers, encoding categories, and
    removing multicollinearity."""
    if is_training and 'Churn' in df.columns:
        df = df.dropna(
            subset=['Churn']
        )  # Drop rows with missing target values

    # Handle outliers in numeric columns
    df = handle_outliers(df)

    # Encode categorical features as dummy variables
    categorical_columns = df.select_dtypes(
        include=['object', 'category']
    ).columns.tolist()
    if 'Churn' in categorical_columns:
        categorical_columns.remove('Churn')
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    # Remove highly correlated features
    df = check_multicollinearity(df)

    return df


def split_data(
    df: pd.DataFrame, target_column: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split the data into training and testing sets."""
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in the DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=0.2, random_state=42)


if __name__ == "__main__":
    # Load the data
    data = load_data('data/customer_data.csv')

    # Ensure the target column 'Churn' is properly encoded
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].astype(int)

    # Preprocess the data
    data = preprocess_data(data)

    # Ensure the target column 'Churn' is still present after preprocessing
    if 'Churn' not in data.columns:
        raise ValueError("Column 'Churn' not found in the preprocessed data")

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(data, 'Churn')

    # Save the processed data
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False, header=['Churn'])
    y_test.to_csv('data/y_test.csv', index=False, header=['Churn'])
