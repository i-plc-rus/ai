import sys
import pandas as pd
import numpy as np
from typing import Optional, Tuple, Dict
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from mylogger import logger


def load_data(file_path: str) -> Optional[DataFrame]:
    """Загрузка данных"""
    try:
        df = pd.read_csv(file_path, header=0, index_col=None)
        logger.info(f"Data was loaded from file: [{file_path}] Shape: [{df.shape}]")
        return df
    except FileNotFoundError as ex:
        logger.error(ex)
    return None


def preprocess_data(df: DataFrame) -> DataFrame:
    """Препроцессинг данных"""
    # данные для label encoders 
    label_encoders = {'gender': {'Male': 1, 'Female': 2},
                      'country': {'France': 1, 'Spain': 2, 'Germany': 3}
                      }
    logger.info(f"Данные для label encoders: {label_encoders}")

    # drop columns
    df = df.drop(columns='customer_id')
    
    # проверка на пропущенные значения и замена их на медианные значения
    mis = df.isna().sum()
    mis = mis[mis > 0]
    for col in mis.index:
        if df[col].dtype == 'O':
            # column is object - find most popular value 
            value = df[col].value_counts().sort_values().index[-1]
        else:
            value = df[col].median()
        df[col] = df[col].fillna(value)
        logger.info(f"Column [{col}]: fill NAN with value: [{value}]")

    # label encoders
    for label, mapping in label_encoders.items():
        df[label] = df[label].map(mapping).astype(np.int16)

    return df


def split_data(df: DataFrame, target_column: str, test_size=0.2) -> Tuple[DataFrame, DataFrame, Series, Series]:
    """Разбиение данных"""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=df[target_column])


if __name__ == "__main__":
    logger.info("Start preprocessing...")
    data = load_data('./data/Bank Customer Churn Prediction copy.csv')
    if data is None or data.shape[0] == 0:
        logger.error("No data!")
        sys.exit(1)
    
    data = preprocess_data(data)
    X_train, X_test, y_train, y_test = split_data(data, 'churn')
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    logger.info("Finish preprocessing")
