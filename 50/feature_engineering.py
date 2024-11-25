import sys
import numpy as np
import pandas as pd
from typing import Dict
from pandas import DataFrame
from mylogger import logger


def feature_engineering(df: DataFrame) -> DataFrame:
    """Создание новых признаков"""
    new_features = {
        'balance': {
            'name': 'balance_group',
            'labels': [1, 2, 3],
            'bins': [-np.inf, 38000.0, 132000.0, np.inf],
            'display_labels': ['<38000', '38000-132000', '>132000']
        },
                            
        'age': {
            'name': 'age_group',
            'labels': [1, 2, 3, 4, 5],
            'bins': [0, 31.0, 38.0, 41.0, 51.0, np.inf],
            'display_labels': ['<31', '31-38', '38-41', '41-51' '>51']
        },
    }

    # разбивка значений признаков на группы
    for column, group in new_features.items():        
        df[group['name']] = pd.cut(df[column],  
                                   bins=group['bins'], 
                                   labels=group['labels']).astype(np.uint8)
        logger.info(f"New feature: [{group['name']}] base on [{column}]: "
                    f"bins: {group['bins']}, labels: {group['labels']}")        
    return df


if __name__ == "__main__":
    logger.info("Start feature engineering...")
    try:
        X_train = pd.read_csv('./data/X_train.csv')
        X_test = pd.read_csv('./data/X_test.csv')
    except FileNotFoundError as ex:
        logger.error(ex)
        sys.exit(1)
                
    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    X_train.to_csv('./data/X_train_fe.csv', index=False)
    X_test.to_csv('./data/X_test_fe.csv', index=False)
    logger.info("Finish feature engineering")
