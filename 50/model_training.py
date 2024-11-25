import sys
import pandas as pd
from typing import Optional, Tuple, Dict
from pandas import DataFrame, Series
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from utils import load_model, save_model, save_decision_threshold, get_decision_threshold, create_model
from mylogger import logger


def create_default_model() -> Pipeline:
    """Создание модели по-умолчанию"""
    features = {
        'numeric': ['credit_score', 'tenure', 'estimated_salary'],
        'category': ['country', 'gender', 'age_group', 'balance_group', 'products_number', 'credit_card', 'active_member'],
        }

    clf = XGBClassifier(booster='gblinear', 
                        verbosity=1,
                        eta=0.3,
                        feature_selector='cyclic',
                        updater='shotgun',
                        reg_lambda=0,
                        reg_alpha=0,
                        random_state=100)

    model = create_model(clf, features)

    logger.warning("Create default model")
    return model


def train_model(model: Pipeline, 
                X: DataFrame, 
                y: Series, 
                use_gs=False) -> Tuple[Pipeline, Optional[Dict]]:

    if use_gs:
        param_grid = {
            'classifier__n_estimators': [10, 25, 50, 100, 150],
            'classifier__eta': [0.001, 0.01, 0.1, 0.2],
            'classifier__feature_selector': ['cyclic', 'random', ],  # 'greedy'
            'classifier__updater': ['coord_descent'],
            'classifier__scale_pos_weight': [None, 3.90],
        }

        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X, y)

        return grid_search.best_estimator_, grid_search.best_params_

    model.fit(X, y)
    return model, None


if __name__ == "__main__":
    logger.info("Start model training...")
    try:
        X_train = pd.read_csv('data/X_train_fe.csv')
        y_train = pd.read_csv('data/y_train.csv').values.ravel()
    except FileNotFoundError as ex:
        logger.error(ex)
        sys.exit(1)
        
    model = load_model()
    if model is None:
        model = create_default_model()

    model, best_param = train_model(model, X_train, y_train, use_gs=True)
    logger.info(f"Model is trained. Best param: {best_param}")
    
    y_train_pred_proba = model.predict_proba(X_train)[:, 1]

    threshold = get_decision_threshold(y_train, y_train_pred_proba, beta=1)
    
    save_model(model)
    save_decision_threshold(threshold)
    
    logger.info("Finish model training")
