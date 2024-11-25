import sys
import numpy as np
import pandas as pd
import joblib
from typing import Union
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from utils import load_model_decision_threshold, load_model
from mylogger import logger


def evaluate_model(model: Pipeline, 
                   X: DataFrame, 
                   y: Union[Series, np.array], 
                   decision_threshold=0.5, 
                   classes=[0, 1]):
    if isinstance(y, Series):
        y = y.to_numpy()

    prediction_probas = model.predict_proba(X)[:, 1]
    predictions = np.where(prediction_probas < decision_threshold, classes[0], classes[1])

    metrics = {} 
    metrics['accuracy'] = accuracy_score(y, predictions)
    metrics['precision'] = precision_score(y, predictions)
    metrics['recall'] = recall_score(y, predictions)
    metrics['f1-score'] = f1_score(y, predictions)
    metrics['roc-auc-score'] = roc_auc_score(y, predictions)
    metrics['report'] = classification_report(y, predictions)
    return metrics


if __name__ == "__main__":
    logger.info("Start model evaluation...")
    
    try:
        X_test = pd.read_csv('data/X_test_fe.csv')
        y_test = pd.read_csv('data/y_test.csv').values.squeeze()
    except FileNotFoundError as ex:
        logger.error(ex)
        sys.exit(1)
    
    threshold = load_model_decision_threshold()
    model = load_model()
    if model is None:
        logger.error("Train model first!")
        sys.exit(1)
        
    metrics = evaluate_model(model, X_test, y_test, threshold)
    for metric, value in metrics.items():
        if metric == 'report':
            print(value)
            continue
        print(f"{metric:15s} = {value}")
        logger.info(f"{metric} = {value}")

    logger.info("Finish model evaluation")