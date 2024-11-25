import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.pipeline import Pipeline
from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering
from utils import load_model, load_model_decision_threshold
from mylogger import logger


def predict_churn(model: Pipeline, new_data: DataFrame, decision_threshold: 0.5, classes=[0, 1]):
    new_data = preprocess_data(new_data)
    new_data = feature_engineering(new_data)
    
    prediction_probas = model.predict_proba(new_data)[:, 1]
    predictions = np.where(prediction_probas < decision_threshold, classes[0], classes[1])
    return predictions


if __name__ == "__main__":
    logger.info("Start model inference...")

    try:
        new_data = pd.read_csv('data/new_customer_data.csv')
    except FileNotFoundError as ex:
        logger.error(ex)
        sys.exit(1)
        
    model = load_model()
    if model is None:
        logger.error("Train model first!")
        sys.exit(1)

    threshold = load_model_decision_threshold()
    predictions = predict_churn(model, new_data, threshold)
    new_data['ChurnPrediction'] = predictions

    new_data.to_csv('data/new_customer_predictions.csv', index=False)
    logger.info("Finish model inference")
