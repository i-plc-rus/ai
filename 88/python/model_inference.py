# model_integration.py

import pandas as pd
import joblib

def load_model():
    """Загружает обученную модель."""
    return joblib.load('model/random_forest_model.joblib')

def preprocess_new_data(new_data_path):
    """Обрабатывает новые данные, аналогично обучающим данным."""
    new_data = pd.read_csv(new_data_path)
    new_data = pd.get_dummies(new_data, drop_first=True)
    return new_data

def predict_new_data(new_data_path):
    """Предсказывает отток для новых данных."""
    model = load_model()
    new_data = preprocess_new_data(new_data_path)
    predictions = model.predict(new_data)
    return predictions

if __name__ == "__main__":
    new_data_predictions = predict_new_data('data/new_customer_data.csv')
    print("Predictions for new data:\n", new_data_predictions)