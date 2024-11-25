# model_training.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def load_processed_data():
    """Загружает обработанные тренировочные и тестовые выборки."""
    X_train = pd.read_csv('data/X_train_scaled.csv')
    X_test = pd.read_csv('data/X_test_scaled.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Обучает модель с использованием гиперпараметрической настройки."""
    rf = RandomForestClassifier(random_state=42)

    # Гиперпараметры для настройки
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    # Поиск лучших гиперпараметров
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()
    model = train_model(X_train, y_train)

    # Сохраняем обученную модель
    joblib.dump(model, 'model/random_forest_model.joblib')
