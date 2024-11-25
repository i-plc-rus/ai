import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
import lightgbm as lgb

RANDOM_STATE = 42


def load_data(X_path, y_path):
    """Загружает данные для обучения и проверяет их на корректность."""
    X = pd.read_csv(X_path)
    y = pd.read_csv(y_path).values.ravel()
    if not set(y).issubset({0, 1}):
        raise ValueError("Целевая переменная должна содержать только бинарные метки 0 и 1.")
    return X, y


def define_models():
    """Определяет модели и параметры для поиска наилучших гиперпараметров."""
    models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [50, 100, 150, 200, 300, 400, 500],
                'max_features': ['sqrt', 'log2'],
                'max_depth': [5, 10, 15, 20, 25, 30, None],
                'min_samples_leaf': [1, 2, 4, 5, 10],
                'min_samples_split': [2, 3, 4, 5]
            }
        },
        'LogisticRegression': {
            'model': LogisticRegression(random_state=RANDOM_STATE),
            'param_grid': {
                'C': np.logspace(-4, 4, 20),
                'solver': ['lbfgs', 'newton-cg', 'liblinear', 'sag', 'saga'],
                'max_iter': [100, 200, 300, 400, 500]
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(random_state=RANDOM_STATE),
            'param_grid': {
                'n_estimators': [25, 50, 75, 100, 200],
                'learning_rate': [0.01, 0.015, 0.025, 0.05, 0.1],
                'max_depth': [3, 5, 6, 7, 9, 12, 15, 17, 25],
                'num_leaves': [5, 10, 20, 30, 40, 50]
            }
        }
    }
    return models


def train_and_evaluate_models(X_train, y_train):
    """Обучает и оценивает каждую модель, возвращая модель с наилучшим ROC-AUC."""
    best_score = 0
    best_model = None
    best_model_name = ""

    models = define_models()

    print("Начало обучения моделей...")
    for model_name, model_data in models.items():
        model = model_data['model']
        param_grid = model_data['param_grid']

        print(f"\nМодель: {model_name}")
        random_search = RandomizedSearchCV(
            model, param_distributions=param_grid, n_iter=50, scoring='roc_auc',
            cv=5, random_state=RANDOM_STATE, n_jobs=-1
        )
        random_search.fit(X_train, y_train)
        print(random_search.best_params_)

        # Оценка модели по метрике ROC-AUC
        best_estimator = random_search.best_estimator_
        best_score_model = random_search.best_score_
        print(f"Лучший ROC-AUC для {model_name}: {best_score_model:.4f}")

        # Сохранение наилучшей модели
        if best_score_model > best_score:
            best_score = best_score_model
            best_model = best_estimator
            best_model_name = model_name

    print(f"\nНаилучшая модель: {best_model_name} с ROC-AUC: {best_score:.4f}")
    return best_model, best_model_name


if __name__ == "__main__":
    # Загрузка данных
    X_train, y_train = load_data('data/X_train_fe.csv', 'data/y_train.csv')
    # Обучение моделей и выбор лучшей
    best_model, best_model_name = train_and_evaluate_models(X_train, y_train)
    # Сохранение лучшей модели
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, f'models/{best_model_name}_churn_model.pkl')
    print(f"Наилучшая модель '{best_model_name}' сохранена как 'models/{best_model_name}_churn_model.pkl'")