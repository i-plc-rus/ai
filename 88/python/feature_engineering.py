# feature_engineering.py

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_split_data():
    """Загружает подготовленные тренировочные и тестовые выборки."""
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    return X_train, X_test, y_train, y_test

def feature_engineering(X_train, X_test):
    """Создает новые признаки и масштабирует данные."""
    # Пример создания новых признаков
    X_train['BalanceSalaryRatio'] = X_train['Balance'] / X_train['EstimatedSalary']
    X_test['BalanceSalaryRatio'] = X_test['Balance'] / X_test['EstimatedSalary']

    # Масштабируем данные
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    return X_train, X_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_split_data()
    X_train, X_test = feature_engineering(X_train, X_test)

    # Сохраняем обработанные данные
    X_train.to_csv('data/X_train_scaled.csv', index=False)
    X_test.to_csv('data/X_test_scaled.csv', index=False)