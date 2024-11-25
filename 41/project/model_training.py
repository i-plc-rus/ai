import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def load_and_split_data(file_path):
    """Загрузка и разделение данных на обучающую и тестовую выборки."""
    df = pd.read_csv(file_path)
    
    target = df['churn']
    features = df.drop('churn', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """Обучение модели логистической регрессии."""
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model

def train_random_forest(X_train, y_train):
    """Обучение модели случайного леса."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Оценка модели."""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    auc = roc_auc_score(y_test, predictions)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc': auc
    }
    
    return metrics

def grid_search_cv(model, param_grid, X_train, y_train):
    """Поиск оптимальных гиперпараметров с помощью GridSearchCV."""
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    print(f'Best parameters: {best_params}')
    print(f'Best score: {best_score:.4f}')
    
    return grid_search.best_estimator_

if __name__ == "__main__":
    file_path = 'data/clients_data.csv'
    
    X_train, X_test, y_train, y_test = load_and_split_data(file_path)
    
    # Логистическая регрессия
    lr_model = train_logistic_regression(X_train, y_train)
    lr_metrics = evaluate_model(lr_model, X_test, y_test)
    print("Logistic Regression Metrics:")
    for key, value in lr_metrics.items():
        print(f'{key}: {value:.4f}')
        
    # Случайный лес
    rf_model = train_random_forest(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print("\nRandom Forest Metrics:")
    for key, value in rf_metrics.items():
        print(f'{key}: {value:.4f}')
        
    # Поиск лучших параметров для случайного леса
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    
    tuned_rf_model = grid_search_cv(RandomForestClassifier(), param_grid, X_train, y_train)
    tuned_rf_metrics = evaluate_model(tuned_rf_model, X_test, y_test)
    print("\nTuned Random Forest Metrics:")
    for key, value in tuned_rf_metrics.items():
        print(f'{key}: {value:.4f}')