from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    # Определение модели
    rf = RandomForestClassifier(random_state=42)

    # Гиперпараметры для настройки
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # GridSearch для настройки параметров
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    
    # Лучшая модель
    best_model = grid_search.best_estimator_
    
    return best_model

# Пример использования
best_model = train_model(X_train_scaled, y_train)
