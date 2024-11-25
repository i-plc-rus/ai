import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
import joblib

def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def train_model(X_train, y_train):
    # Обработка несбалансированности классов
    X_train, y_train = handle_imbalance(X_train, y_train)

    # Создаем модель случайного леса
    model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Настройка гиперпараметров
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    random_search = RandomizedSearchCV(
        model, param_distributions=param_dist, 
        n_iter=20, cv=5, scoring='roc_auc', 
        n_jobs=-1, random_state=42
    )
    
    random_search.fit(X_train, y_train)

    return random_search.best_estimator_

if __name__ == "__main__":
    X_train = pd.read_csv('data/X_train_fe.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()

    best_model = train_model(X_train, y_train)

    # Оценка модели на тренировочной выборке
    y_train_pred = best_model.predict(X_train)
    roc_auc = roc_auc_score(y_train, y_train_pred)
    print(f"ROC-AUC на тренировочной выборке: {roc_auc}")

    # Сохранение лучшей модели
    joblib.dump(best_model, 'models/churn_model.pkl')
