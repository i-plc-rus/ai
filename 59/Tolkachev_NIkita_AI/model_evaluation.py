import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import os


def load_best_model(models_dir='models'):
    """Находит и загружает лучшую модель из указанной директории."""
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_churn_model.pkl')]
    if not model_files:
        raise FileNotFoundError("Не найдено ни одной модели в директории 'models'. Убедитесь, что обучение завершено.")

    # Загрузка последней сохранённой модели
    best_model_path = os.path.join(models_dir, model_files[0])
    print(f"Загружается модель: {best_model_path}")
    return joblib.load(best_model_path)

def load_data(X_test_path, y_test_path):
    """Загружает тестовые данные и проверяет наличие всех признаков."""
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    return X_test, y_test

def load_model(model_path):
    """Загружает обученную модель."""
    return joblib.load(model_path)

def evaluate_model(model, X_test, y_test):
    """Оценивает качество модели и выводит метрики производительности"""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    # Метрики
    accuracy = accuracy_score(y_test, predictions)
    classification_rep = classification_report(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)
    conf_matrix = confusion_matrix(y_test, predictions)
    print(f"Точность модели: {accuracy}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Отчет по классификации:\n{classification_rep}")
    return accuracy, classification_rep, roc_auc, conf_matrix, probabilities

def plot_confusion_matrix(conf_matrix):
    """Визуализация матрицы ошибок."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

def plot_roc_curve(y_test, probabilities):
    """Визуализация ROC-кривой."""
    fpr, tpr, _ = roc_curve(y_test, probabilities)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc_score(y_test, probabilities):.2f})')
    plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Основной этап оценки модели
    X_test, y_test = load_data('data/X_test_fe.csv', 'data/y_test.csv')
    model = load_best_model()
    accuracy, classification_rep, roc_auc, conf_matrix, probabilities = evaluate_model(model, X_test, y_test)
    plot_confusion_matrix(conf_matrix)
    plot_roc_curve(y_test, probabilities)
