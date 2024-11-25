import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def load_test_data():
    """Загружает тестовые данные и модель."""
    X_test = pd.read_csv('data/X_test_scaled.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    model = joblib.load('model/random_forest_model.joblib')
    return X_test, y_test, model

def evaluate_model(X_test, y_test, model):
    """Оценивает модель с использованием различных метрик и визуализирует результаты."""
    predictions = model.predict(X_test)
    proba_predictions = model.predict_proba(X_test)[:, 1]

    # Метрики
    print("Accuracy:", accuracy_score(y_test, predictions))
    print("Classification Report:\n", classification_report(y_test, predictions))
    print("ROC-AUC:", roc_auc_score(y_test, proba_predictions))

    # ROC-кривая
    fpr, tpr, thresholds = roc_curve(y_test, proba_predictions)
    plt.plot(fpr, tpr, label="ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="best")
    plt.show()

if __name__ == "__main__":
    X_test, y_test, model = load_test_data()
    evaluate_model(X_test, y_test, model)