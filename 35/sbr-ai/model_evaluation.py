"""Evaluate model."""

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data_preprocessing import preprocess_data
from feature_engineering import feature_engineering


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series):
    """Evaluate the performance of the trained model on test data."""
    X_test = preprocess_data(X_test, is_training=False)
    X_test = feature_engineering(X_test, is_training=False)

    # Load feature names used in training
    feature_names = joblib.load('models/feature_names.pkl')
    X_test = X_test[feature_names]

    # Make predictions
    predictions = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    # Calculate accuracy
    accuracy = accuracy_score(y_test, predictions)

    # Compute ROC AUC for binary classification
    unique_classes = len(set(y_test))
    if unique_classes == 2:
        auc = roc_auc_score(y_test, proba)
    else:
        print(
            f"Предупреждение: в y_test присутствует {unique_classes} класс. "
            "ROC AUC не может быть вычислен."
        )
        auc = None

    # Classification report and confusion matrix
    report = classification_report(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)

    # Confusion matrix visualization
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.close()

    return accuracy, auc, report, conf_matrix


if __name__ == "__main__":
    # Load test data
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')['Churn']

    # Load the trained model
    model = joblib.load('models/churn_model.pkl')

    # Evaluate the model
    accuracy, auc, report, conf_matrix = evaluate_model(model, X_test, y_test)

    # Print evaluation results
    print(f"Accuracy: {accuracy}")
    if auc is not None:
        print(f"AUC: {auc}")
    print(f"Classification Report:\n{report}")
    print(f"Confusion Matrix:\n{conf_matrix}")

    # Print class distribution in the test data
    print(f"\nРаспределение классов в y_test:")
    print(y_test.value_counts())
