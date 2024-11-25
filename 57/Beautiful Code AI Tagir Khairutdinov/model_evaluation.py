import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    # ROC-кривая
    fpr, tpr, _ = roc_curve(y_test, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Model')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

    # Важность признаков
    feature_importances = pd.Series(model.feature_importances_, index=X_test.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title('Feature Importances')
    plt.xlabel('Features')
    plt.ylabel('Importance Score')
    plt.grid(True)
    plt.show()

    return accuracy, roc_auc, report

if __name__ == "__main__":
    X_test = pd.read_csv('data/X_test_fe.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    model = joblib.load('models/churn_model.pkl')
    accuracy, roc_auc, report = evaluate_model(model, X_test, y_test)

    print(f"Accuracy: {accuracy}")
    print(f"ROC-AUC: {roc_auc}")
    print(f"Classification Report:\n{report}")

