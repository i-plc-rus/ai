from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    # Предсказания
    y_pred = model.predict(X_test)
    
    # Отчет классификации
    print(classification_report(y_test, y_pred))
    
    # ROC-AUC
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    print(f'ROC-AUC Score: {roc_auc}')

    # ROC кривая
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Пример использования
evaluate_model(best_model, X_test_scaled, y_test)
