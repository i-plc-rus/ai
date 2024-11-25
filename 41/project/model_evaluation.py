import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, plot_confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Оценка модели."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    cm = confusion_matrix(y_test, predictions)
    cr = classification_report(y_test, predictions, output_dict=True)
    
    return cm, cr, probabilities

def plot_confusion_matrix(cm, classes):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, ax=ax)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.xaxis.set_ticklabels(classes)
    ax.yaxis.set_ticklabels(classes, rotation=90)
    plt.title('Confusion Matrix')
    plt.show()

def plot_precision_recall_curve(probabilities, y_true):
    from sklearn.metrics import PrecisionRecallCurve
    precisions, recalls, thresholds = precision_recall_curve(y_true, probabilities)
    
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.show()

if __name__