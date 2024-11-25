import numpy as np
import plotly.express as px
import joblib
from pandas import DataFrame
from typing import Optional
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from mylogger import logger


def load_model(path='./models/churn_model.pkl') -> Optional[Pipeline]:
    try:
        model = joblib.load(path)
        logger.info(f"Model was uploaded from [{path}]")
        return model
    except FileNotFoundError as ex:
        logger.error(ex)
    return None


def save_model(model: Pipeline, path='./models/churn_model.pkl'):
    try:
        if joblib.dump(model, path):
            logger.info(f"Model is written to [{path}]")
            return
        logger.error("Write exception! File: [{path}]!")
    except Exception as ex:
        logger.error(ex)


def save_decision_threshold(threshold: float, path='./models/model_threshold.txt'):
    with open(path, 'wt') as f:
        f.write(f"{threshold}\n")
    logger.info(f"Decision threshold [{threshold}] was written to [{path}]")    


def load_model_decision_threshold(path='./models/model_threshold.txt') -> float:
    # load decision threshold for model predictions
    try:
        try:
            with open(path, 'rt') as f:
                text = f.read().strip()
                if text:
                    threshold = float(text)
                    logger.info(f"Decision threshold was loaded from [{path}]. Value is: [{threshold}]]")
                else:
                    raise ValueError
        except FileNotFoundError as ex:
            logger.error(ex)
            raise ValueError
        except IOError as ex:
            logger.error(ex)
            raise ValueError
    except ValueError as ex:
        logger.error(ex)
        threshold = 0.5
        logger.warning(f"Принудительно установлено значение: {threshold}")
    return threshold


def display_correlation_matrix(cm: DataFrame, title="Матрица корреляции (Пирсона)"):
    dims = [col.replace('_', ' ').capitalize() for col in cm.columns.values]

    fig = px.imshow(cm, 
                    x=dims, 
                    y=dims, 
                    color_continuous_scale='Reds', 
                    aspect="auto")
    fig.update_traces(text=cm, texttemplate="%{text:0.3f}")
    fig.update_xaxes(side="bottom")

    fig.update_layout(title=title,
                      dragmode='select', 
                      width=1000, 
                      height=1000, 
                      hovermode='closest',
                      template='seaborn')
    fig.show()    


def get_decision_threshold(y_true, y_pred_score, beta=1):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_score)
    fscores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls)

    idx = np.argmax(fscores)
    threshold = thresholds[idx]
    return threshold


def display_precision_recall_curve(y_true, y_pred_score, beta=1):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_score)
    fscores = (1 + beta**2) * (precisions * recalls) / (beta**2 * precisions + recalls)

    idx = np.argmax(fscores)
    threshold = thresholds[idx]

    fig = px.line(x=recalls, 
                  y=precisions, 
                  title=f"Precision-recall curve (max f-score={fscores[idx]:.03f} beta={beta})")

    fig.add_hline(y=precisions[idx], 
                  line_width=2, 
                  line_dash="dash", 
                  line_color="red",
                  annotation_text=f"Precision: {precisions[idx]:.3f}", 
                  annotation_position="bottom right")

    fig.add_vline(x=recalls[idx], 
                  line_width=2, 
                  line_dash="dot", 
                  line_color="red",
                  annotation_text=f"Recall: {recalls[idx]:.3f}", 
                  annotation_position="top right")

    fig.add_annotation(x=recalls[idx], 
                       y=precisions[idx],
                       showarrow=True, 
                       text=f"Decision threshold={thresholds[idx]:.4f}")

    fig.update_layout(xaxis_title='Recall',
                      yaxis_title='Precision',
                      dragmode='select', 
                      width=900, 
                      height=900, 
                      hovermode='closest',
                      template='seaborn')

    fig.show()
    return threshold


def display_roc_curve(y_true, y_pred_score, threshold=0.5):
    y_pred = np.where(y_pred_score < threshold, 0, 1)
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_score)
    
    fig = px.line(x=fpr, 
                  y=tpr, 
                  title=f"ROC-AUC curve (score: {roc_auc_score(y_true, y_pred):0.3f})", )

    fig.add_scatter(x=[0, 1], 
                    y=[0, 1], 
                    legendgroup='', 
                    name='Random', 
                    line={'color': 'red', 'dash': 'solid'}, 
                    showlegend=False, )
    
    fig.update_layout(xaxis_title='False positive rate',
                      yaxis_title='True positive rate',
                      dragmode='select', 
                      width=900, 
                      height=900, 
                      hovermode='closest',
                      template='seaborn')

    fig.show()


def display_confusion_matrix(y_true, y_pred, labels=[0, 1], width=600):
    cm = confusion_matrix(y_true, y_pred)
    
    dims = [str(l) for l in labels]

    fig = px.imshow(cm, 
                    x=dims, 
                    y=dims, 
                    color_continuous_scale='Reds', 
                    aspect="auto")

    fig.update_traces(text=cm, texttemplate="%{text}")

    fig.update_layout(title="Confusion matrix",
                      xaxis_title='Predicted',
                      yaxis_title='True',
                      dragmode='select', 
                      width=width, 
                      height=width, 
                      hovermode='closest',
                      template='seaborn')
    fig.show()


def create_model(clf, features: dict) -> Pipeline:
    transformers = []

    if features['numeric']:
        transformers.append(("num", 
                             Pipeline(steps=[("scaler", StandardScaler()),]), 
                             features['numeric']))

    if features['category']:
        transformers.append(("cat", 
                             Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="error")),]),
                             features['category']))

    if transformers:
        model = Pipeline(steps=[("transformer", ColumnTransformer(transformers=transformers, 
                                                                  remainder='drop')), 
                                ("classifier", clf)])
        return model
    
    model = Pipeline(steps=[("classifier", clf)])
    return model
