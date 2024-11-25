import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from matplotlib.animation import FuncAnimation

# Функция для загрузки данных
def load_data(file_path):
    """Загрузка данных из CSV файла."""
    return pd.read_csv(file_path)

# Функция для предобработки данных
def preprocess_data(df):
    """Предобработка данных: очистка, создание признаков и кодирование."""
    
    # 1. Обработка пропусков
    df.fillna(df.mean(), inplace=True)  # Заполнение числовых пропусков средним
    df.fillna('Unknown', inplace=True)  # Заполнение категориальных пропусков

    # 2. Создание новых признаков
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['new_feature'] = df['feature1'] * df['feature2']  # Пример создания нового признака

    # 3. Удаление выбросов с использованием межквартильного диапазона (IQR)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    return df

# Функция для визуализации распределения целевой переменной
def plot_target_distribution(y):
    """Визуализация распределения целевой переменной."""
    sns.countplot(x=y)
    plt.title('Распределение целевой переменной')
    plt.show()

# Функция для визуализации корреляционной матрицы
def plot_correlation_matrix(X):
    """Визуализация корреляционной матрицы."""
    plt.figure(figsize=(10, 8))
    correlation = X.corr()
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Корреляционная матрица')
    plt.show()

# Функция для анимации обучения модели
def animate_training_progress(model, X_train, y_train):
    """Анимация процесса обучения модели."""
    fig, ax = plt.subplots()
    
    def update(frame):
        ax.clear()
        model.fit(X_train[:frame], y_train[:frame])  # Обучаем модель на первых frame примерах
        y_pred = model.predict(X_train)
        ax.scatter(range(len(y_pred)), y_pred, color='blue', label='Предсказания')
        ax.scatter(range(len(y_train)), y_train, color='red', alpha=0.5, label='Истинные значения')
        ax.set_title(f'Обучение модели на {frame} примерах')
        ax.legend()
    
    ani = FuncAnimation(fig, update, frames=range(10, len(X_train), 10), repeat=False)
    plt.show()

# Функция для обучения модели
def train_model(X, y):
    """Обучение модели Random Forest."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

# Функция для оценки модели
def evaluate_model(model, X_test, y_test):
    """Оценка модели на тестовых данных."""
    y_pred = model.predict(X_test)
    
    print("Матрица ошибок:")
    print(confusion_matrix(y_test, y_pred))
    
    print("nОтчет о классификации:")
    print(classification_report(y_test, y_pred))
    
    # Дополнительные метрики
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    f1 = f1_score(y_test, y_pred)
    
    print(f"ROC AUC: {roc_auc:.2f}")
    print(f"F1 Score: {f1:.2f}")

# Функция для кросс-валидации модели
def cross_validate_model(model, X, y):
    """Кросс-валидация модели."""
    scores = cross_val_score(model, X, y, cv=5)
    print(f"Кросс-валидация: Средняя точность: {scores.mean():.2f}, Стандартное отклонение: {scores.std():.2f}")

# Основная функция
def main(file_path):
    """Основная функция для выполнения всех этапов."""
    df = load_data(file_path)
    
    # Предобработка данных
    df = preprocess_data(df)


    # Разделение на признаки и целевую переменную
    X = df.drop('target', axis=1)  # Замените 'target' на имя вашей целевой переменной
    y = df['target']
    
    # Визуализация распределения целевой переменной
    plot_target_distribution(y)
    
    # Визуализация корреляционной матрицы
    plot_correlation_matrix(X)
    
    # Кодирование категориальных переменных
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(exclude=['object']).columns
    
    # Применение OneHotEncoder к категориальным переменным и StandardScaler к числовым
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )
    
    # Применение преобразований и разделение данных на тренировочную и тестовую выборки
    X_processed = preprocessor.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    
    # Обучение модели с анимацией
    model = RandomForestClassifier(random_state=42)
    animate_training_progress(model, X_train, y_train)
    
    # Оценка модели
    evaluate_model(model, X_test, y_test)
    
    # Кросс-валидация модели
    cross_validate_model(model, X_processed, y)

# Запуск программы
if __name__ == "__main__":
    main('data.csv')  # Замените 'data.csv' на путь к вашему файлу с данными
