import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_data(X_train_path, y_train_path):
    """Загружает тренировочные данные и целевую переменную."""
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    return X_train, y_train


def analyze_data(X, y):
    """
    Выполняет анализ данных:
    - Строит матрицу корреляций, включая целевую переменную.
    - Определяет значимые зависимости и потенциальные кандидаты для feature engineering.
    """
    # Объединяем X и y для анализа корреляций с целевой переменной
    data = pd.concat([X, y], axis=1)

    # Вычисляем корреляции и отображаем их
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Матрица корреляций (включая целевую переменную 'Exited')")
    plt.savefig('correlation_matrix')
    plt.show()

    # Вывод признаков с высокой корреляцией с целевой переменной
    print("Наиболее значимые признаки, коррелирующие с оттоком (Exited):")
    print(correlation_matrix['exited'].sort_values(ascending=False))


def feature_engineering(X):
    """
    Создаёт новые признаки на основе анализа корреляций и гипотез:
    - balancePerProduct: Средний баланс на каждый продукт
    - tenureToAge: Отношение стажа обслуживания к возрасту
    - creditAge: Условный кредитный возраст (кредитный рейтинг / возраст)
    """
    X['balancePerProduct'] = X['balance'] / (X['numofproducts'] + 1)
    X['tenureToAge'] = X['tenure'] / (X['age'] + 1)
    X['creditAge'] = X['creditscore'] / (X['age'] + 1)

    # Масштабирование числовых признаков
    scaler = StandardScaler()
    numeric_features = X.select_dtypes(include=['float64', 'int64']).columns
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    return X


def save_transformed_data(X_train, X_test, y_train, y_test):
    """Сохраняет преобразованные тренировочные и тестовые наборы данных."""
    X_train.to_csv('data/X_train_fe.csv', index=False)
    X_test.to_csv('data/X_test_fe.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)


if __name__ == "__main__":
    X_train, y_train = load_data('data/X_train.csv', 'data/y_train.csv')
    X_test, y_test = load_data('data/X_test.csv', 'data/y_test.csv')
    analyze_data(X_train, y_train)
    # Обработка тренировочных данных
    X_train = feature_engineering(X_train)
    # Применение тех же преобразований к тестовым данным
    X_test = feature_engineering(X_test)
    save_transformed_data(X_train, X_test, y_train, y_test)
