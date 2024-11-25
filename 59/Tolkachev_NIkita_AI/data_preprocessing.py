import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42

def load_data(file_path):
    """Загружает данные из CSV файла."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Очищает данные от дубликатов и аномалий, преобразует заголовки столбцов."""
    print("Количество пропущенных значений в каждом столбце:")
    print(df.isnull().sum())
    df = df.drop_duplicates()
    df.columns = df.columns.str.lower()
    df = df.drop(columns=['rownumber', 'customerid', 'surname'])
    df = df[df['balance'] >= 0]
    df = df[df['creditscore'] > 432]
    df = df[df['age'] < 72]
    return df

def check_target_values(y):
    """Проверяет, что целевая переменная содержит только бинарные значения 0 и 1."""
    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Целевая переменная содержит непрерывные значения. Ожидаются бинарные метки 0 и 1.")

def scale_features(X):
    """Масштабирует числовые признаки для улучшения работы модели."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled, columns=X.columns)

def preprocess_data(df, target_column):
    """Полная предобработка данных с кодированием категорий и масштабированием признаков."""
    df = clean_data(df)
    df = pd.get_dummies(df, drop_first=True)
    check_target_values(df[target_column])
    y = df[target_column]
    X = df.drop(columns=[target_column])
    X = scale_features(X)
    return X, y

def split_and_save_data(X, y, test_size=0.2, random_state=RANDOM_STATE):
    """Разделяет и сохраняет данные на обучающие и тестовые наборы."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

if __name__ == "__main__":
    # Основной этап загрузки и предобработки данных
    data = load_data('data/customer_data.csv')
    X, y = preprocess_data(data, 'exited')
    split_and_save_data(X, y)
