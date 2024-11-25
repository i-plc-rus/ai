import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def preprocess_data(file_path):
    # Загрузка данных
    data = pd.read_csv(file_path)

    # Удаление пропусков или их заполнение
    data.fillna(data.median(), inplace=True)

    # Преобразование категориальных признаков
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    
    # Разделение на тренировочную и тестовую выборки
    X = data.drop('Exited', axis=1)
    y = data['Exited']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Пример вызова функции
X_train, X_test, y_train, y_test = preprocess_data('data/customer_data.csv')
