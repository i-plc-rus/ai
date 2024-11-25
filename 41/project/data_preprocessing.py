import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

def load_data(file_path):
    """Загрузка данных из файла."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Предобработка данных."""
    
    # Преобразование категориальных переменных
    categorical_features = ['gender', 'country']
    numerical_features = ['age', 'income']
    
    # Заполнение пропущенных значений числовых признаков медианой
    imputer_numerical = SimpleImputer(strategy='median')
    
    # Заполнение пропущенных значений категориальных признаков наиболее частым значением
    imputer_categorical = SimpleImputer(strategy='most_frequent')
    
    # Создание конвейера для обработки числовых признаков
    num_pipeline = Pipeline([
        ('imputer', imputer_numerical),
    ])
    
    # Создание конвейера для обработки категориальных признаков
    cat_pipeline = Pipeline([
        ('imputer', imputer_categorical),
        ('onehotencoder', OneHotEncoder(sparse=False))
    ])
    
    # Объединение всех преобразований в одном объекте
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),
            ('cat', cat_pipeline, categorical_features)
        ]
    )
    
    # Применение преобразования ко всем данным
    df_processed = pd.DataFrame(preprocessor.fit_transform(df), index=df.index)
    
    return df_processed

if __name__ == "__main__":
    file_path = ''
    df = load_data(file_path)
    df_processed = preprocess_data(df)