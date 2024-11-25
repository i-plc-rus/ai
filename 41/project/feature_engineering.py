import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA

def create_new_features(df):
    """Создание новых признаков."""
    
    # Добавление нового признака, который может оказаться полезным
    df['total_spend'] = df['monthly_spend_1'] + df['monthly_spend_2'] + df['monthly_spend_3']
    
    return df

def feature_selection(df, target):
    """
    Отбор признаков с использованием метода SelectKBest.
    :param df: DataFrame с признаками
    :param target: целевая переменная
    """
    
    selector = SelectKBest(score_func=f_classif, k=10)
    selected_features = selector.fit_transform(df, target)
    
    indices = selector.get_support(indices=True)
    features_selected = df.columns[indices]
    
    print("Selected Features:", features_selected)
    
    X_selected = pd.DataFrame(selected_features, columns=features_selected)
    
    return X_selected

def dimensionality_reduction(df):
    pca = PCA(n_components=0.95)
    reduced_df = pca.fit_transform(df)
    
    reduced_df = pd.DataFrame(reduced_df)
    
    explained_variance = pca.explained_variance_ratio_
    print(f'Explained variance ratio: {explained_variance}')
    
    return reduced_df

if __name__ == "__main__":
    file_path = 'path_to_your_data.csv'
    df = pd.read_csv(file_path)
    
    df_with_new_features = create_new_features(df)
    
    target = df_with_new_features['churn']
    features = df_with_new_features.drop('churn', axis=1)
    
    selected_features_df = feature_selection(features, target)
    final_df = dimensionality_reduction(selected_features_df)