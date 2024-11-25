import pandas as pd
from sklearn.preprocessing import StandardScaler

def feature_engineering(df):
    # Создание новых признаков
    df['TotalTransactions'] = df['NumOfProducts'] * df['Tenure']
    df['BalanceSalaryRatio'] = df['Balance'] / (df['EstimatedSalary'] + 1)  # чтобы избежать деления на 0

    # Масштабирование признаков
    scaler = StandardScaler()
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns

    df[num_cols] = scaler.fit_transform(df[num_cols])

    return df

if __name__ == "__main__":
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')

    X_train = feature_engineering(X_train)
    X_test = feature_engineering(X_test)

    X_train.to_csv('data/X_train_fe.csv', index=False)
    X_test.to_csv('data/X_test_fe.csv', index=False)

