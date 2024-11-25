from sklearn.preprocessing import StandardScaler

def feature_engineering(X_train, X_test):
    # Масштабирование признаков
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled

# Пример использования
X_train_scaled, X_test_scaled = feature_engineering(X_train, X_test)
