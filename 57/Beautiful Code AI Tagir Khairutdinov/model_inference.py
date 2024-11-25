import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

def load_model(model_path):
    return joblib.load(model_path)

def preprocess_new_data(new_data):
    # Предполагается, что новые данные требуют таких же преобразований
    new_data['TotalTransactions'] = new_data['NumOfProducts'] * new_data['Tenure']
    new_data['BalanceSalaryRatio'] = new_data['Balance'] / (new_data['EstimatedSalary'] + 1)
    
    # Масштабирование признаков
    scaler = StandardScaler()
    num_cols = new_data.select_dtypes(include=['float64', 'int64']).columns
    new_data[num_cols] = scaler.fit_transform(new_data[num_cols])

    return new_data

def predict_churn(model, new_data):
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)[:, 1]  # вероятность оттока
    return predictions, probabilities

if __name__ == "__main__":
    model = load_model('models/churn_model.pkl')
    new_data = pd.read_csv('data/new_customer_data.csv')

    # Предобработка новых данных
    new_data = preprocess_new_data(new_data)

    # Прогнозирование
    predictions, probabilities = predict_churn(model, new_data)
    new_data['ChurnPrediction'] = predictions
    new_data['ChurnProbability'] = probabilities

    # Сохранение результатов
    new_data.to_csv('data/new_customer_predictions.csv', index=False)

    print("Прогнозирование завершено. Результаты сохранены в 'data/new_customer_predictions.csv'.")
