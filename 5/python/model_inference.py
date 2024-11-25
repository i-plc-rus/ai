import pandas as pd

def predict_new_data(model, new_data_path):
    # Загрузка новых данных
    new_data = pd.read_csv(new_data_path)
    
    # Предобработка новых данных (здесь должны быть те же шаги предобработки, что и в основном скрипте)
    # Пример: масштабирование новых данных
    new_data_scaled = scaler.transform(new_data)
    
    # Предсказание
    predictions = model.predict(new_data_scaled)
    
    return predictions

# Пример использования
predictions = predict_new_data(best_model, 'data/new_customer_data.csv')
