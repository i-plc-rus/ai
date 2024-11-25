import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)


def load_best_model(models_dir='models'):
    """Находит и загружает лучшую модель из указанной директории."""
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_churn_model.pkl')]
    if not model_files:
        raise FileNotFoundError("Не найдено ни одной модели в директории 'models'. Убедитесь, что обучение завершено.")

    best_model_path = os.path.join(models_dir, model_files[0])
    print(f"Загружается модель: {best_model_path}")
    return joblib.load(best_model_path)


# Загрузка модели при запуске сервера
model = load_best_model()


@app.route('/predict', methods=['POST'])
def predict():
    """
    Принимает JSON с данными клиента и возвращает предсказание оттока.
    Ожидаемый формат входных данных:
    {
        "feature1": значение,
        "feature2": значение,
        ...
    }
    """
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Нет данных для предсказания'}), 400

    # Преобразование входных данных в DataFrame
    input_df = pd.DataFrame([data])

    # Проверка и добавление недостающих столбцов
    model_features = model.booster_.feature_name() if hasattr(model, "booster_") else model.feature_names_in_
    for feature in model_features:
        if feature not in input_df.columns:
            input_df[feature] = 0

    # Упорядочивание столбцов
    input_df = input_df[model_features]

    # Получение предсказания
    prediction = model.predict_proba(input_df)[:, 1][0]  # Вероятность оттока

    # Формирование ответа
    response = {
        'churn_probability': prediction,
        'churn_risk': 'high' if prediction > 0.5 else 'low'
    }

    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)