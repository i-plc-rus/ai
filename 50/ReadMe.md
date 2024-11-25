# Модель предсказания оттока клиентов

## Данные

Данные для исследования взяты с ресурса:

[Bank Customer ChurnPrediction](https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/data)

Для демонстрации работы препроцессинга, в данные внесены пропуски.

Для работы используется файл [Данные](data/Bank Customer Churn Prediction copy.csv)

## Среда выполнения

Использовался python версии 3.11, установленный в виртуальное окружение.

Пакеты, используемые при работе перечислены в [requirements.txt](requirements.txt)

## Файлы и их назначение

* [sber_contest_Yudin.ipynb](sber_contest_Yudin.ipynb) - основной рабочий ноутбук

* [data_preprocessing.py](data_preprocessing.py) - автономный модуль для препроцессинга данных

* [feature_engineering.py](feature_engineering.py) - автономный модуль для создания новых признаков

* [model_training.py](model_training.py) - автономный модуль для тренировки модели

* [model_evaluation.py](model_evaluation.py) - автономный модуль для оценки модели

* [model_inference.py](model_inference.py) - автономный модуль для вывода модели

* [utils.py](utils.py) - библиотека с необходимыми функциями

* [mylogger.py](mylogger.py) - модуль для логгера

