import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import joblib
from rich import print
from rich.panel import Panel
from rich.progress import track

def train_model(X_train, y_train):
    print("[bold blue]Начало обучения модели...[/bold blue]")
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    
    with track(range(1), description="Обучение модели...") as progress:
        grid_search.fit(X_train, y_train)
        progress.advance()
    
    print("[bold green]Модель успешно обучена![/bold green]")
    return grid_search.best_estimator_

if __name__ == "__main__":
    print("[bold yellow]Загрузка данных...[/bold yellow]")
    X_train = pd.read_csv('data/X_train_fe.csv')
    y_train = pd.read_csv('data/y_train.csv').values.ravel()

    model = train_model(X_train, y_train)
    
    print("[bold yellow]Сохранение модели...[/bold yellow]")
    joblib.dump(model, 'models/churn_model.pkl')
    print(Panel.fit("[bold cyan]Модель сохранена в models/churn_model.pkl[/bold cyan]"))
    
    print(Panel.fit(f"[bold green]Лучшие параметры модели:[/bold green]\n{model.get_params()}"))
