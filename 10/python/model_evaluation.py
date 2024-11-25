import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report
from rich import print
from rich.panel import Panel
from rich.table import Table

def evaluate_model(model, X_test, y_test):
    print("[bold blue]Оценка модели...[/bold blue]")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True)
    return accuracy, report

def display_results(accuracy, report):
    print(Panel(f"[bold green]Точность модели: {accuracy:.2%}[/bold green]"))
    
    table = Table(title="Отчет о классификации")
    table.add_column("Метрика", style="cyan")
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="magenta")
    table.add_column("F1-score", style="magenta")
    
    for label in ['0', '1']:
        table.add_row(
            f"Класс {label}",
            f"{report[label]['precision']:.2f}",
            f"{report[label]['recall']:.2f}",
            f"{report[label]['f1-score']:.2f}"
        )
    
    print(table)

if __name__ == "__main__":
    X_test = pd.read_csv('data/X_test_fe.csv')
    y_test = pd.read_csv('data/y_test.csv').values.ravel()

    model = joblib.load('models/churn_model.pkl')
    accuracy, report = evaluate_model(model, X_test, y_test)

    display_results(accuracy, report)
