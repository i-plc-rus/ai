import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def load_predictions(file_path):
    return pd.read_csv(file_path)

def plot_churn_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['ChurnProbability'], bins=20, alpha=0.7, color='blue')
    plt.title('Distribution of Churn Probability')
    plt.xlabel('Churn Probability')
    plt.ylabel('Number of Customers')
    st.pyplot(plt)

def display_top_customers(df, n=10):
    top_customers = df.sort_values(by='ChurnProbability', ascending=False).head(n)
    st.dataframe(top_customers)

def main():
    st.title("Customer Churn Prediction Dashboard")
    
    # Загружаем данные предсказаний
    data_path = 'data/new_customer_predictions.csv'
    df = load_predictions(data_path)

    st.sidebar.header("Options")
    n_customers = st.sidebar.slider("Number of top customers to display", min_value=5, max_value=20, value=10)
    
    st.header("Churn Probability Distribution")
    plot_churn_distribution(df)

    st.header(f"Top {n_customers} Customers with Highest Churn Probability")
    display_top_customers(df, n=n_customers)

if __name__ == "__main__":
    main()
