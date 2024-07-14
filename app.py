import yfinance as yf
import pandas as pd

def fetch_data(ticker):
    stock_data = yf.download(ticker, start="2020-01-01", end="2023-01-01")
    return stock_data

def preprocess_data(data):
    data = data[['Open', 'High', 'Low', 'Close']]
    data['Future_Close'] = data['Close'].shift(-1)
    data.dropna(inplace=True)
    return data

stock_data = fetch_data('AAPL')
processed_data = preprocess_data(stock_data)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(data):
    X = data[['Open', 'High', 'Low', 'Close']]
    y = data['Future_Close']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

model, X_test, y_test = train_model(processed_data)

import plotly.graph_objs as go

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    
    scatter_trace = go.Scatter(x=y_test, y=predictions, mode='markers', name='Predictions vs Actual')
    line_trace = go.Scatter(x=y_test, y=y_test, mode='lines', name='Ideal', line=dict(color='red'))
    
    layout = go.Layout(title='Predictions vs Actual', xaxis=dict(title='Actual'), yaxis=dict(title='Predicted'))
    fig = go.Figure(data=[scatter_trace, line_trace], layout=layout)
    
    return fig

fig = evaluate_model(model, X_test, y_test)

import streamlit as st

def main():
    st.title('Stock Prediction App')
    
    ticker = st.text_input('Enter Stock Ticker', 'AAPL')
    if ticker:
        data = fetch_data(ticker)
        processed_data = preprocess_data(data)
        model, X_test, y_test = train_model(processed_data)
        fig = evaluate_model(model, X_test, y_test)
        
        st.plotly_chart(fig)
        st.write('Future Closing Price Predictions:')
        predictions = model.predict(X_test)
        results = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
        st.write(results)

if __name__ == '__main__':
    main()
