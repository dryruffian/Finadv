import yfinance as yf
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

@st.cache_data
def fetch_stock_data(symbol, period="1y"):
    try:
        stock = yf.Ticker(symbol)
        hist = stock.history(period=period)
        return hist
    except Exception as e:
        st.error(f"Error fetching stock data: {e}")
        return None

def predict_stock_price(data, days=30):
    data = data.reset_index()
    data['Date'] = pd.to_datetime(data['Date'])
    data['Days'] = (data['Date'] - data['Date'].min()).dt.days

    X = data[['Days']]
    y = data['Close']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    future_dates = pd.date_range(start=data['Date'].max(), periods=days + 1)[1:]
    future_days = (future_dates - data['Date'].min()).days.values.reshape(-1, 1)

    predictions = model.predict(future_days)

    return future_dates, predictions