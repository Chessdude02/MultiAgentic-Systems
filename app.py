import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

from agents.data_agent import fetch_stock_data
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.prediction_agent import PredictionAgent
from agents.symbolic_override_agent import SymbolicOverrideAgent
from agents.neuro_symbolic_agent import NeuroSymbolicAgent

st.title("📈 Stock Price Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Time Period", ["1mo", "3mo", "6mo", "1y"], index=2)
interval = st.selectbox("Select Interval", ["1d", "1h", "30m"], index=0)

if st.button("Predict"):
    df = fetch_stock_data(symbol, period, interval)
    st.subheader("Raw Stock Data")
    st.write(df.tail())

    feature_engineer = FeatureEngineerAgent()
    predictor = PredictionAgent()
    symbolic = SymbolicOverrideAgent()
    fusion = NeuroSymbolicAgent()

    df = feature_engineer.add_indicators(df).dropna()
    st.subheader("Engineered Features")
    st.write(df.tail())

    # 🧠 Make prediction
    prediction = fusion.predict(df)

    # 📅 Calculate next date
    last_date = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["Date"].iloc[-1])
    next_date = last_date + timedelta(days=1)

    # 📊 Show output
    st.subheader("📊 Prediction Output")
    st.markdown(f"**Predicted Closing Price for {next_date.strftime('%A, %B %d, %Y')}:** ${prediction:.2f}")

    # 📈 Plot
    plt.figure(figsize=(10, 4))
    plt.plot(df["Close"], label="Historical Close Price")
    plt.axhline(prediction, color='r', linestyle='--', label=f"Prediction for {next_date.strftime('%b %d')}")
    plt.legend()
    st.pyplot(plt)
