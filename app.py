import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from agents.data_agent import fetch_stock_data
from agents.feature_engineer_agent import FeatureEngineerAgent
from agents.forecast_agent import ForecastAgent, outlook
from agents.neuro_symbolic_agent import NeuroSymbolicAgent
from agents.prediction_agent import PredictionAgent

INTERVAL_STEP = {"1d": pd.offsets.BDay(1), "1h": pd.Timedelta(hours=1), "30m": pd.Timedelta(minutes=30)}


@st.cache_resource
def load_predictor():
    return PredictionAgent().load_models("models")


@st.cache_resource
def load_forecaster():
    return ForecastAgent.load()


@st.cache_data(ttl=3600)
def load_outlook(symbol):
    return outlook(symbol, agent=load_forecaster())


@st.cache_resource
def load_risk_model():
    from improvements.risk_model import RiskModel
    return RiskModel.load()


@st.cache_data(ttl=3600)
def load_risk(symbol):
    return load_risk_model().forecast(symbol)


st.title("📈 Stock Price Predictor")

symbol = st.text_input("Enter Stock Symbol", "AAPL")
period = st.selectbox("Select Time Period", ["3mo", "6mo", "1y", "2y"], index=1)
interval = st.selectbox("Select Interval", ["1d", "1h", "30m"], index=0)

if st.button("Predict"):
    try:
        predictor = load_predictor()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    trained_interval = predictor.meta.get("interval", "1d")
    if interval != trained_interval:
        st.warning(f"Models were trained on {trained_interval} bars; predictions on "
                   f"{interval} bars are out of distribution.")

    try:
        df = fetch_stock_data(symbol, period, interval)
        df = FeatureEngineerAgent().add_indicators(df)
        result = NeuroSymbolicAgent(predictor=predictor).predict_detailed(df)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("Engineered Features")
    st.write(df.dropna().tail())

    prediction = result["prediction"]
    last_date = df.index[-1]
    next_date = last_date + INTERVAL_STEP[interval]
    fmt = "%A, %B %d, %Y" if interval == "1d" else "%A, %B %d, %Y %H:%M"

    # 📊 Show output
    st.subheader("📊 Prediction Output")
    change = prediction / result["last_close"] - 1
    st.markdown(f"**Predicted Closing Price for {next_date.strftime(fmt)}:** "
                f"${prediction:.2f} ({change:+.2%} vs last close ${result['last_close']:.2f})")
    st.caption(result["explanation"])
    st.write({name: f"{r:+.3%}" for name, r in result["model_returns"].items()})

    # 📈 Plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["Close"], label="Historical Close Price")
    ax.scatter([next_date], [prediction], color="r", zorder=3,
               label=f"Prediction for {next_date.strftime('%b %d')}")
    ax.legend()
    st.pyplot(fig)

    # 🔭 Pooled multi-ticker model (see backtest.py)
    st.subheader("🔭 Multi-day outlook (pooled 100-stock model)")
    try:
        view = load_outlook(symbol)
    except FileNotFoundError as e:
        st.info(str(e))
    except ValueError as e:
        st.warning(f"Outlook unavailable: {e}")
    else:
        c1, c2 = st.columns(2)
        c1.metric(f"P(beats SPY over next {view['horizon']} days)", f"{view['p_beat_spy']:.1%}")
        c2.metric(f"Forecast {view['vol_horizon']}-day volatility (annualised)",
                  f"{view['forecast_vol']:.1%}",
                  f"{view['forecast_vol'] - view['trailing_vol']:+.1%} vs trailing 20d",
                  delta_color="inverse")
        st.caption(
            f"As of {view['as_of']}; model trained through {view['trained_through']}. "
            "In walk-forward tests the volatility forecast clearly beats the trailing "
            "estimate, while the direction signal is weak (AUC ~0.51) - treat it as a "
            "slight tilt, not a trade signal."
        )

    # 🛡️ 20-day risk model (improvements/risk_model.py)
    st.subheader("🛡️ 20-day risk outlook (S&P 500 risk model)")
    try:
        risk = load_risk(symbol)
    except FileNotFoundError as e:
        st.info(str(e))
    except ValueError as e:
        st.warning(f"Risk outlook unavailable: {e}")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast 20-day vol (annualised)", f"{risk['forecast_vol']:.1%}",
                  f"{risk['forecast_vol'] - risk['trailing_vol']:+.1%} vs trailing",
                  delta_color="inverse")
        c2.metric("80% price range (20 days)",
                  f"${risk['range_80'][0]:,.2f} - ${risk['range_80'][1]:,.2f}")
        c3.metric("95% price range (20 days)",
                  f"${risk['range_95'][0]:,.2f} - ${risk['range_95'][1]:,.2f}")
        earn = (f"Last earnings {risk['days_since_earnings']} sessions ago"
                if risk["days_since_earnings"] is not None else "Earnings timing unknown")
        if risk["next_earnings"]:
            earn += f"; next expected {risk['next_earnings']}"
        st.caption(
            f"As of {risk['as_of']}. Forecast is at the {risk['vol_percentile_3y']:.0f}th percentile of this "
            f"ticker's 3-year volatility history. {earn}. Model: XGBoost trained on the QLIKE loss, "
            "the best risk forecaster in a 22-year S&P 500 walk-forward test; ranges are calibrated "
            "to hold ~80% / ~95% of the time."
        )

st.divider()
st.subheader("🤖 AI risk brief")
st.caption("Claude calls the risk, price, news and outlook agents as tools and writes a short brief. "
           "Needs ANTHROPIC_API_KEY (or `ant auth login`); otherwise an offline template is used.")
brief_symbols = st.text_input("Tickers for the brief (comma-separated)", symbol)
if st.button("Generate risk brief"):
    from agents.risk_brief_agent import RiskBriefAgent
    tickers = [s.strip().upper() for s in brief_symbols.split(",") if s.strip()]
    with st.spinner("Writing brief..."):
        text, source = RiskBriefAgent(risk_model=load_risk_model()).brief(tickers)
    st.markdown(text)
    st.caption(f"Source: {source}")
