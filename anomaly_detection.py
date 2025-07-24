import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from decimal import Decimal
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import AroonIndicator, ADXIndicator
from datetime import datetime, timedelta
from sklearn.impute import SimpleImputer

API_KEY = 'Rz5iUqxB2rN57pUKjvZYGZiXY8erLy0N'

# Fetch data using Financial Modeling Prep API
def fetch_stock_data(symbol, period):
    today = pd.Timestamp.today().normalize()
    if period == '3mo':
        start_date = today - pd.Timedelta(days=90)
    elif period == '6mo':
        start_date = today - pd.Timedelta(days=182)
    elif period == '1y':
        start_date = today - pd.Timedelta(days=365)
    elif period == '5y':
        start_date = today - pd.Timedelta(days=365*5)
    else:
        start_date = today - pd.Timedelta(days=365)  # default 1 year
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = today.strftime('%Y-%m-%d')

    url = f'https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_str}&to={end_str}&apikey={API_KEY}'
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to fetch data: {response.status_code} {response.text}")
        return None

    data_json = response.json()
    if 'historical' not in data_json:
        st.error(f"No historical data found for symbol {symbol}")
        return None

    hist_data = data_json['historical']
    df = pd.DataFrame(hist_data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    df.sort_index(inplace=True)

    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
    }, inplace=True)

    if 'Adj Close' not in df.columns:
        df['Adj Close'] = df['Close']

    return df

# Your existing functions adapted for use here (unchanged)

def calculate_adl(data):
    adl = [0]
    for i in range(1, len(data)):
        adl.append(adl[i-1] + ((data['Close'].iloc[i] - data['Low'].iloc[i]) - (data['High'].iloc[i] - data['Close'].iloc[i])) * data['Volume'].iloc[i] / (data['High'].iloc[i] - data['Low'].iloc[i]))
    data['ADL'] = adl
    return data

def calculate_technical_indicators(data):
    obv = OnBalanceVolumeIndicator(data['Close'], data['Volume'])
    data['OBV'] = obv.on_balance_volume()
    data = calculate_adl(data)
    aroon = AroonIndicator(data['High'], data['Low'], window=14)
    data['Aroon_Up'] = aroon.aroon_up()
    data['Aroon_Down'] = aroon.aroon_down()
    adx = ADXIndicator(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx.adx()
    return data

def train_model(data):
    X = data[['OBV', 'Aroon_Up', 'Aroon_Down', 'ADX', 'ADL']].values
    y = data['Close'].values
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    model = LinearRegression()
    model.fit(X_imputed, y)
    return model

def predict_prices(model, data, prediction_days=7):
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=prediction_days)
    future_data = pd.DataFrame({
        'OBV': data['OBV'].iloc[-1],
        'Aroon_Up': data['Aroon_Up'].iloc[-1],
        'Aroon_Down': data['Aroon_Down'].iloc[-1],
        'ADX': data['ADX'].iloc[-1],
        'ADL': data['ADL'].iloc[-1],
    }, index=future_dates)
    predicted_prices = model.predict(future_data)
    return predicted_prices, future_dates

def add_sma(data, sma_period=20):
    data['SMA'] = data['Close'].rolling(window=sma_period).mean()
    return data

def calculate_volatility(data):
    returns = data['Close'].pct_change()
    volatility = np.sqrt(returns.var() * 252)  # Annualized volatility
    return Decimal(volatility)

def calculate_risk(data, risk_factors):
    risk_data = {}
    if 'volatility' in risk_factors:
        volatility = calculate_volatility(data)
        risk_data['Volatility Risk'] = volatility
    
    if not risk_data:
        return None
    
    val = list(risk_data.values())[0]
    key = list(risk_data.keys())[0]
    
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=float(val),
        title={'text': key},
        gauge={
            'axis': {'range': [0, 1] if 'volatility' in risk_factors else [0, 100]},
            'bar': {'color': 'darkblue'},
            'steps': [
                {'range': [0, 0.25], 'color': 'green'},
                {'range': [0.25, 0.5], 'color': 'yellow'},
                {'range': [0.5, 0.75], 'color': 'orange'},
                {'range': [0.75, 1], 'color': 'red'},
            ]
        }
    ))
    return fig

def detect_anomalies_iforest(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    outlier_detector = IsolationForest(contamination=0.05)
    data['Anomaly_IForest'] = outlier_detector.fit_predict(scaled_data)
    return data

def detect_anomalies_lof(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    outlier_detector = LocalOutlierFactor(n_neighbors=30, contamination=0.03)
    data['Anomaly_LOF'] = outlier_detector.fit_predict(scaled_data)
    return data

def detect_anomalies_dbscan(data, eps=0.5, min_samples=5):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume']])
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Anomaly_DBSCAN'] = dbscan.fit_predict(scaled_data)
    return data

def detect_anomalies_svm(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    svm_detector = OneClassSVM(nu=0.05)
    data['Anomaly_SVM'] = svm_detector.fit_predict(scaled_data)
    return data

def combine_anomalies(data):
    # Logical OR of anomalies detected by Isolation Forest and LOF
    data['Combined_Anomaly'] = np.logical_or(data['Anomaly_IForest'] == -1, data['Anomaly_LOF'] == -1)
    return data

def generate_explanations(data):
    explanations = []
    for i in range(len(data)):
        explanation = None
        if data['Combined_Anomaly'].iloc[i]:
            if data['Anomaly_IForest'].iloc[i] == -1:
                anomaly_type = "Isolation Forest"
            elif data['Anomaly_LOF'].iloc[i] == -1:
                anomaly_type = "Local Outlier Factor"
            elif data['Anomaly_DBSCAN'].iloc[i] == -1:
                anomaly_type = "DBSCAN"
            elif data['Anomaly_SVM'].iloc[i] == -1:
                anomaly_type = "One-Class SVM"
            else:
                anomaly_type = "Unknown"

            explanation = f"Anomaly detected by {anomaly_type}. "
            if data['Close'].iloc[i] > data['SMA'].iloc[i]:
                explanation += "Price above 20-day SMA, indicating bullish trend. "
            else:
                explanation += "Price below 20-day SMA, indicating bearish trend. "
            if data['Volume'].iloc[i] > data['Volume'].mean():
                explanation += "Volume higher than average, possible increased market activity."
            else:
                explanation += "Volume lower than average, possible decreased market activity."
        explanations.append(explanation)
    return explanations

# Plot functions for Streamlit display using plotly

def plot_sma_trend(data, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name='SMA'))
    fig.add_trace(go.Bar(x=data.index, y=data['Volume'], name='Volume',
                         yaxis='y2',
                         marker=dict(color='rgba(31,119,180,0.5)')))
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis=dict(title=y_axis, type=y_scale),
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        height=800)
    return fig

def plot_all_data(data, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='markers', name='Close Price'))
    anomaly_colors = {'Anomaly_DBSCAN': 'red', 'Anomaly_LOF': 'blue', 'Anomaly_SVM': 'green', 'Anomaly_IForest': 'purple'}
    for col, color in anomaly_colors.items():
        if col in data.columns:
            indices = data.index[data[col] == -1]
            fig.add_trace(go.Scatter(x=indices, y=data.loc[indices]['Close'],
                                     mode='markers', name=f'Anomaly {col.split("_")[1]}',
                                     marker=dict(color=color, size=7)))
    fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, yaxis_type=y_scale)
    return fig

def plot_anomalies(data, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'], high=data['High'],
                                 low=data['Low'], close=data['Close'],
                                 name='Candlestick'))

    anomaly_colors = {'Anomaly_IForest': 'purple', 'Anomaly_LOF': 'blue', 'Anomaly_DBSCAN': 'red', 'Anomaly_SVM': 'green'}
    for col, color in anomaly_colors.items():
        indices = data.index[data[col] == -1]
        for idx in indices:
            i = data.index.get_loc(idx)
            start = max(0, i-1)
            end = min(len(data)-1, i+1)
            fig.add_shape(type="rect", xref="x", yref="paper",
                          x0=data.index[start], y0=0,
                          x1=data.index[end], y1=1,
                          fillcolor=color, opacity=0.2,
                          layer="below", line_width=0)
    fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, yaxis_type=y_scale)
    return fig

def plot_causes(data):
    anomaly_rows = data[data['Combined_Anomaly']]
    causes_df = pd.DataFrame({
        'Date': anomaly_rows.index.strftime('%Y-%m-%d'),
        'Explanation': anomaly_rows['Explanation']
    })
    return causes_df

def plot_prediction(data, predicted_prices, future_dates):
    prior_data = data[data.index >= (future_dates[0] - timedelta(days=20))]
    predicted_data = pd.DataFrame({'Close': predicted_prices}, index=future_dates)
    fig = go.Figure([
        go.Scatter(x=prior_data.index, y=prior_data['Close'], mode='lines', name='Prior Close Price'),
        go.Scatter(x=predicted_data.index, y=predicted_data['Close'], mode='lines', name='Predicted Close Price', line=dict(color='red'))
    ])
    fig.update_layout(title='Actual vs Predicted Close Prices')
    return fig

# --- Streamlit UI ---

st.title("Stock Analysis, Anomaly Detection, and Price Prediction")

# Sidebar inputs
symbol = st.sidebar.text_input("Stock Symbol", value="AAPL").upper()
period = st.sidebar.selectbox("Time Period", ['3mo', '6mo', '1y', '5y'], index=3)
risk_factors = st.sidebar.multiselect("Select Risk Factors", options=['volatility'], default=['volatility'])

x_axis = st.sidebar.selectbox("X-Axis", ['Date', 'Custom'], index=0)
y_axis = st.sidebar.selectbox("Y-Axis", ['Close', 'Volume'], index=0)
y_scale = st.sidebar.selectbox("Y-Axis Scale", ['linear', 'log'], index=0)

# Load data
data = fetch_stock_data(symbol, period)
if data is None:
    st.stop()

data = add_sma(data)
data = calculate_technical_indicators(data)

# Detect anomalies
data = detect_anomalies_iforest(data)
data = detect_anomalies_lof(data)
data = detect_anomalies_dbscan(data)
data = detect_anomalies_svm(data)
data = combine_anomalies(data)
data['Explanation'] = generate_explanations(data)

# Train and predict
model = train_model(data)
predicted_prices, future_dates = predict_prices(model, data)

# Show Tabs in Streamlit
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "SMA and Trend Lines",
    "All Data",
    "Anomalies",
    "Causes",
    "Risk",
    "Prediction"
])

with tab1:
    st.plotly_chart(plot_sma_trend(data, x_axis, y_axis, y_scale), use_container_width=True)

with tab2:
    st.plotly_chart(plot_all_data(data, x_axis, y_axis, y_scale), use_container_width=True)

with tab3:
    st.plotly_chart(plot_anomalies(data, x_axis, y_axis, y_scale), use_container_width=True)

with tab4:
    causes_df = plot_causes(data)
    if not causes_df.empty:
        st.dataframe(causes_df)
    else:
        st.write("No anomalies detected.")

with tab5:
    risk_fig = calculate_risk(data, risk_factors)
    if risk_fig:
        st.plotly_chart(risk_fig, use_container_width=True)
    else:
        st.write("No risk factors selected or available.")

with tab6:
    st.plotly_chart(plot_prediction(data, predicted_prices, future_dates), use_container_width=True)
