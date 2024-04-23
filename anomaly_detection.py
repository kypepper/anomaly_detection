import dash
from dash import dcc, html
import plotly.graph_objects as go
from dash.dependencies import Input, Output
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM 
from decimal import Decimal, getcontext
from ta.volume import OnBalanceVolumeIndicator
from ta.trend import AroonIndicator
from ta.trend import ADXIndicator
from datetime import date
import datetime
from datetime import timedelta
from sklearn.impute import SimpleImputer

# Define functions for data processing
def fetch_stock_data(symbol, period):
    data = yf.download(symbol, period=period)
    return data

def calculate_adl(data):
    adl = [0]
    for i in range(1, len(data)):
        adl.append(adl[i-1] + ((data['Close'][i] - data['Low'][i]) - (data['High'][i] - data['Close'][i])) * data['Volume'][i] / (data['High'][i] - data['Low'][i]))
    data['ADL'] = adl
    return data

def calculate_technical_indicators(data):
    obv = OnBalanceVolumeIndicator(data['Close'], data['Volume'])
    data['OBV'] = obv.on_balance_volume()

    # Calculate ADL
    data = calculate_adl(data)

    # Calculate Aroon
    aroon = AroonIndicator(data['High'], data['Low'], window=14)  # Pass both high and low prices
    data['Aroon_Up'] = aroon.aroon_up()
    data['Aroon_Down'] = aroon.aroon_down()

    adx = ADXIndicator(data['High'], data['Low'], data['Close'])
    data['ADX'] = adx.adx()
    
    return data


def train_model(data):
    # Extract input features (X) and target variable (y)
    X = data[['OBV', 'Aroon_Up', 'Aroon_Down', 'ADX', 'ADL']].values
    y = data['Close'].values
    
    # Impute missing values in input features using mean strategy
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    # Initialize and train the linear regression model
    model = LinearRegression()
    model.fit(X_imputed, y)
    
    return model

def predict_prices(model, data, prediction_days=7):
    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=prediction_days+1, closed='right')
    future_dates = future_dates[1:]  # Exclude the last date
    
    # Create a DataFrame with the same columns as the training data
    future_data = pd.DataFrame(columns=['OBV', 'Aroon_Up', 'Aroon_Down', 'ADX', 'ADL'], index=future_dates)
    
    # Assuming the 'data' DataFrame contains the latest values for the features
    future_data['OBV'] = data['OBV'].iloc[-1]
    future_data['Aroon_Up'] = data['Aroon_Up'].iloc[-1]
    future_data['Aroon_Down'] = data['Aroon_Down'].iloc[-1]
    future_data['ADX'] = data['ADX'].iloc[-1]
    future_data['ADL'] = data['ADL'].iloc[-1]
    
    # Predict prices for the future dates
    predicted_prices = model.predict(future_data)
    
    return predicted_prices, future_dates



def add_sma(data, sma_period=20):
    data['SMA'] = data['Close'].rolling(window=sma_period).mean()
    return data

def fit_linear_regression(data):
    X = np.array(data.index.astype(np.int64) // 10**9).reshape(-1, 1)
    y = data['Close']
    model = LinearRegression()
    model.fit(X, y)
    predicted_prices = model.predict(X)
    return predicted_prices

def calculate_volatility(data):
    returns = data['Close'].pct_change()
    volatility = np.sqrt(returns.var() * 252)  # Annualized volatility
    return Decimal(volatility)

def calculate_risk(data, risk_factors):
    risk_data = {}
    if 'volatility' in risk_factors:
        volatility = calculate_volatility(data)
        risk_data['Volatility Risk'] = volatility
    
    return go.Figure(go.Indicator(
        mode='gauge+number',
        value=list(risk_data.values())[0],  # Use the first risk value for simplicity
        title=list(risk_data.keys())[0],  # Use the first risk key for simplicity
        gauge=dict(
            axis=dict(range=[0, 1] if 'volatility' in risk_factors else [0, 100]),
            bar=dict(color='darkblue'),
            steps=[
                dict(range=[0, 0.25] if 'volatility' in risk_factors else [0, 25], color='green'),
                dict(range=[0.25, 0.5] if 'volatility' in risk_factors else [25, 50], color='yellow'),
                dict(range=[0.5, 0.75] if 'volatility' in risk_factors else [50, 75], color='orange'),
                dict(range=[0.75, 1] if 'volatility' in risk_factors else [75, 100], color='red')
            ]
        )
    ))

# Function for DBSCAN anomaly detection
def detect_anomalies_dbscan(data, eps=0.5, min_samples=5):
    # Scale the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close', 'Volume']])  # Include relevant features for DBSCAN
    # Apply DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    data['Anomaly_DBSCAN'] = dbscan.fit_predict(scaled_data)
    return data


def detect_anomalies_svm(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    svm_detector = OneClassSVM(nu=0.05)  # nu is the fraction of outliers
    data['Anomaly_SVM'] = svm_detector.fit_predict(scaled_data)
    return data

def detect_anomalies(data):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['Close']])
    outlier_detector = IsolationForest(contamination=0.05)
    data['Anomaly'] = outlier_detector.fit_predict(scaled_data)
    return data

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

def combine_anomalies(data):
    data['Combined_Anomaly'] = np.logical_or(data['Anomaly_IForest'], data['Anomaly_LOF'])
    return data

def generate_explanations(data):
    explanations = []
    for i in range(len(data)):
        explanation = ""
        if data['Combined_Anomaly'][i]:
            # Determine the anomaly type
            if data['Anomaly_IForest'][i] == -1:
                anomaly_type = "Isolation Forest"
            elif data['Anomaly_LOF'][i] == -1:
                anomaly_type = "Local Outlier Factor"
            elif data['Anomaly_DBSCAN'][i] == -1:
                anomaly_type = "DBSCAN"
            elif data['Anomaly_SVM'][i] == -1:
                anomaly_type = "One-Class SVM"
            else:
                anomaly_type = "Unknown"

            # Provide enhanced explanation based on anomaly type and market conditions
            if anomaly_type == "Isolation Forest":
                explanation += "Anomaly detected by Isolation Forest. "
            elif anomaly_type == "Local Outlier Factor":
                explanation += "Anomaly detected by Local Outlier Factor. "
            elif anomaly_type == "DBSCAN":
                explanation += "Anomaly detected by DBSCAN. "
            elif anomaly_type == "One-Class SVM":
                explanation += "Anomaly detected by One-Class SVM. "

            # Additional context based on market conditions and historical trends
            if data['Close'][i] > data['SMA'][i]:
                explanation += "The stock price is currently above the 20-day Simple Moving Average (SMA), indicating a potential bullish trend. "
            else:
                explanation += "The stock price is currently below the 20-day Simple Moving Average (SMA), suggesting a possible bearish trend. "
            
            if data['Volume'][i] > data['Volume'].mean():
                explanation += "The trading volume is higher than the average, possibly indicating increased market activity. "
            else:
                explanation += "The trading volume is lower than the average, suggesting reduced market activity. "
                
            # Add any other relevant factors contributing to the anomaly
            
        else:
            explanation = None
        
        explanations.append(explanation)
    
    return explanations


# Define functions for visualization and UI

def sma_trend_lines_with_volume(data, predicted_prices, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                         open=data['Open'],
                         high=data['High'],
                         low=data['Low'],
                         close=data['Close'],
                         name='Candlestick'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA'], name='SMA'))
    fig.add_trace(go.Scatter(x=data.index, y=predicted_prices, mode='lines', name='Trend'))
    fig.add_trace(go.Bar(x=data.index, 
                         y=data['Volume'], 
                         name='Volume', 
                         yaxis='y2',
                         marker=dict(color='rgba(31,119,180,0.5)', line=dict(color='rgba(31,119,180,0.5)', width=1)),
                         width=0.5*20*30*30*100  # Set the width of the bars (adjust as needed)
                        ))
    
    fig.update_layout(
        xaxis_title=x_axis,
        yaxis=dict(title=y_axis, type=y_scale),
        yaxis2=dict(title='Volume', overlaying='y', side='right'),
        height=800
    )
    return fig

def anomalies(data, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                     open=data['Open'],
                     high=data['High'],
                     low=data['Low'],
                     close=data['Close'],
                     name='Candlestick'))

    # Highlight anomalies detected by Isolation Forest
    anomaly_indices_iforest = data.index[data['Anomaly_IForest'] == -1]
    for index in anomaly_indices_iforest:
        anomaly_start = max(0, data.index.get_loc(index) - 1)
        anomaly_end = min(len(data) - 1, data.index.get_loc(index) + 1)

        fig.add_shape(type="rect",
                      xref="x", yref="paper",
                      x0=data.index[anomaly_start], y0=0,
                      x1=data.index[anomaly_end], y1=1,
                      fillcolor="purple",
                      opacity=0.2,
                      layer="below",
                      line_width=0)
        
    # Highlight anomalies detected by Local Outlier Factor
    anomaly_indices_lof = data.index[data['Anomaly_LOF'] == -1]
    for index in anomaly_indices_lof:
        anomaly_start = max(0, data.index.get_loc(index) - 1)
        anomaly_end = min(len(data) - 1, data.index.get_loc(index) + 1)

        fig.add_shape(type="rect",
                      xref="x", yref="paper",
                      x0=data.index[anomaly_start], y0=0,
                      x1=data.index[anomaly_end], y1=1,
                      fillcolor="blue",
                      opacity=0.2,
                      layer="below",
                      line_width=0)

    # Highlight anomalies detected by DBSCAN
    anomaly_indices_dbscan = data.index[data['Anomaly_DBSCAN'] == -1]
    for index in anomaly_indices_dbscan:
        anomaly_start = max(0, data.index.get_loc(index) - 1)
        anomaly_end = min(len(data) - 1, data.index.get_loc(index) + 1)

        fig.add_shape(type="rect",
                      xref="x", yref="paper",
                      x0=data.index[anomaly_start], y0=0,
                      x1=data.index[anomaly_end], y1=1,
                      fillcolor="red",
                      opacity=0.2,
                      layer="below",
                      line_width=0)

    # Highlight anomalies detected by One-Class SVM
    anomaly_indices_svm = data.index[data['Anomaly_SVM'] == -1]
    for index in anomaly_indices_svm:
        anomaly_start = max(0, data.index.get_loc(index) - 1)
        anomaly_end = min(len(data) - 1, data.index.get_loc(index) + 1)

        fig.add_shape(type="rect",
                      xref="x", yref="paper",
                      x0=data.index[anomaly_start], y0=0,
                      x1=data.index[anomaly_end], y1=1,
                      fillcolor="green",
                      opacity=0.2,
                      layer="below",
                      line_width=0)

    fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, yaxis_type=y_scale)
    return fig


def all_data(data, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='markers', name='Close Price'))
    
    anomaly_indices_dbscan = data.index[data['Anomaly_DBSCAN'] == -1]
    anomaly_indices_lof = data.index[data['Anomaly_LOF'] == -1]
    anomaly_indices_svm = data.index[data['Anomaly_SVM'] == -1]  # Include anomalies detected by sliding window
    anomaly_indices_iforest = data.index[data['Anomaly_IForest'] == -1]  # Include anomalies detected by Isolation Forest
    
        # Add anomalies detected by DBSCAN
    fig.add_trace(go.Scatter(x=anomaly_indices_dbscan, y=data.loc[anomaly_indices_dbscan]['Close'], 
                             mode='markers', name='Anomaly DBSCAN', 
                             marker=dict(color='red', size=7)))
    
    # Add anomalies detected by LOF
    fig.add_trace(go.Scatter(x=anomaly_indices_lof, y=data.loc[anomaly_indices_lof]['Close'], 
                             mode='markers', name='Anomaly LOF', 
                             marker=dict(color='blue', size=7)))
    
    # Add anomalies detected by Sliding Window
    fig.add_trace(go.Scatter(x=anomaly_indices_svm, y=data.loc[anomaly_indices_svm]['Close'], 
                             mode='markers', name='Anomaly_SVM', 
                             marker=dict(color='green', size=7)))
    
    # Add anomalies detected by Isolation Forest
    fig.add_trace(go.Scatter(x=anomaly_indices_iforest, y=data.loc[anomaly_indices_iforest]['Close'], 
                             mode='markers', name='Anomaly IForest', 
                             marker=dict(color='purple', size=7)))
    
    fig.update_layout(xaxis_title=x_axis, yaxis_title=y_axis, yaxis_type=y_scale)
    return fig



def causes(data, x_axis, y_axis, y_scale):
    fig = go.Figure()
    fig.add_trace(go.Table(
        header=dict(values=['Date', 'Explanation'],
                    fill_color='paleturquoise',
                    align='left'),
        cells=dict(values=[data.index[data['Anomaly'] == -1], data['Explanation'][data['Anomaly'] == -1]],
                   fill_color='lavender',
                   align='left')))
    return fig

# Create the Dash app
app = dash.Dash(__name__)

server = app.server
# Define the layout of the app
app.layout = html.Div([
    html.Div([
        html.Label('Stock Symbol'),
        dcc.Input(id='stock-symbol-input', type='text', value='AAPL'),
        html.Label('Time Period'),
        dcc.Dropdown(
            id='time-period-dropdown',
            options=[
                {'label': '1 Week', 'value': '1wk'},
                {'label': '1 Month', 'value': '1mo'},
                {'label': '3 Months', 'value': '3mo'},
                {'label': '6 Months', 'value': '6mo'},
                {'label': '1 Year', 'value': '1y'},
                {'label': '5 Years', 'value': '5y'}
            ],
            value='1y'
        ),
        html.Label('Select Risk Factors'),
        dcc.Checklist(
            id='risk-factors-checklist',
            options=[
                {'label': 'Volatility', 'value': 'volatility'},
            ],
            value=['volatility', ]
        ),
    ]),
    dcc.Tabs([
        dcc.Tab(label='SMA and Trend Lines', children=[
            dcc.Graph(id='sma-trend-lines', style={'height': '800px'})
        ]),
        dcc.Tab(label='Anomalies', children=[
            dcc.Graph(id='anomalies', style={'height': '800px'})
        ]),
        dcc.Tab(label='All Data', children=[
            dcc.Graph(id='all-data', style={'height': '800px'})
        ]),
        dcc.Tab(label='Causes', children=[
            dcc.Graph(id='causes', style={'height': '800px'})
        ]),
        dcc.Tab(label='Risk', children=[
            dcc.Graph(id='risk', style={'height': '400px'})
        ]),
        dcc.Tab(label='Prediction', children=[
            dcc.Graph(id='prediction-chart', style={'height': '800px'})
        ]),
    ]),
    html.Div([
        html.Label('X-Axis'),
        dcc.Dropdown(
            id='x-axis-dropdown',
            options=[
                {'label': 'Date', 'value': 'Date'},
                {'label': 'Custom', 'value': 'Custom'}
            ],
            value='Date'
        ),
        html.Label('Y-Axis'),
        dcc.Dropdown(
            id='y-axis-dropdown',
            options=[
                {'label': 'Close Price', 'value': 'Close'},
                {'label': 'Volume', 'value': 'Volume'}
            ],
            value='Close'
        ),
        html.Label('Y-Axis Scale'),
        dcc.Dropdown(
            id='y-scale-dropdown',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Logarithmic', 'value': 'log'}
            ],
            value='linear'
        )
    ]),
])

# Callback to update the graphs based on axis selections
@app.callback(
    [Output('sma-trend-lines', 'figure'),
     Output('anomalies', 'figure'),
     Output('all-data', 'figure'),
     Output('causes', 'figure'),
     Output('risk', 'figure'),
     Output('prediction-chart', 'figure')],
    [Input('stock-symbol-input', 'value'),
     Input('time-period-dropdown', 'value'),
     Input('x-axis-dropdown', 'value'),
     Input('y-axis-dropdown', 'value'),
     Input('y-scale-dropdown', 'value'),
     Input('risk-factors-checklist', 'value')]
)
def update_graphs(symbol, period, x_axis, y_axis, y_scale, risk_factors):
    # Get stock data for the specified symbol and time period
    data = fetch_stock_data(symbol, period)
    data = add_sma(data)
    predicted_prices = fit_linear_regression(data)
    data = detect_anomalies(data)
    data = detect_anomalies_iforest(data)
    data = detect_anomalies_lof(data)
    data = detect_anomalies_dbscan(data)  # Add DBSCAN anomalies
    data = detect_anomalies_svm(data)
    data = combine_anomalies(data)
    data['Explanation'] = generate_explanations(data)
    data = calculate_technical_indicators(data)
    model = train_model(data)
    predicted_prices, future_dates = predict_prices(model, data)

    # Calculate start and end dates for the desired data range
    end_date = future_dates[-1]
    start_date = end_date - timedelta(days=20)  # Three weeks prior

# Filter data for the prior three weeks and the prediction week
    prior_data = data[(data.index >= start_date) & (data.index <= end_date)]
    predicted_data = pd.DataFrame({'Close': predicted_prices}, index=future_dates)

    prediction_chart = go.Figure([
        go.Scatter(x=prior_data.index, y=prior_data['Close'], mode='lines', name='Prior Close Price'),
        go.Scatter(x=predicted_data.index, y=predicted_data['Close'], mode='lines', name='Predicted Close Price', line=dict(color='red'))
    ])
    
    prediction_chart.update_layout(title='Actual vs Predicted Close Prices')
    
    risk_figure = calculate_risk(data, risk_factors)


    figures = [
        sma_trend_lines_with_volume(data, predicted_prices, x_axis, y_axis, y_scale),
        anomalies(data, x_axis, y_axis, y_scale),
        all_data(data, x_axis, y_axis, y_scale),
        causes(data, x_axis, y_axis, y_scale),
        risk_figure,
        prediction_chart
    ]
    
    return figures

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
