import os
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pmdarima.arima import auto_arima
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from pylab import rcParams
import plotly.graph_objects as go


# Test for stationarity (do the ADF test)
def test_stationarity(timeseries):
    # Determining rolling statistics
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()
    
    adft = adfuller(timeseries, autolag='AIC')

    output = pd.Series(adft[0:4], index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)

    return adft[1]  # Return the p-value

def ARIMA_model(ticker, start='2020-01-01', end='2024-01-01', forecast_days=90):
    # Fetch Data
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)

    # Ensure 'Date' is in datetime format
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Test stationarity 
    df_close = df['Close']
    # Test for stationarity
    p_value = test_stationarity(df_close)

    # If not stationary, apply log transformation and differencing
    if p_value > 0.05:
        print("The series is not stationary. Applying transformations...")
        df_log = np.log(df_close)
        df_log_diff = df_log.diff().dropna()
        
        # Test for stationarity again
        p_value_diff = test_stationarity(df_log_diff)
        
        if p_value_diff <= 0.05:
            print("The series is now stationary after differencing.")
        else:
            print("The series is still not stationary after differencing. Further transformation may be required.")
        data_for_model = df_log
    else:
        print("The series is stationary.")
        data_for_model = df_close

    # Fit the model on the entire dataset
    auto_model = auto_arima(data_for_model, seasonal=True, m=12, trace=True)
    model = auto_model.fit(data_for_model)
    print(model.summary())

    # Forecast
    forecast_period = forecast_days  # Extend forecast period to the specified number of days into the future
    forecast, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)

    # Add this line after the forecast is generated
    forecast = add_noise(forecast, noise_level=0.02)
    
    # Convert data back to original scale
    if p_value > 0.05:
        forecast_exp = np.exp(forecast)
        conf_int_exp = np.exp(conf_int)
    else:
        forecast_exp = forecast
        conf_int_exp = conf_int

    return df_close, forecast_exp, conf_int_exp

def add_noise(forecast, noise_level=0.02):
    noise = np.random.normal(0, noise_level, forecast.shape)
    forecast_with_noise = forecast + noise
    return forecast_with_noise

if __name__ == '__main__':
    ticker = 'GE'
    start = '2020-01-01'
    end = '2024-07-20'
    forecast_days = 90
    historical_data, forecast, conf_int = ARIMA_model(ticker, start, end, forecast_days)

    # Create traces for the plot
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Historical Data'))
    future_dates = pd.date_range(historical_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int[:, 0],
        fill=None,
        mode='lines',
        line=dict(color='pink', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int[:, 1],
        fill='tonexty',  # fill area between trace0 and trace1
        mode='lines', line=dict(color='pink'),
        name='Confidence Interval'
    ))

    # Update layout
    fig.update_layout(
        title=ticker + ' Stock Price Prediction',
        xaxis_title='Time',
        yaxis_title=ticker + ' Stock Price',
        legend_title='Legend',
        font=dict(size=12)
    )

    # Show plot
    fig.show()
