import warnings
import yfinance as yf
warnings.filterwarnings('ignore')  # Ignore warnings to keep the output clean
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima import auto_arima
import plotly.graph_objects as go


def test_stationarity(timeseries):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check the stationarity of a time series.

    Parameters:
        timeseries (pd.Series): The time series data to test for stationarity.

    Returns:
        float: The p-value from the ADF test.
    """
    # Calculate rolling mean and standard deviation for the time series
    rolmean = timeseries.rolling(12).mean()
    rolstd = timeseries.rolling(12).std()

    # Perform the Augmented Dickey-Fuller test
    adft = adfuller(timeseries, autolag='AIC')

    # Print the test statistics and critical values
    output = pd.Series(adft[0:4], index=['Test Statistics','p-value','No. of lags used','Number of observations used'])
    for key, values in adft[4].items():
        output['critical value (%s)' % key] = values
    print(output)

    return adft[1]  # Return the p-value of the ADF test

def ARIMA_model(ticker, start='2020-01-01', end='2024-01-01', forecast_days=90):
    """
    Fit an ARIMA model to the stock price data and generate a forecast.

    Parameters:
        ticker (str): The stock ticker symbol.
        start (str): The start date for fetching the data.
        end (str): The end date for fetching the data.
        forecast_days (int): The number of days to forecast into the future.

    Returns:
        tuple: A tuple containing the historical closing prices, forecasted prices, and confidence intervals.
    """
    # Fetch historical stock price data using yfinance
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)

    # Convert 'Date' column to datetime format and set it as the DataFrame index
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Extract the 'Close' prices for stationarity testing and modeling
    df_close = df['Close']

    # Test if the series is stationary
    p_value = test_stationarity(df_close)

    # If the series is not stationary, apply log transformation and differencing
    if p_value > 0.05:
        print("The series is not stationary. Applying transformations...")
        df_log = np.log(df_close)  # Apply log transformation to stabilize variance
        df_log_diff = df_log.diff().dropna()  # Apply differencing to remove trend and make the series stationary
        
        # Test the transformed series for stationarity again
        p_value_diff = test_stationarity(df_log_diff)
        
        if p_value_diff <= 0.05:
            print("The series is now stationary after differencing.")
        else:
            print("The series is still not stationary after differencing. Further transformation may be required.")
        data_for_model = df_log  # Use the log-transformed data for modeling
    else:
        print("The series is stationary.")
        data_for_model = df_close  # Use the original data for modeling if stationary

    # Fit an ARIMA model to the entire dataset
    auto_model = auto_arima(data_for_model, seasonal=True, m=12, trace=True)
    model = auto_model.fit(data_for_model)
    print(model.summary())  # Print model summary to show details about the fitted model

    # Generate forecast for the specified number of days
    forecast_period = forecast_days  # Extend forecast period to the specified number of days into the future
    forecast, conf_int = model.predict(n_periods=forecast_period, return_conf_int=True)

    # Add noise (line) to the forecast to simulate variability (after the forecast is generated)
    forecast = add_noise(forecast, noise_level=0.02)
    
    # Convert forecast and confidence intervals back to the original scale if transformations were applied
    if p_value > 0.05:
        forecast_exp = np.exp(forecast)
        conf_int_exp = np.exp(conf_int)
    else:
        forecast_exp = forecast
        conf_int_exp = conf_int

    return df_close, forecast_exp, conf_int_exp

def add_noise(forecast, noise_level=0.02):
    """
    Add random noise to the forecasted data.

    Parameters:
        forecast (np.array): The forecasted data.
        noise_level (float): The standard deviation of the Gaussian noise to add.

    Returns:
        np.array: The forecasted data with added noise.
    """
    # Generate random noise with specified standard deviation
    noise = np.random.normal(0, noise_level, forecast.shape)
    # Add noise to the forecast
    forecast_with_noise = forecast + noise
    return forecast_with_noise

# Testing the functionality of this file
if __name__ == '__main__':
    # Define the ticker symbol, start and end dates, and forecast period
    ticker = 'GE'
    start = '2020-01-01'
    end = '2024-07-20'
    forecast_days = 90

    # Run the ARIMA model and get the historical data, forecast, and confidence intervals
    historical_data, forecast, conf_int = ARIMA_model(ticker, start, end, forecast_days)

    # Create a plotly figure to visualize the data and forecast
    fig = go.Figure()

    # Plot stock prices
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Historical Data'))

    # Generate dates for the forecasted period
    future_dates = pd.date_range(historical_data.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='B')

    # Plot forecasted stock prices
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))

    # Plot the lower bound of the confidence interval
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int[:, 0],
        fill=None,
        mode='lines',
        line=dict(color='pink', width=0),
        showlegend=False
    ))

    # Plot the upper bound of the confidence interval and fill the area between the confidence bounds
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int[:, 1],
        fill='tonexty',  # Fill area between the lower and upper confidence bounds (between trace0 and trace1)
        mode='lines', line=dict(color='pink'),
        name='Confidence Interval'
    ))

    # Update plot layout with titles and labels
    fig.update_layout(
        title=ticker + ' Stock Price Prediction',
        xaxis_title='Time',
        yaxis_title=ticker + ' Stock Price',
        legend_title='Legend',
        font=dict(size=12)
    )

    # Show the plot
    fig.show()
