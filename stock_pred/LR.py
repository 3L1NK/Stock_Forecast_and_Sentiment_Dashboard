from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import timedelta
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller

def test_stationarity(timeseries):
    """
    Perform the Augmented Dickey-Fuller (ADF) test to check the stationarity of a time series.

    Parameters:
        timeseries (pd.Series): The time series data to test for stationarity.

    """
    # Calculate rolling mean and standard deviation
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()

    # Perform the Augmented Dickey-Fuller test
    dftest = adfuller(timeseries, autolag='AIC')

    # Create a Series to store test results
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])

    # Add critical values to the Series
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value

    # Print the test results
    print('Results of Dickey-Fuller Test:')
    print(dfoutput)

def create_features(data, window=40):
    """
    Create features and labels for time series prediction using a sliding window approach.

    Parameters:
        data (np.array): The time series data as a numpy array.
        window (int): The size of the window to use for creating features.

    Returns:
        tuple: A tuple containing:
            - X (np.array): The input features array.
            - y (np.array): The target values array.
    """
    X, y = [], []

    # Slide the window across the data to create features and labels
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])

    # Return the features and labels as numpy arrays
    return np.array(X), np.array(y)

def LR_model(ticker, start='2020-01-01', end='2024-01-01', forecast_period=90, window=40):
    """
    Train a Linear Regression model on historical stock prices and make future predictions.

    Parameters:
        ticker (str): The stock ticker symbol.
        start (str): The start date for fetching the data.
        end (str): The end date for fetching the data.
        forecast_period (int): The number of days to forecast into the future.
        window (int): The size of the window used to create features for the model.

    Returns:
        tuple: A tuple containing four DataFrames:
            - actual_prices_df (pd.DataFrame): Actual historical prices.
            - predicted_train_prices_df (pd.DataFrame): Predicted prices on the training set.
            - predicted_test_prices_df (pd.DataFrame): Predicted prices on the test set.
            - future_forecasted_prices_df (pd.DataFrame): Forecasted prices for the future period.
    """
    # Fetch historical stock price data from Yahoo Finance
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Fill any missing 'Close' prices with the median value
    prices = df['Close'].fillna(df['Close'].median())

    # Test if the price series is stationary
    test_stationarity(prices)

    # Convert prices to a numpy array and reshape for modeling
    prices = np.array(prices).reshape(-1, 1)

    # Create features and labels for the Linear Regression model
    X, y = create_features(prices.flatten(), window=window)
    X = X.reshape(X.shape[0], -1)

    # Split the data into training and testing sets
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

    # Train the Linear Regression model
    reg = LinearRegression().fit(xtrain, ytrain)

    # Save the trained model to a file using pickle
    with open('prediction.pickle', 'wb') as f:
        pickle.dump(reg, f)

    # Predict prices on the entire dataset
    predicted_prices = reg.predict(X)

    # Split the predicted prices into training and testing sets
    train_pred = predicted_prices[:len(xtrain)]
    test_pred = predicted_prices[len(xtrain):]

    # Generate future dates for the forecast period
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_period + 1)]

    # Forecast future prices using the trained model
    future_predicted_prices = []
    last_window = X[-1].reshape(1, -1)  # Start with the last window of data
    for _ in range(forecast_period):
        next_pred = reg.predict(last_window)
        future_predicted_prices.append(next_pred[0])
        last_window = np.append(last_window[:, 1:], next_pred).reshape(1, -1)  # Slide the window

    # Create DataFrame for actual prices
    actual_prices_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual Price': prices.flatten()
    })

    # Create DataFrame for predicted training prices
    predicted_train_prices_df = pd.DataFrame({
        'Date': df['Date'][window:window + len(train_pred)],
        'Predicted Price': train_pred.flatten()
    })

    # Create DataFrame for predicted testing prices
    predicted_test_prices_df = pd.DataFrame({
        'Date': df['Date'][window + len(train_pred):window + len(train_pred) + len(test_pred)],
        'Predicted Price': test_pred.flatten()
    })

    # Create DataFrame for forecasted future prices
    future_forecasted_prices_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Price': np.array(future_predicted_prices).flatten()
    })

    return actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df

def plot_result(actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df):
    """
    Plot the actual, predicted, and forecasted stock prices.

    Parameters:
        actual_prices_df (pd.DataFrame): DataFrame containing actual historical prices.
        predicted_train_prices_df (pd.DataFrame): DataFrame containing predicted prices on the training set.
        predicted_test_prices_df (pd.DataFrame): DataFrame containing predicted prices on the test set.
        future_forecasted_prices_df (pd.DataFrame): DataFrame containing forecasted prices for the future period.

    Returns:
        None: Displays an interactive plotly graph showing actual, predicted, and forecasted prices.
    """
    # Merge all DataFrames on the 'Date' column
    data = actual_prices_df.merge(predicted_train_prices_df, on='Date', how='left').merge(predicted_test_prices_df, on='Date', how='left').merge(future_forecasted_prices_df, on='Date', how='outer')

    # Define the initial range for the last 2 years of data
    last_date = pd.to_datetime(data['Date']).max()
    two_years_ago = last_date - pd.DateOffset(years=2)

    # Initialize a Plotly figure
    fig = go.Figure()

    # Add actual price line to the plot
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Actual Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add predicted train price line in green
    fig.add_trace(go.Scatter(
        x=predicted_train_prices_df['Date'],
        y=predicted_train_prices_df['Predicted Price'],
        mode='lines',
        name='Predicted Train Price',
        line=dict(color='green')
    ))

    # Add predicted test price line in yellow
    fig.add_trace(go.Scatter(
        x=predicted_test_prices_df['Date'],
        y=predicted_test_prices_df['Predicted Price'],
        mode='lines',
        name='Predicted Test Price',
        line=dict(color='orange')
    ))

    # Add forecasted price line in red
    fig.add_trace(go.Scatter(
        x=future_forecasted_prices_df['Date'],
        y=future_forecasted_prices_df['Forecasted Price'],
        mode='lines',
        name='Forecast Price',
        line=dict(color='red')
    ))

    # Update layout with titles, labels, and grid visibility
    fig.update_layout(
        title='Stock Prediction & Forecast using Linear Regression',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        legend=dict(
            x=0.5,
            y=-0.2,
            xanchor='center',
            yanchor='top',
            orientation='h'
        )
    )

    # Set initial range for the x-axis to the last 2 years and add a range slider
    fig.update_xaxes(
        range=[two_years_ago, last_date],
        rangeslider_visible=True  # Adds a range slider for zooming out
    )

    # Add a shaded rectangle to highlight the forecasted period
    fig.add_shape(
        type="rect",
        x0=future_forecasted_prices_df['Date'].iloc[0],  # Start date of the forecast period
        x1=future_forecasted_prices_df['Date'].iloc[-1],  # End date of the forecast period
        y0=0,  # Start y value of the background rectangle
        y1=1,  # End y value of the background rectangle, 1 represents the top of the plot
        xref="x",
        yref="paper",
        fillcolor="LightSkyBlue",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    fig.show()

# Testing functionality of this file
if __name__ == '__main__':
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2024-07-23'

    # Run the linear regression model and get actual, predicted, and forecasted prices
    actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df = LR_model(ticker, start, end)

    # Plot the results
    plot_result(actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df)
