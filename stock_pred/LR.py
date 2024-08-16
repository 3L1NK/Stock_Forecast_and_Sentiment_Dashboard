from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import pickle
import yfinance as yf
from datetime import timedelta
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from statsmodels.tsa.stattools import adfuller
import plotly.express as px

def test_stationarity(timeseries):
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
    
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print('Results of Dickey-Fuller Test:')
    print(dfoutput)

def create_features(data, window=40):
    X, y = [], []
    for i in range(window, len(data)):
        X.append(data[i-window:i])
        y.append(data[i])
    return np.array(X), np.array(y)

def LR_model(ticker, start='2020-01-01', end='2024-01-01', forecast_period=90, window=40):
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)
    df['Date'] = pd.to_datetime(df['Date'])
    prices = df['Close'].fillna(df['Close'].median())
    
    test_stationarity(prices)

    prices = np.array(prices).reshape(-1, 1)
    X, y = create_features(prices.flatten(), window=window)
    X = X.reshape(X.shape[0], -1)

    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)
    reg = LinearRegression().fit(xtrain, ytrain)

    with open('prediction.pickle', 'wb') as f:
        pickle.dump(reg, f)

    predicted_prices = reg.predict(X)

    # Split the predicted prices into training and testing sets
    train_pred = predicted_prices[:len(xtrain)]
    test_pred = predicted_prices[len(xtrain):]

    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, forecast_period + 1)]

    future_predicted_prices = []
    last_window = X[-1].reshape(1, -1)
    for _ in range(forecast_period):
        next_pred = reg.predict(last_window)
        future_predicted_prices.append(next_pred[0])
        last_window = np.append(last_window[:, 1:], next_pred).reshape(1, -1)

    actual_prices_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual Price': prices.flatten()
    })

    predicted_train_prices_df = pd.DataFrame({
        'Date': df['Date'][window:window + len(train_pred)],
        'Predicted Price': train_pred.flatten()
    })

    predicted_test_prices_df = pd.DataFrame({
        'Date': df['Date'][window + len(train_pred):window + len(train_pred) + len(test_pred)],
        'Predicted Price': test_pred.flatten()
    })

    future_forecasted_prices_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Price': np.array(future_predicted_prices).flatten()
    })

    return actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df

def plot_result(actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df):
    data = actual_prices_df.merge(predicted_train_prices_df, on='Date', how='left').merge(predicted_test_prices_df, on='Date', how='left').merge(future_forecasted_prices_df, on='Date', how='outer')

    # Define the initial range for the last 2 years
    last_date = pd.to_datetime(data['Date']).max()
    two_years_ago = last_date - pd.DateOffset(years=2)

    fig = go.Figure()

    # Add actual price line
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

    fig.update_xaxes(
        range=[two_years_ago, last_date],
        rangeslider_visible=True  # Adds a range slider for zooming out
    )

    fig.add_shape(
        type="rect",
        x0=future_forecasted_prices_df['Date'].iloc[0],  # start date of the range
        x1=future_forecasted_prices_df['Date'].iloc[-1],  # end date of the range
        y0=0,  # start y value of the background rectangle
        y1=1,  # end y value of the background rectangle, 1 represents the top of the plot
        xref="x",
        yref="paper",
        fillcolor="LightSkyBlue",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    fig.show()

if __name__ == '__main__':
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2024-07-23'
    actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df = LR_model(ticker, start, end)
    plot_result(actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df)
