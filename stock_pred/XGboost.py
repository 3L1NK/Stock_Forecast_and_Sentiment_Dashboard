import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import math
import xgboost as xgb
import plotly.graph_objs as go

def create_dataset(dataset, time_step=1):
    """
    Convert an array of values into a dataset matrix for time series forecasting.

    Parameters:
        dataset (np.array): The time series data to be used for creating the dataset.
        time_step (int): The number of previous time steps to consider for predicting the next value.

    Returns:
        tuple: A tuple containing:
            - dataX (np.array): The input features for the model.
            - dataY (np.array): The target labels for the model.
    """
    dataX, dataY = [], []
    # Create sliding windows of size 'time_step' to generate features and labels
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def XGBoost_model(ticker, start='2020-01-01', end='2024-01-01'):
    """
    Builds, trains, and evaluates an XGBoost model for stock price prediction.

    Parameters:
        ticker (str): The stock ticker symbol.
        start (str): The start date for fetching the data.
        end (str): The end date for fetching the data.

    Returns:
        tuple: A tuple containing two DataFrames:
            - result_df (pd.DataFrame): A DataFrame containing the actual, training predicted, and testing predicted prices.
            - future_df (pd.DataFrame): A DataFrame containing the forecasted prices for a future period.
    """
    # Fetch historical stock data using yfinance
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)

    # Convert 'Date' column to datetime format
    dates = pd.to_datetime(df['Date'])

    # Use only the 'Close' price for prediction
    df1 = df.reset_index()['Close']

    # Initialize a MinMaxScaler to scale the data to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    # Scale the data
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Split dataset into train and test sets
    training_size = int(len(df1) * 0.8)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    # Create features and labels using the create_dataset function
    # Reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100  # Define the time step (window size)
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Convert the training and testing data to DMatrix format, which is required by XGBoost
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_dmatrix = xgb.DMatrix(X_test, label=ytest)

    # Parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',   # Use squared error for regression
        'eval_metric': 'rmse',  # Use root mean squared error as the evaluation metric
        'eta': 0.1,  # Learning rate
        'max_depth': 6,  # Maximum depth of a tree
        'subsample': 0.8,  # Subsample ratio of the training instance
        'colsample_bytree': 0.8  # Subsample ratio of columns when constructing each tree
    }

    # Train the model
    num_round = 400  # Number of boosting rounds
    model = xgb.train(params, train_dmatrix, num_round, evals=[(test_dmatrix, 'test')])

    # Make predictions on the training and testing data
    train_predict = model.predict(train_dmatrix)
    test_predict = model.predict(test_dmatrix)

    # Inverse transform the predictions to the original scale
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))

    # Calculate RMSE for training and testing predictions
    rmse_training = math.sqrt(mean_squared_error(y_train, train_predict))
    print('RMSE training= ' + str(rmse_training))
    rmse_test = math.sqrt(mean_squared_error(ytest, test_predict))
    print('RMSE test= ' + str(rmse_test))

    # Prepare arrays for plotting the predictions
    # Shift train predictions for plotting
    look_back = 100  # Should match the time_step used to create features

    # Initialize arrays to hold the predictions for plotting
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

    # Predict stock prices for the next 30 days
    x_input = test_data[-100:].reshape(1, -1)
    temp_input = list(x_input[0])

    lst_output = []
    n_steps = 100
    i = 0
    while i < 30:  # Forecast for the next 30 days
        if len(temp_input) > 100:
            x_input = np.array(temp_input[-100:])
        else:
            x_input = np.array(temp_input)

        x_input = xgb.DMatrix(x_input.reshape(1, -1))  # Convert to DMatrix format for prediction
        yhat = model.predict(x_input)  # Predict the next value
        temp_input.append(yhat[0])   # Add the predicted value to the input sequence
        lst_output.append(yhat[0])  # Store the predicted value
        i += 1

    # Prepare future dates for plotting the forecast
    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    future_dates = pd.to_datetime(future_dates)

    # Prepare DataFrame for plotting
    future_predictions = scaler.inverse_transform(np.array(lst_output).reshape(-1, 1))

    result_df = pd.DataFrame({
        'Date': df['Date'],
        'Actual Price': scaler.inverse_transform(df1).flatten(),
        'Training Predicted Price': np.append(trainPredictPlot.flatten(), [np.nan] * (len(df1) - len(trainPredictPlot))),
        'Testing Predicted Price': np.append(testPredictPlot.flatten(), [np.nan] * (len(df1) - len(testPredictPlot))),
    })

    future_df = pd.DataFrame({
        'Date': future_dates,
        'Forecasted Price': future_predictions.flatten()
    })

    return result_df, future_df

def plot(result_df, future_df):
    """
    Plots the actual, predicted, and forecasted stock prices using Plotly.

    Parameters:
        result_df (pd.DataFrame): A DataFrame containing actual, training predicted, and testing predicted prices.
        future_df (pd.DataFrame): A DataFrame containing forecasted prices for a future period.

    Returns:
        None: Displays an interactive Plotly graph showing actual, predicted, and forecasted prices.
    """
    fig = go.Figure()

# Add actual price line
    fig.add_trace(go.Scatter(
        x=result_df['Date'],
        y=result_df['Actual Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add training predicted price line
    fig.add_trace(go.Scatter(
        x=result_df['Date'],
        y=result_df['Training Predicted Price'],
        mode='lines',
        name='Training Predicted Price',
        line=dict(color='green')
    ))

    # Add testing predicted price line
    fig.add_trace(go.Scatter(
        x=result_df['Date'],
        y=result_df['Testing Predicted Price'],
        mode='lines',
        name='Testing Predicted Price',
        line=dict(color='orange')
    ))

    # Add forecasted price line
    fig.add_trace(go.Scatter(
        x=future_df['Date'],
        y=future_df['Forecasted Price'],
        mode='lines',
        name='Forecasted Price',
        line=dict(color='red')
    ))

    # Add a light blue shaded area for the forecast period
    fig.add_shape(
        type="rect",
        x0=future_df['Date'].iloc[0],
        x1=future_df['Date'].iloc[-1],
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        fillcolor="LightSkyBlue",
        opacity=0.3,
        layer="below",
        line_width=0,
    )

    # Update layout
    fig.update_layout(
        title='Stock Price Prediction & Forecast using LSTM',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5
        )
    )

    # Set the range to include both the historical and forecasted dates
    two_years_ago = result_df['Date'].max() - pd.DateOffset(years=2)
    fig.update_xaxes(
        range=[two_years_ago, future_df['Date'].max()],
        rangeslider_visible=True  # Adds a range slider for zooming out
    )

    fig.show()

# Testing functionality
if __name__ == '__main__':
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2024-06-07'

    # Train the model and get predictions
    result_df, future_df = XGBoost_model(ticker, start, end)

    # Plot the actual, predicted, and forecasted prices
    plot(result_df, future_df)
