import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta
import math
import matplotlib.pyplot as plt
import xgboost as xgb
import plotly.graph_objs as go


# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def XGBoost_model(ticker, start='2020-01-01', end='2024-01-01'):
    # Fetch data
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)
    dates = pd.to_datetime(df['Date'])

    # Take only Close data set into df1
    df1 = df.reset_index()['Close']
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Split dataset into train and test split
    training_size = int(len(df1) * 0.8)
    test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    # Reshape into X=t,t+1,t+2,t+3 and Y=t+4
    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, ytest = create_dataset(test_data, time_step)

    # Convert the data into DMatrix format for XGBoost
    train_dmatrix = xgb.DMatrix(X_train, label=y_train)
    test_dmatrix = xgb.DMatrix(X_test, label=ytest)

    # Parameters for XGBoost
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.1,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }

    # Train the model
    num_round = 400
    model = xgb.train(params, train_dmatrix, num_round, evals=[(test_dmatrix, 'test')])

    # Predict
    train_predict = model.predict(train_dmatrix)
    test_predict = model.predict(test_dmatrix)

    # Transform back to original form
    train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
    test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))

    rmse_training = math.sqrt(mean_squared_error(y_train, train_predict))
    print('RMSE training= ' + str(rmse_training))
    rmse_test = math.sqrt(mean_squared_error(ytest, test_predict))
    print('RMSE test= ' + str(rmse_test))

    # Shift train predictions for plotting
    look_back = 100

    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

    # Prediction for the next 30 days
    x_input = test_data[-100:].reshape(1, -1)
    temp_input = list(x_input[0])

    lst_output = []
    n_steps = 100
    i = 0
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[-100:])
        else:
            x_input = np.array(temp_input)

        x_input = xgb.DMatrix(x_input.reshape(1, -1))
        yhat = model.predict(x_input)
        temp_input.append(yhat[0])
        lst_output.append(yhat[0])
        i += 1

    # Prepare future dates for plotting
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
        rangeslider_visible=True
    )

    fig.show()

if __name__ == '__main__':
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2024-06-07'
    result_df, future_df = XGBoost_model(ticker, start, end)
    plot(result_df, future_df)