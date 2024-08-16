import numpy as np
import yfinance as yf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from datetime import timedelta
import plotly.graph_objs as go


def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

def LSTM_model(ticker, start='2020-01-01', end='2024-01-01'):
    print("Downloading data")
    df = yf.download(ticker, start, end)
    df.reset_index(inplace=True)
    dates = pd.to_datetime(df['Date'])

    df1 = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(df1)

    training_size = int(len(df1) * 0.8)
    #test_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

    time_step = 100
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    print("Building model")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(100, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(25))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    print("Training model")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64, verbose=1)

    print("Model training complete")
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    rmse_training = np.sqrt(mean_squared_error(y_train, train_predict))
    print('RMSE training= ' + str(rmse_training))
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predict))
    print('RMSE test= ' + str(rmse_test))

    look_back = 100

    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

    x_input = test_data[-100:].reshape(1, -1)
    temp_input = list(x_input[0])

    lst_output = []
    n_steps = 100
    i = 0
    while i < 30:
        if len(temp_input) > 100:
            x_input = np.array(temp_input[-100:])
            x_input = x_input.reshape((1, n_steps, 1))
        else:
            x_input = np.array(temp_input)
            x_input = x_input.reshape((1, len(temp_input), 1))

        yhat = model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        lst_output.append(yhat[0][0])
        i += 1

    last_date = df['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 31)]
    future_dates = pd.to_datetime(future_dates)

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


def plotting(result_df, future_df):
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
        legend_title='Legend',
        legend=dict(
                yanchor="bottom",
                y = 0.05,
                x = 0.02, 
                xanchor="left"
            )
    )

    #Set the range to include both the historical and forecasted dates
    two_years_ago = result_df['Date'].max() - pd.DateOffset(years=2)
    fig.update_xaxes(
        range=[two_years_ago, future_df['Date'].max()],
        rangeslider_visible=True
    )

    fig.show()


if __name__=='__main__':
    result_df, future_df = LSTM_model('AAPL', start='2020-01-01', end='2024-06-07')
    #plotting(result_df,future_df)

