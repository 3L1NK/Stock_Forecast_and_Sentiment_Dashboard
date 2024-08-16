# import packages / libraries
from pytrends.request import TrendReq
import plotly.graph_objs as go
import pandas as pd
import time
import os


def plot_from_csv(file_name):
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)
    data = pd.read_csv(file_path, skiprows=1)

    data['Week'] = pd.to_datetime(data['Week'])
    data.set_index('Week', inplace=True)

    if 'apple' in file_name:
        ticker = 'Apple'
    elif 'amazon' in file_name:
        ticker = 'Amazon'
    elif 'microsoft' in file_name:
        ticker = 'Microsoft'
    else:
        ticker = 'Google'

    # Initialize a plotly figure
    fig = go.Figure()

    # Loop through each column (excluding the index) and add a trace for it
    for column in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))
    
    fig.update_layout(
        title=f'Google Trends Data for {ticker}',
        xaxis_title='Date',
        yaxis_title='Search Interest',
        legend_title='Search Terms'
    )
    return fig

# function to fetch trends data with delay
def get_trends_data(keywords):
    # initialize pytrends (connecting to google)
    pytrends = TrendReq()
    try:
        pytrends.build_payload(keywords, cat=0, timeframe='today 12-m', geo='', gprop='')
        time.sleep(5)  # Add a delay of 5 seconds between requests
        return pytrends.interest_over_time()
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# define keywords (can take up to 5 keywords)
keywords = ["Apple"]
# Fetch interest over time
interest_over_time_df = get_trends_data(keywords)

# Check if data is fetched successfully
if interest_over_time_df is not None and not interest_over_time_df.empty:
    print(interest_over_time_df.head())
    print(interest_over_time_df)
else:
    print("Failed to retrieve Google Trends data.")
