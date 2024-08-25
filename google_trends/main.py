# Import packages / libraries
from pytrends.request import TrendReq
import plotly.graph_objs as go
import pandas as pd
import time
import os


def plot_from_csv(file_name):
    """
    Plots Google Trends data from a CSV file using Plotly.

    Parameters:
        file_name (str): The name of the CSV file containing Google Trends data.

    Returns:
        plotly.graph_objs.Figure: A Plotly figure object containing the trends plot.
    """
    # Get the current working directory and construct the full file path
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, file_name)

    # Read the CSV file into a DataFrame, skipping the first row
    data = pd.read_csv(file_path, skiprows=1)

    # Convert the 'Week' column to datetime format and set it as the index
    data['Week'] = pd.to_datetime(data['Week'])
    data.set_index('Week', inplace=True)

    # Determine the ticker symbol based on the file name
    if 'apple' in file_name:
        ticker = 'Apple'
    elif 'amazon' in file_name:
        ticker = 'Amazon'
    elif 'microsoft' in file_name:
        ticker = 'Microsoft'
    else:
        ticker = 'Google'

    # Initialize a Plotly figure
    fig = go.Figure()

    # Loop through each column (excluding the index) in the DataFrame and add a trace to the plot for each search term
    for column in data.columns:
        fig.add_trace(go.Scatter(x=data.index, y=data[column], mode='lines', name=column))

    # Update the layout of the plot with titles and labels
    fig.update_layout(
        title=f'Google Trends Data for {ticker}',
        xaxis_title='Date',
        yaxis_title='Search Interest',
        legend_title='Search Terms'
    )

    return fig

# function to fetch trends data with delay
def get_trends_data(keywords):
    """
    Fetches Google Trends data for a list of keywords with a delay to avoid rate limiting.

    Parameters:
        keywords (list): A list of up to 5 keywords to search on Google Trends.

    Returns:
        pd.DataFrame: A DataFrame containing the interest over time for the specified keywords.
                      Returns None if an error occurs or no data is retrieved.
    """
    # Initialize the Pytrends connection to Google
    pytrends = TrendReq()
    try:
        # Build the payload with the specified keywords and fetch the interest over time data
        pytrends.build_payload(keywords, cat=0, timeframe='today 12-m', geo='', gprop='')
        # Add a delay of 5 seconds between requests to avoid being rate limited by Google
        time.sleep(5)
        # Return the interest over time data
        return pytrends.interest_over_time()
    except Exception as e:
        # Print an error message if fetching the data fails
        print(f"An error occurred: {e}")
        return None

# Define keywords for which to fetch Google Trends data (can take up to 5 keywords)
keywords = ["Apple"]

# Fetch interest over time data using the get_trends_data function
interest_over_time_df = get_trends_data(keywords)

# Check if data is fetched successfully and display it
if interest_over_time_df is not None and not interest_over_time_df.empty:
    print(interest_over_time_df)  # Print the entire DataFrame
else:
    print("Failed to retrieve Google Trends data.")  # Error message if data retrieval fails
