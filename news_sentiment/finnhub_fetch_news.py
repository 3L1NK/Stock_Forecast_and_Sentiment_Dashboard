import pandas as pd
import finnhub
from datetime import datetime, timedelta

def fetch_and_process_news_data(ticker):
    """
    Fetches and processes news articles for a given stock ticker from Finnhub over the past 160 days.
    The function fetches news in chunks, processes the data, and saves it to a CSV file.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
        pd.DataFrame: A DataFrame containing the processed news articles, with columns such as 'ID', 'Date', 'Time', 'DateTime', 'Headline', 'Summary', and 'URL'.
    """
    # Initialize the Finnhub client
    finnhub_client = finnhub.Client(api_key='cpfiao9r01quaqfq8q4gcpfiao9r01quaqfq8q50')

    # Define the date range: from 160 days ago to today
    date_to = datetime.today()
    date_from = date_to - timedelta(days=160)

    # Initialize an empty DataFrame to store all fetched news articles
    all_news_df = pd.DataFrame()

    # Fetch news in chunks of 3 days until the entire date range is covered
    while date_from < date_to:
        # Convert date range to string format for API request
        date_from_str = date_from.strftime('%Y-%m-%d')
        next_date_to = date_from + timedelta(days=3)
        if next_date_to > date_to:
            next_date_to = date_to
        date_to_str = next_date_to.strftime('%Y-%m-%d')

        # Fetch news for the current chunk of dates
        news_response = finnhub_client.company_news(ticker, _from=date_from_str, to=date_to_str)

        # If news is fetched, convert to DataFrame and append to the DataFrame all_news_df
        if news_response:
            news_df = pd.DataFrame(news_response)
            all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)

        # Move to the next date range
        date_from = next_date_to

    # Convert the datetime field from UNIX timestamp to readable format
    all_news_df['datetime'] = pd.to_datetime(all_news_df['datetime'], unit='s')

    # Filter out any news outside the exact 160-day window
    date_from_limit = (datetime.today() - timedelta(days=160)).date()
    date_to_limit = datetime.today().date()
    all_news_df = all_news_df[(all_news_df['datetime'].dt.date >= date_from_limit) &
                              (all_news_df['datetime'].dt.date <= date_to_limit)]

    # Drop unnecessary columns from the DataFrame
    all_news_df = all_news_df.drop(columns=['category', 'image', 'related', 'source'])

    # Add separate date and time columns
    all_news_df = all_news_df.assign(
        date=all_news_df['datetime'].dt.date,
        time=all_news_df['datetime'].dt.time
    )

    # Rearrange the columns
    all_news_df = all_news_df[['id', 'date', 'time', 'datetime', 'headline', 'summary', 'url']]

    # Rename columns for readability
    all_news_df.rename(columns={'id': 'ID', 'date': 'Date', 'time': 'Time', 'datetime': 'DateTime', 'headline': 'Headline', 'summary': 'Summary','url': 'URL'}, inplace=True)

    # Sort the news articles by Date and Time in descending order
    all_news_df.sort_values(by=['Date', 'Time'], ascending=False, inplace=True)

    # Remove any duplicate rows from the DataFrame
    all_news_df = all_news_df.drop_duplicates()

    # Save the processed news DataFrame to a CSV file named after the ticker symbol
    all_news_df.to_csv(f'news_{ticker}.csv', index=False)

    return all_news_df

def verify_missing_dates(all_news_df):
    """
    Verifies whether there are any missing dates in the news data within the last 160 days.

    Parameters:
        all_news_df (pd.DataFrame): A DataFrame containing the processed news articles with a 'Date' column.

    Returns:
        list: A list of missing dates (as strings) where no news articles were fetched.
    """
    # Calculate the date range to verify (past 160 days)
    date_from_limit = (datetime.today() - timedelta(days=160)).date()
    date_to_limit = datetime.today().date()

    # Create a list of all dates within the range
    all_dates = pd.date_range(start=date_from_limit, end=date_to_limit)

    # Extract the fetched dates from the news DataFrame
    fetched_dates = all_news_df['Date'].unique()

    # Identify missing dates by comparing all_dates with fetched_dates
    missing_dates = set(all_dates.date) - set(fetched_dates)

    # Format missing dates as strings
    missing_dates_str = [date.strftime('%Y-%m-%d') for date in missing_dates]

    # Display the missing dates
    print("Missing dates:", missing_dates_str)

    return missing_dates_str


# Define the ticker symbol to fetch and process news for
ticker = 'AAPL'
# ticker = 'AMZN'
# ticker = 'GOOG'
# ticker = 'MSFT'

# Fetch and process news data for the specified ticker
all_news_df = fetch_and_process_news_data(ticker)

# Verify and identify any missing dates in the fetched news data
missing_dates = verify_missing_dates(all_news_df)

