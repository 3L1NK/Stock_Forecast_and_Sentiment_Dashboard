import pandas as pd
import finnhub
from datetime import datetime, timedelta

def fetch_and_process_news_data(ticker):
    # Initialize the Finnhub client
    finnhub_client = finnhub.Client(api_key='cpfiao9r01quaqfq8q4gcpfiao9r01quaqfq8q50')

    # Calculate the date range for the past 160 days
    date_to = datetime.today()
    date_from = date_to - timedelta(days=160)

    # Initialize an empty DataFrame to store all news
    all_news_df = pd.DataFrame()

    # Fetch news in chunks of `chunk_size` days
    while date_from < date_to:
        date_from_str = date_from.strftime('%Y-%m-%d')
        next_date_to = date_from + timedelta(days=3)
        if next_date_to > date_to:
            next_date_to = date_to
        date_to_str = next_date_to.strftime('%Y-%m-%d')

        news_response = finnhub_client.company_news(ticker, _from=date_from_str, to=date_to_str)

        # Convert the response to a DataFrame and append to all_news_df
        if news_response:
            news_df = pd.DataFrame(news_response)
            all_news_df = pd.concat([all_news_df, news_df], ignore_index=True)

        # Move to the next date range
        date_from = next_date_to

    # Convert the datetime field from UNIX timestamp to readable format
    all_news_df['datetime'] = pd.to_datetime(all_news_df['datetime'], unit='s')

    # Filter out any news outside the exact `days_back` window
    date_from_limit = (datetime.today() - timedelta(days=160)).date()
    date_to_limit = datetime.today().date()
    all_news_df = all_news_df[(all_news_df['datetime'].dt.date >= date_from_limit) &
                              (all_news_df['datetime'].dt.date <= date_to_limit)]

    # Drop unnecessary columns
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

    # Sort by Date and Time
    all_news_df.sort_values(by=['Date', 'Time'], ascending=False, inplace=True)

    # Drop all duplicate rows
    all_news_df = all_news_df.drop_duplicates()

    # Save the DataFrame to a CSV file
    all_news_df.to_csv(f'news_{ticker}.csv', index=False)

    return all_news_df

def verify_missing_dates(all_news_df):
    # Calculate the date range to verify
    date_from_limit = (datetime.today() - timedelta(days=160)).date()
    date_to_limit = datetime.today().date()

    # Create a list of all dates in the range
    all_dates = pd.date_range(start=date_from_limit, end=date_to_limit)

    # Extract the fetched dates from the DataFrame
    fetched_dates = all_news_df['Date'].unique()

    # Identify missing dates
    missing_dates = set(all_dates.date) - set(fetched_dates)

    # Format missing dates as strings
    missing_dates_str = [date.strftime('%Y-%m-%d') for date in missing_dates]

    # Display the missing dates
    print("Missing dates:", missing_dates_str)

    return missing_dates_str


# Define the ticker symbol
# ticker = 'AAPL'
# ticker = 'AMZN'
# ticker = 'GOOG'
ticker = 'MSFT'
all_news_df = fetch_and_process_news_data(ticker)
missing_dates = verify_missing_dates(all_news_df)

