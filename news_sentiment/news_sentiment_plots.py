import pandas as pd
from datetime import datetime
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

def calculate_sentiment_moving_avg(file_path, model_name):
    # Load the data
    df = pd.read_csv(file_path)

    # If using VADER, calculate sentiment scores based on headlines
    if model_name == 'VADER Sentiment':
        nltk.download('vader_lexicon', quiet=True) # Download VADER lexicon if not already downloaded
        vader = SentimentIntensityAnalyzer()
        df['Positive Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['pos'])
        df['Negative Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['neg'])
        df[f'{model_name} Score'] = df['Positive Score'] - df['Negative Score']

    # Convert the 'DateTime' column to datetime format
    df['DateTime'] = pd.to_datetime(df['DateTime'])

    # Calculate the daily average sentiment score
    df_daily_avg = df.groupby(df['DateTime'].dt.date)[f'{model_name} Score'].mean().reset_index()
    df_daily_avg.columns = ['Date', f'Daily Average {model_name} Score']

    # Convert the Date column back to datetime format
    df_daily_avg['Date'] = pd.to_datetime(df_daily_avg['Date'])

    # Calculate the moving average with a window size of 10 days
    df_daily_avg['10-Day Moving Average'] = df_daily_avg[f'Daily Average {model_name} Score'].rolling(window=10).mean()

    return df_daily_avg