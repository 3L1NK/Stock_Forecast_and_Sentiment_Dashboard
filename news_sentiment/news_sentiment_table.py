import pandas as pd
import finnhub
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import pipeline

def fetch_news_for_ticker(ticker):
    # Initialize the Finnhub client
    finnhub_client = finnhub.Client(api_key='cpfiao9r01quaqfq8q4gcpfiao9r01quaqfq8q50')

    # Calculate the date range for 1 day
    date_to = datetime.today()
    date_from = date_to - timedelta(days=1)

    # Convert dates to strings
    date_from_str = date_from.strftime('%Y-%m-%d')
    date_to_str = date_to.strftime('%Y-%m-%d')

    # Fetch news for the single day
    news_response = finnhub_client.company_news(ticker, _from=date_from_str, to=date_to_str)

    # Convert the response to a DataFrame
    if news_response:
        news_df = pd.DataFrame(news_response)
    else:
        news_df = pd.DataFrame()

    # Convert the datetime field from UNIX timestamp to readable format
    if not news_df.empty:
        news_df['datetime'] = pd.to_datetime(news_df['datetime'], unit='s')

        # Filter out any news outside the exact 1-day window
        date_from_limit = date_from.date()
        date_to_limit = date_to.date()
        news_df = news_df[(news_df['datetime'].dt.date >= date_from_limit) &
                          (news_df['datetime'].dt.date <= date_to_limit)]

        # Drop unnecessary columns
        news_df = news_df.drop(columns=['category', 'image', 'related', 'source'])

        # Add separate date and time columns
        news_df = news_df.assign(
            date=news_df['datetime'].dt.date,
            time=news_df['datetime'].dt.time
        )

        # Rearrange the columns
        news_df = news_df[['id', 'date', 'time', 'datetime', 'headline', 'summary', 'url']]

        news_df.rename(
            columns={'id': 'ID', 'date': 'Date', 'time': 'Time', 'datetime': 'DateTime', 'headline': 'Headline', 'summary': 'Summary',
                     'url': 'URL'}, inplace=True)

        # Format the Date column as day.month.year
        news_df['Date'] = news_df['Date'].apply(lambda x: x.strftime('%d.%m.%Y'))

        # Check for duplicate rows and drop them
        news_df = news_df.drop_duplicates()

        # Select only the first 10 rows
        news_df = news_df.head(10)

    return news_df


def vader_news_sentiment(df):
    # Download VADER lexicon if not already downloaded
    nltk.download('vader_lexicon', quiet=True)

    # Sentiment Analysis on news article titles
    vader = SentimentIntensityAnalyzer()

    df['Positive Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['pos'])
    df['Negative Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['neg'])
    df['VADER Sentiment Score'] = df['Positive Score'] - df['Negative Score']

    # Apply rounding using a lambda function
    df['VADER Sentiment Score'] = df['VADER Sentiment Score'].apply(lambda x: round(x, 3))

    return df


def twitter_roBERTa_news_sentiment(df):
    MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Define a function to analyze sentiment of a text
    def analyze_sentiment(text):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        sentiments = [(config.id2label[ranking[i]], scores[ranking[i]]) for i in range(scores.shape[0])]
        return sentiments

    # Apply sentiment analysis on the 'headline' column
    df['Sentiment'] = df['Headline'].apply(lambda x: analyze_sentiment(x))

    # Define a function to extract sentiment scores into separate columns
    def extract_sentiment_scores(sentiment):
        score_dict = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for label, score in sentiment:
            score_dict[label] = score
        return pd.Series(score_dict)

    # Apply the function to the sentiment column
    sentiment_scores = df['Sentiment'].apply(extract_sentiment_scores)

    # Concatenate the new sentiment score columns with the original dataframe
    df = pd.concat([df, sentiment_scores], axis=1)

    # Calculate the Twitter RoBERTa Sentiment Score
    df['Twitter RoBERTa Sentiment Score'] = df['positive'] - df['negative']

    # Apply rounding using a lambda function
    df['Twitter RoBERTa Sentiment Score'] = df['Twitter RoBERTa Sentiment Score'].apply(lambda x: round(x, 3))

    return df


def distilroberta_news_sentiment(df):

    MODEL = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    # Define a function to analyze sentiment of a text
    def analyze_sentiment(text):
        encoded_input = tokenizer(text, return_tensors='pt')
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        sentiments = [(config.id2label[ranking[i]], scores[ranking[i]]) for i in range(scores.shape[0])]
        return sentiments

    # Apply sentiment analysis on the 'headline' column
    df['Sentiment'] = df['Headline'].apply(lambda x: analyze_sentiment(x))

    # Define a function to extract sentiment scores into separate columns
    def extract_sentiment_scores(sentiment):
        score_dict = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for label, score in sentiment:
            score_dict[label] = score
        return pd.Series(score_dict)

    # Apply the function to the sentiment column
    sentiment_scores = df['Sentiment'].apply(extract_sentiment_scores)

    # Concatenate the new sentiment score columns with the original dataframe
    df = pd.concat([df, sentiment_scores], axis=1)

    # Calculate DistilRoberta Sentiment Score
    df['DistilRoberta Sentiment Score'] = df['positive'] - df['negative']

    # Apply rounding using a lambda function
    df['DistilRoberta Sentiment Score'] = df['DistilRoberta Sentiment Score'].apply(lambda x: round(x, 3))

    return df


def BART_MNLI_impact_on_stock_price(df):
    # Load Twitter RoBERTa
    df = twitter_roBERTa_news_sentiment(df)

    # Load the BART MNLI model
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    # Define candidate labels
    candidate_labels = [
        'The news article suggests a positive impact on the stock price of Apple Inc.',
        'The news article suggests a negative impact on the stock price of Apple Inc.',
        'The news article suggests no significant impact on the stock price of Apple Inc.'
    ]

    # Function to classify and format the result
    def classify_with_mnli(headline, classifier, candidate_labels):
        result = classifier(headline, candidate_labels)
        # Get the highest probability label and its corresponding probability
        label = result['labels'][0]
        probability = result['scores'][0]

        # Shorten the label
        if 'positive impact' in label:
            shortened_label = 'positive'
        elif 'negative impact' in label:
            shortened_label = 'negative'
        elif 'no significant impact' in label:
            shortened_label = 'neutral'

        return f'{shortened_label}: {probability:.3f}'

    # Apply the classification and store in a new column
    df['BART MNLI: Impact on Stock Price'] = df['Headline'].apply(lambda headline: classify_with_mnli(headline, classifier, candidate_labels))

    return df


def combine_sentiment_models(df):
    # Get the dataframes from each sentiment analysis function
    df_vader = vader_news_sentiment(df.copy())
    df_twitter_roberta = twitter_roBERTa_news_sentiment(df.copy())
    df_distilroberta = distilroberta_news_sentiment(df.copy())
    df_bart_mnli = BART_MNLI_impact_on_stock_price(df.copy())

    # Select relevant columns to merge (ID and DateTime) and their respective sentiment scores
    df_vader = df_vader[['ID', 'Date', 'Time', 'DateTime', 'Headline', 'Summary', 'URL', 'VADER Sentiment Score']]
    df_twitter_roberta = df_twitter_roberta[['ID', 'DateTime', 'Twitter RoBERTa Sentiment Score']]
    df_distilroberta = df_distilroberta[['ID', 'DateTime', 'DistilRoberta Sentiment Score']]
    df_bart_mnli = df_bart_mnli[['ID', 'DateTime', 'BART MNLI: Impact on Stock Price']]


    # Merge dataframes on 'ID' and 'DateTime'
    combined_df = df_vader.merge(df_twitter_roberta, on=['ID', 'DateTime'], how='inner')
    combined_df = combined_df.merge(df_distilroberta, on=['ID', 'DateTime'], how='inner')
    combined_df = combined_df.merge(df_bart_mnli, on=['ID', 'DateTime'], how='inner')

    return combined_df


# Example usage
#ticker = 'AAPL'
# Step 1: Fetch news for the given ticker
#news_df = fetch_news_for_ticker(ticker)
# Step 2: Combine sentiment models
#combined_sentiment_df = combine_sentiment_models(news_df)
#print(combined_sentiment_df)