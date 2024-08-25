import pandas as pd
import finnhub
from datetime import datetime, timedelta
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
from transformers import pipeline
import logging
from transformers import logging as transformers_logging

def fetch_news_for_ticker(ticker):
    """
    Fetches news articles for a given stock ticker from Finnhub for the last day.

    Parameters:
        ticker (str): Stock ticker symbol.

    Returns:
        pd.DataFrame: A DataFrame containing the most recent news articles for the specified ticker,
                      with columns such as 'ID', 'Date', 'Time', 'DateTime', 'Headline', 'Summary', and 'URL'.
    """
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

        # Rename columns for readability
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
    """
    Analyzes the sentiment of news headlines using the VADER sentiment analysis tool.

    Parameters:
        df (pd.DataFrame): DataFrame containing news articles with a 'Headline' column.

    Returns:
        pd.DataFrame: Original DataFrame with added columns for Positive Score, Negative Score, and VADER Sentiment Score.
    """
    # Download VADER lexicon if not already downloaded
    nltk.download('vader_lexicon', quiet=True)

    # Sentiment Analysis on news article titles
    vader = SentimentIntensityAnalyzer()

    # Calculate the positive and negative sentiment score for each headline
    df['Positive Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['pos'])
    df['Negative Score'] = df['Headline'].apply(lambda title: vader.polarity_scores(title)['neg'])

    # Calculate the VADER Sentiment Score by subtracting the negative score from the positive score
    df['VADER Sentiment Score'] = df['Positive Score'] - df['Negative Score']

    # Round the VADER Sentiment Score to three decimal places
    df['VADER Sentiment Score'] = df['VADER Sentiment Score'].apply(lambda x: round(x, 3))

    return df


def twitter_roberta_news_sentiment(df):
    """
    Analyzes the sentiment of news headlines using the Twitter RoBERTa model.

    Parameters:
        df (pd.DataFrame): DataFrame containing news articles with a 'Headline' column.

    Returns:
        pd.DataFrame: Original DataFrame with added columns for positive, neutral, negative sentiment scores,
                      and the Twitter RoBERTa Sentiment Score.
    """
    # Suppress the warning related to unused weights
    logging.basicConfig(level=logging.ERROR)
    transformers_logging.set_verbosity_error()

    # Load the Twitter RoBERTa model and tokenizer
    MODEL = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def analyze_sentiment(headline):
        """
        Analyzes the sentiment of a single text using the Twitter RoBERTa model.

        Parameters:
            headline (str): The news headline to analyze.

        Returns:
            list: A list of tuples containing sentiment labels and their corresponding scores.
        """
        # Tokenize the headline and perform sentiment analysis
        encoded_input = tokenizer(headline, return_tensors='pt')
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        sentiments = [(config.id2label[ranking[i]], scores[ranking[i]]) for i in range(scores.shape[0])]
        return sentiments

    # Apply sentiment analysis on the 'headline' column
    df['Sentiment'] = df['Headline'].apply(lambda x: analyze_sentiment(x))

    def extract_sentiment_scores(sentiment):
        """
        Extracts sentiment scores into separate columns for positive, neutral, and negative sentiments.

        Parameters:
            sentiment (list): A list of tuples containing sentiment labels and their corresponding scores.

        Returns:
            pd.Series: A series with sentiment scores.
        """
        # Initialize a dictionary to hold sentiment scores, defaulting to 0.0
        score_dict = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for label, score in sentiment:
            score_dict[label] = score
        return pd.Series(score_dict)

    # Apply the extraction function to the 'Sentiment' column to create separate sentiment score columns
    sentiment_scores = df['Sentiment'].apply(extract_sentiment_scores)

    # Concatenate the new sentiment score columns with the original dataframe
    df = pd.concat([df, sentiment_scores], axis=1)

    # Calculate the Twitter RoBERTa Sentiment Score by subtracting the negative score from the positive score
    df['Twitter RoBERTa Sentiment Score'] = df['positive'] - df['negative']

    # Round the Twitter RoBERTa Sentiment Score to three decimal places
    df['Twitter RoBERTa Sentiment Score'] = df['Twitter RoBERTa Sentiment Score'].apply(lambda x: round(x, 3))

    return df


def distilroberta_news_sentiment(df):
    """
    Analyzes the sentiment of news headlines using the DistilRoberta model fine-tuned for financial news.

    Parameters:
        df (pd.DataFrame): DataFrame containing news articles with a 'Headline' column.

    Returns:
        pd.DataFrame: Original DataFrame with added columns for positive, neutral, negative sentiment scores,
                      and the DistilRoberta Sentiment Score.
    """
    # Load the DistilRoberta model and tokenizer
    MODEL = 'mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis'
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def analyze_sentiment(headline):
        """
        Analyzes the sentiment of a single text using the DistilRoberta model.

        Parameters:
            headline (str): The news headline to analyze.

        Returns:
            list: A list of tuples containing sentiment labels and their corresponding scores.
        """
        # Tokenize the headline and perform sentiment analysis
        encoded_input = tokenizer(headline, return_tensors='pt')
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        sentiments = [(config.id2label[ranking[i]], scores[ranking[i]]) for i in range(scores.shape[0])]
        return sentiments

    # Apply sentiment analysis on the 'headline' column
    df['Sentiment'] = df['Headline'].apply(lambda x: analyze_sentiment(x))

    def extract_sentiment_scores(sentiment):
        """
        Extracts sentiment scores into separate columns for positive, neutral, and negative sentiments.

        Parameters:
            sentiment (list): A list of tuples containing sentiment labels and their corresponding scores.

        Returns:
            pd.Series: A series with sentiment scores.
        """
        # Initialize a dictionary to hold sentiment scores, defaulting to 0.0
        score_dict = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for label, score in sentiment:
            score_dict[label] = score
        return pd.Series(score_dict)

    # Apply the function to the sentiment column
    sentiment_scores = df['Sentiment'].apply(extract_sentiment_scores)

    # Concatenate the new sentiment score columns with the original dataframe
    df = pd.concat([df, sentiment_scores], axis=1)

    # Calculate DistilRoberta Sentiment Score by subtracting the negative score from the positive score
    df['DistilRoberta Sentiment Score'] = df['positive'] - df['negative']

    # Round the DistilRoberta Sentiment Score to three decimal places
    df['DistilRoberta Sentiment Score'] = df['DistilRoberta Sentiment Score'].apply(lambda x: round(x, 3))

    return df


def bart_large_mnli_impact_on_stock_price(df):
    """
    Analyzes the impact of news articles on stock prices using the BART Large MNLI model.

    Parameters:
        df (pd.DataFrame): DataFrame containing news articles with a 'Headline' column.

    Returns:
        pd.DataFrame: Original DataFrame with an added column indicating the predicted impact on stock prices.
    """
    # Load Twitter RoBERTa for initial sentiment analysis
    df = twitter_roberta_news_sentiment(df)

    # Load the BART Large MNLI model for zero-shot classification
    classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

    # Define candidate labels representing different impacts on stock price
    candidate_labels = [
        'The news article suggests a positive impact on the stock price of Apple Inc.',
        'The news article suggests a negative impact on the stock price of Apple Inc.',
        'The news article suggests no significant impact on the stock price of Apple Inc.'
    ]

    def classify_with_bart_large_mnli(headline, classifier, candidate_labels):
        """
        Classifies a headline into one of the candidate labels using BART Large MNLI.

        Parameters:
            headline (str): The headline to classify.
            classifier: The BART Large MNLI classification pipeline.
            candidate_labels (list): List of candidate labels for classification.

        Returns:
            str: The label with the highest probability and its associated probability.
        """
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

        # Return the shortened label with the probability
        return f'{shortened_label}: {probability:.3f}'

    # Apply the classification and store in a new column
    df['BART MNLI: Impact on Stock Price'] = df['Headline'].apply(lambda headline: classify_with_bart_large_mnli(headline, classifier, candidate_labels))

    return df


def combine_news_analysis_models(df):
    """
    Combines the results from multiple news analysis models, including sentiment analysis models
    and a model for predicting the impact of news on stock prices, into a single DataFrame.

    Parameters:
        df (pd.DataFrame): DataFrame containing news articles with a 'Headline' column.

    Returns:
        pd.DataFrame: Combined DataFrame with analysis results from VADER, Twitter RoBERTa, DistilRoberta, and BART Large MNLI models.
    """
    # Get the dataframes from each analysis function
    df_vader = vader_news_sentiment(df.copy())
    df_twitter_roberta = twitter_roberta_news_sentiment(df.copy())
    df_distilroberta = distilroberta_news_sentiment(df.copy())
    df_bart_mnli = bart_large_mnli_impact_on_stock_price(df.copy())

    # Select relevant columns to merge (ID and DateTime) and their respective analysis results
    df_vader = df_vader[['ID', 'Date', 'Time', 'DateTime', 'Headline', 'Summary', 'URL', 'VADER Sentiment Score']]
    df_twitter_roberta = df_twitter_roberta[['ID', 'DateTime', 'Twitter RoBERTa Sentiment Score']]
    df_distilroberta = df_distilroberta[['ID', 'DateTime', 'DistilRoberta Sentiment Score']]
    df_bart_mnli = df_bart_mnli[['ID', 'DateTime', 'BART MNLI: Impact on Stock Price']]


    # Merge dataframes on 'ID' and 'DateTime'
    combined_df = df_vader.merge(df_twitter_roberta, on=['ID', 'DateTime'], how='inner')
    combined_df = combined_df.merge(df_distilroberta, on=['ID', 'DateTime'], how='inner')
    combined_df = combined_df.merge(df_bart_mnli, on=['ID', 'DateTime'], how='inner')

    return combined_df
