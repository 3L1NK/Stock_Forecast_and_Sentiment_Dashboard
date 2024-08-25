from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd


def analyze_news_sentiment_twitter_roberta(ticker):
    """
    Analyzes the sentiment of news headlines for a given stock ticker using the Twitter RoBERTa model.
    The function loads the news data, applies sentiment analysis, and saves the results to a new CSV file.

    Parameters:
        ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

    Returns:
        pd.DataFrame: A DataFrame containing the original news data along with sentiment analysis results,
                      including positive, neutral, negative scores, and the Twitter RoBERTa Sentiment Score.
    """
    # Load the CSV file containing news data for the specified ticker
    news_data = pd.read_csv(f'news_{ticker}.csv')

    # Load the Twitter RoBERTa model, tokenizer, and configuration
    MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    config = AutoConfig.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL)

    def analyze_sentiment(headline):
        """
        Analyzes the sentiment of a single headline using Twitter RoBERTa.

        Parameters:
            headline (str): The news headline to analyze.

        Returns:
            list: A list of tuples containing sentiment labels ('positive', 'neutral', 'negative')
                  and their corresponding scores.
        """
        # Tokenize the headline and perform sentiment analysis
        encoded_input = tokenizer(headline, return_tensors='pt')
        output = model(**encoded_input)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        ranking = np.argsort(scores)[::-1]
        sentiments = [(config.id2label[ranking[i]], scores[ranking[i]]) for i in range(scores.shape[0])]
        return sentiments

    # Apply sentiment analysis on each headline in the 'Headline' column
    news_data['Sentiment'] = news_data['Headline'].apply(lambda x: analyze_sentiment(x))

    def extract_sentiment_scores(sentiment):
        """
        Extracts sentiment scores from the sentiment analysis output and maps them into separate columns.

        Parameters:
            sentiment (list): A list of tuples with sentiment labels and scores.

        Returns:
            pd.Series: A series with sentiment scores for 'positive', 'neutral', and 'negative'.
        """
        # Initialize a dictionary to hold sentiment scores, defaulting to 0.0
        score_dict = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for label, score in sentiment:
            score_dict[label] = score
        return pd.Series(score_dict)

    # Apply the extraction function to the 'Sentiment' column to create separate sentiment score columns
    sentiment_scores = news_data['Sentiment'].apply(extract_sentiment_scores)

    # Concatenate the new sentiment score columns with the original DataFrame
    news_data = pd.concat([news_data, sentiment_scores], axis=1)

    # Calculate the Twitter RoBERTa Sentiment Score by subtracting the negative score from the positive score
    news_data['Twitter RoBERTa Sentiment Score'] = news_data['positive'] - news_data['negative']

    # Save the DataFrame to a CSV file
    news_data.to_csv(f'news_{ticker}_twitter_roberta.csv', index=False)

    return news_data

# Set the stock ticker symbol and analyze the sentiment
ticker = 'AAPL'
# ticker = 'AMZN'
# ticker = 'GOOG'
# ticker = 'MSFT'

# Perform sentiment analysis on the news data for the specified ticker
news_data = analyze_news_sentiment_twitter_roberta(ticker)

# Print the first few rows of the resulting DataFrame to verify the results
print(news_data.head())
