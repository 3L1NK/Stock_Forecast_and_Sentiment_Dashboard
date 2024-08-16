from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import pandas as pd


def analyze_news_sentiment_distilroberta(ticker):
    # Load the CSV file
    news_data = pd.read_csv(f'news_{ticker}.csv')

    # Load the model, tokenizer, and config
    MODEL = "mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis"
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
    news_data['Sentiment'] = news_data['Headline'].apply(lambda x: analyze_sentiment(x))

    # Define a function to extract sentiment scores into separate columns
    def extract_sentiment_scores(sentiment):
        score_dict = {label: 0.0 for label in ['positive', 'neutral', 'negative']}
        for label, score in sentiment:
            score_dict[label] = score
        return pd.Series(score_dict)

    # Apply the function to the sentiment column
    sentiment_scores = news_data['Sentiment'].apply(extract_sentiment_scores)

    # Concatenate the new sentiment score columns with the original dataframe
    news_data = pd.concat([news_data, sentiment_scores], axis=1)

    # Calculate the DistilRoBERTa Sentiment Score
    news_data['DistilRoberta Sentiment Score'] = news_data['positive'] - news_data['negative']

    # Save the DataFrame to a CSV file
    news_data.to_csv(f'news_{ticker}_distilroberta.csv', index=False)

    return news_data

# Define the ticker symbol
ticker = 'AAPL'
# ticker = 'AMZN'
# ticker = 'GOOG'
# ticker = 'MSFT'
news_data = analyze_news_sentiment_distilroberta(ticker)
print(news_data.head())