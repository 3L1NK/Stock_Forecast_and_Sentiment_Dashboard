import pandas as pd
from transformers import pipeline


def analyze_impact_on_stock_price_bart_large_mnli(ticker):
    """
        Analyzes the impact of news articles on stock prices using the BART Large MNLI model.
        The function loads news data (previously analyzed with Twitter RoBERTa), applies
        BART Large MNLI classification, calculates the impact probabilities on stock prices,
        and saves the results to a new CSV file.

        Parameters:
            ticker (str): Stock ticker symbol (e.g., 'AAPL' for Apple Inc.).

        Returns:
            pd.DataFrame: A DataFrame containing the original news data along with the classified
                          impact on stock price, sentiment, and calculated BART Large MNLI Scores.
    """
    # Load the CSV file containing news data with Twitter RoBERTa sentiment scores
    news_data = pd.read_csv(f'news_{ticker}_twitter_roberta.csv')

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
        Classifies a headline into one of the candidate labels using BART Large MNLI
        and returns the label with the highest probability along with the associated probabilities.

        Parameters:
            headline (str): The news headline to classify.
            classifier: The BART Large MNLI classification pipeline.
            candidate_labels (list): List of candidate labels for classification.

        Returns:
            tuple: A tuple containing the highest probability label and a list of probabilities for each label.
        """
        result = classifier(headline, candidate_labels)
        return result['labels'][0], result['scores']  # Return the label with the highest score and its probabilities

    def map_impact_on_stock_price(impact_on_stock_price):
        """
        Maps the detailed impact label from BART Large MNLI to a simplified form: Positive, Negative, or Neutral.

        Parameters:
            impact_on_stock_price (str): The detailed impact label from BART Large MNLI.

        Returns:
            str: A simplified impact label (Positive, Negative, Neutral) or 'Unknown Impact' if no match is found.
        """
        if 'positive impact' in impact_on_stock_price:
            return 'Positive'
        elif 'negative impact' in impact_on_stock_price:
            return 'Negative'
        elif 'no significant impact' in impact_on_stock_price:
            return 'Neutral'
        else:
            return 'Unknown Impact'

    # Create new columns for the sentiment, impact on stock price, and probabilities
    news_data['Sentiment'] = ''
    news_data['Impact on Stock Price'] = ''
    news_data['Positive Impact (Probability)'] = 0.0
    news_data['Negative Impact (Probability)'] = 0.0
    news_data['Neutral Impact (Probability)'] = 0.0

    # Iterate over each row in the DataFrame to classify the headline and determine the impact on stock price
    for index, row in news_data.iterrows():
        headline = row['Headline']

        # Determine sentiment based on the highest Twitter RoBERTa score
        sentiment_scores = {
            'Positive': row['positive'],
            'Negative': row['negative'],
            'Neutral': row['neutral']
        }
        sentiment = max(sentiment_scores, key=sentiment_scores.get)

        # Classify the impact on stock price using BART Large MNLI
        impact_on_stock_price, probabilities = classify_with_bart_large_mnli(headline, classifier, candidate_labels)
        shortened_impact_on_stock_price = map_impact_on_stock_price(impact_on_stock_price)

        # Save the results into the DataFrame
        news_data.at[index, 'Sentiment'] = sentiment
        news_data.at[index, 'Impact on Stock Price'] = shortened_impact_on_stock_price
        news_data.at[index, 'Positive Impact (Probability)'] = probabilities[0]
        news_data.at[index, 'Negative Impact (Probability)'] = probabilities[1]
        news_data.at[index, 'Neutral Impact (Probability)'] = probabilities[2]

    # Calculate the BART MNLI Score by subtracting the negative impact probabilities from the positive impact probabilities
    news_data['BART MNLI Score'] = news_data['Positive Impact (Probability)'] - news_data['Negative Impact (Probability)']

    # Save the updated DataFrame to a new CSV file
    news_data.to_csv(f'news_{ticker}_bart_mnli.csv', index=False)

    return news_data


# Set the stock ticker symbol
ticker = 'AAPL'
# ticker = 'AMZN'
# ticker = 'GOOG'
# ticker = 'MSFT'

# Analyze the impact of news articles on the stock price using BART Large MNLI
processed_data = analyze_impact_on_stock_price_bart_large_mnli(ticker)
