## Table of Contents

<!-- TOC -->

- [Market Sentiment Analysis Dashboard](#market-sentiment-analysis-dashboard)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Cloning Repository](#cloning-repository)
  - [Running the Dashboard (Local)](#running-the-dashboard-local)
  - [Accessing Live Website](#accessing-live-website)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Stock Data](#stock-data)
  - [Forecasting Modeling](#forecasting-modeling)
  - [Sentiment Analysis](#sentiment-analysis)
  - [Google Trends](#google-trends)

<!-- /TOC -->

# Market Sentiment Analysis Dashboard

The market sentiment analysis dashboard focuses on the stock market. It is designed for individuals with some prior knowledge and experience in the stock market who are interested in making informed investment decisions. This dashboard helps users analyze market sentiment and provides valuable insights to guide their investment choices in specific stocks.


# Features

- **Real-Time Stock Data**: Provides up-to-date stock information, including opening price, high/low, volume, market capitalization, and more.
- **Stock Price Prediction**: Utilizes multiple predictive models (Linear Regression, XGBoost, ARIMA, LSTM) to forecast future stock prices.
- **Google Trends Integration**: Allows users to analyze search trends for selected keywords, offering insights into market sentiment.
- **Sentiment Analysis**: 



# Installation
## Prerequisites
- Python 3.8+
- Python packages (install using `pip install -r requirements.txt`)

## Cloning Repository
By using this code you can clone the repository from Github.

``` git clone https://github.com/lanasolovej Applied_Predictive_Analytics.git ```

## Running the Dashboard (Local)

1. Make sure you have all the necessary packages installed.
2. Navigate to the project directory.
3. Run the following command:
   ```python app.py```
4. The dashboard will automatically open in your default web browser. If it doesn't, you can manually open it by navigating to http://127.0.0.1:8050/.

## Accessing Live Website
 waiting for news table 

# Project Structure
- `app.py`: The main file that initializes and runs the Dash app.
- `assets/styles.py`: Contains CSS styles for the dashboard components.
- `stock_pred/`: Directory containing predictive model implementations (Linear Regression, XGBoost, ARIMA, LSTM).
- `google_trends/`: Contains scripts for fetching and processing Google Trends data.
- `news_sentiment/`: 

# How It Works
## Stock Data
Real-time stock data is fetched using the **yfinance** library.
Stock indicators and charts are displayed, offering a snapshot of the stock's current performance.
## Forecasting Modeling
The dashboard integrates several state-of-the-art machine learning and statistical models, each tailored to provide insights into future stock movements:

- **Linear Regression**: A foundational model that establishes a clear and interpretable prediction curve for short to mid-term forecasting.
- **XGBoost (Extreme Gradient Boosting)**: XGBoost is a machine learning powerhouse that excels in prediction tasks by optimizing model accuracy through boosting.
- **ARIMA (AutoRegressive Integrated Moving Average)** : ARIMA is particularly effective for producing highly accurate forecasts in stable and cyclical market conditions.
- **LSTM (Long Short-Term Memory)**: LSTM model is finely tuned to capture and predict time-series data with long-range dependencies.
  
## Sentiment Analysis
The dashboard employs one sentiment analysis tool (VADER) and two Large Language Models (LLMs) (Twitter RoBERTa and DistilRoberta) to analyze the sentiment of news article headlines related to the four tickers featured on the dashboard. Additionally, BART Large MNLI is applied to determine whether a news headline suggests a positive, negative, or neutral impact on the company's stock price.

- **VADER**: VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media, while also applicable to texts from other domains.
- **Twitter RoBERTa**: Twitter RoBERTa is a transformer-based model fine-tuned specifically for sentiment analysis. It was trained on approximately 124 million sentiment-annotated tweets from January 2018 to December 2021.
- **DistilRoberta**: DistilRoberta is a distilled version of the RoBERTa-base model, which is on average twice as fast. It was trained on sentences from financial news categorized by sentiment, making it particularly suitable for sentiment analysis in financial contexts.
- **BART Large MNLI**: BART Large MNLI is a versatile model used for Natural Language Inference (NLI) tasks and zero-shot classification. This model classifies text into categories without requiring task-specific training by converting the input text into a premise and comparing it to user-provided labels as hypotheses. It calculates the probabilities of each hypothesis being an entailment, contradiction, or neutral in relation to the premise. The hypothesis with the highest probability of entailment is chosen as the most likely category.


## Google Trends
- Users can input up to four keywords to track their search trends on Google over time.
- The results are displayed in a line chart, providing insights into public interest in specific topics.
