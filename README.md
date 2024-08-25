## Table of Contents

<!-- TOC -->

- [Market Sentiment Dashboard](#market-sentiment-dashboard)
- [Features](#features)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [Cloning Repository](#cloning-repository)
  - [Running the Dashboard (Local)](#running-the-dashboard-local)
  - [Accessing Live Website](#accessing-live-website)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Stock Data](#stock-data)
  - [Google Trends](#google-trends)
  - [Stock Price Prediction](#stock-price-prediction)
  - [News](#news)
  - [Sentiment Analysis](#sentiment-analysis)
  - [BART Large MNLI: Impact on Stock Price](#bart-large-mnli-impact-on-stock-price)

<!-- /TOC -->

# Market Sentiment Dashboard

The Market Sentiment Dashboard focuses on the stock market. It is designed for individuals with some prior knowledge and experience in the stock market who are interested in making informed investment decisions. This dashboard helps users analyze market sentiment and provides valuable insights to guide their investment choices in specific stocks.

# Features

- **Real-Time Stock Data:** Provides up-to-date stock information, including opening price, high/low, volume, market capitalization, and more.
- **Google Trends Integration:** Allows users to analyze search trends for selected keywords, offering insights into market sentiment.
- **Stock Price Prediction:** Utilizes multiple predictive models (Linear Regression, XGBoost, ARIMA, LSTM) to forecast future stock prices.
- **News Integration:** Provides recent news articles related to the four stock tickers featured on the dashboard.
- **Sentiment Analysis:** Employs one sentiment analysis tool (VADER) and two Large Language Models (LLMs) (Twitter RoBERTa and DistilRoberta) to analyze the sentiment of news article headlines related to the featured stock tickers.
- **BART Large MNLI:** Applies BART Large MNLI to determine whether a news headline suggests a positive, negative, or neutral impact on a company's stock price.

# Installation

## Prerequisites

- Python 3.8+
- Python packages (install using `pip install -r requirements.txt`)

## Cloning Repository

Clone the repository from GitHub using the following command:

`git clone https://github.com/lanasolovej Applied_Predictive_Analytics.git`

## Running the Dashboard (Local)

1. Make sure you have all the necessary packages installed.
2. Navigate to the project directory.
3. Run the following command:
   `python app.py`
4. The dashboard will automatically open in your default web browser. If it doesn't, you can manually open it by navigating to http://127.0.0.1:8050/.

## Accessing Live Website

Click on this link to access the deployed website:
[Deployed Website](https://appliedpredictiveanalytics-b6rsdb5r2a-lm.a.run.app)

**Note:** After deployment, we noticed that the news table is not displaying correctly on the deployed website, although the data is being loaded. No errors related to the news sentiment table were observed in the Logs Explorer on the Google Cloud Platform.

# Project Structure

- `app.py` : The main file that initializes and runs the Dash app.
- `assets/` : Contains CSS styles for the dashboard components.
- `stock_pred/` : Directory containing predictive model implementations (Linear Regression, XGBoost, ARIMA, LSTM).
- `google_trends/` : Contains Python scripts for fetching and processing Google Trends data. It also stores CSV files with the fetched Google Trends data for preselected keywords for each ticker featured on the dashboard.
- `news_sentiment/` : Contains Python scripts for fetching and processing news data, applying sentiment analysis (VADER, Twitter RoBERTa, and DistilRoberta), and using BART Large MNLI. Additionally, it stores CSV files with the fetched news data for every featured ticker and the results from the applied models.

# How It Works

The Market Sentiment Dashboard is designed to provide insights for four different stock tickers:

- **AAPL**: Apple Inc.
- **AMZN**: Amazon.com Inc.
- **GOOG**: Alphabet Inc.
- **MSFT**: Microsoft Corporation

Users can select a ticker from the dropdown menu to view detailed information, including real-time stock data, Google Trends analysis, stock price predictions, news articles, sentiment analysis, and the impact of news headlines on stock prices, as analyzed by BART Large MNLI, all tailored specifically for the selected company.

## Stock Data

Real-time stock data is fetched using the **yfinance** library.
Stock indicators and charts are displayed, offering a snapshot of the stock's current performance.

## Google Trends

- Users can input up to four keywords to track their search trends on Google over time.
- The results are displayed in a line chart, providing insights into public interest in specific topics.

## Stock Price Prediction

The dashboard integrates several state-of-the-art machine learning and statistical models, each tailored to provide insights into future stock movements:

- **Linear Regression**: Linear Regression is a straightforward and interpretable statistical model that predicts stock prices by identifying a linear relationship between the target variable and one or more predictors. It's most effective for short to mid-term forecasting in stable market conditions.
- **XGBoost (Extreme Gradient Boosting)**: XGBoost is a powerful machine learning algorithm known for its efficiency and performance in predictive tasks. It works by sequentially improving weak learners, making it highly effective in capturing complex patterns in the data for accurate forecasts, but lacks in fluctuated trends.
- **ARIMA (AutoRegressive Integrated Moving Average)** : ARIMA is a traditional statistical model tailored for time-series forecasting. It excels in producing reliable predictions in stable and cyclical market environments by modeling the dependencies between observations.
- **LSTM (Long Short-Term Memory)**: LSTM networks are a type of recurrent neural network (RNN) designed to handle sequential data. They are particularly effective at capturing long-range dependencies in time-series data, making them ideal for predicting stock prices with more complex temporal patterns.

## News

News data is fetched in real-time from the **Finnhub Stock API** and displayed in the news table on the dashboard.

## Sentiment Analysis

The dashboard employs one sentiment analysis tool (VADER) and two Large Language Models (LLMs) (Twitter RoBERTa and DistilRoberta) to analyze the sentiment of news article headlines related to the four tickers featured on the dashboard. The sentiment analysis results are both displayed and visualized on the dashboard.

- **VADER**: VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media, while also applicable to texts from other domains.
- **Twitter RoBERTa**: Twitter RoBERTa is a transformer-based model fine-tuned specifically for sentiment analysis. It was trained on approximately 124 million sentiment-annotated tweets from January 2018 to December 2021.
- **DistilRoberta**: DistilRoberta is a distilled version of the RoBERTa-base model, which is on average twice as fast. It was trained on sentences from financial news categorized by sentiment, making it particularly suitable for sentiment analysis in financial contexts.

## BART Large MNLI: Impact on Stock Price

BART Large MNLI is applied to determine whether a news headline suggests a positive, negative, or neutral impact on the company's stock price. The results of BART Large MNLI are displayed on the dashboard.

- **BART Large MNLI**: BART Large MNLI is a versatile model used for Natural Language Inference (NLI) tasks and zero-shot classification. This model classifies text into categories without requiring task-specific training by converting the input text into a premise and comparing it to user-provided labels as hypotheses. It calculates the probabilities of each hypothesis being an entailment, contradiction, or neutral in relation to the premise. The hypothesis with the highest probability of entailment is chosen as the most likely category.

## Deployment

The application is deployed using Google Cloud Platform (GCP).
To deploy the application on GCP, a Docker container is used to package the application along with its dependencies. Docker ensures that the application runs consistently across different environments.
Firstly we have to create a new project which generates a unique project ID that is used for deployment.
Then we need to ensure that gunicorn is included in the requirements.txt file. Also, we have to add 'server = app.server' to the app.py file to enable it to work with Gunicorn. Once we have the provided id, we can run these following commands in the terminal:

- gcloud config set project <project-id>
- gcloud run deploy --source .

These two commands will start the deployment process. The deployment took roughly 30 minutes. Once it is complete, a url will be shown, which will redirect to the live website: [Deployed Website](https://appliedpredictiveanalytics-b6rsdb5r2a-lm.a.run.app)
