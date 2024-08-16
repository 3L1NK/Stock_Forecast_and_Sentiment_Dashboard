# import packages
from dash import Dash, html, dcc, callback, Output, Input, clientside_callback, State, ALL, ctx, dash_table
from dash.exceptions import PreventUpdate
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import dash
import dash_bootstrap_components as dbc
from dash_bootstrap_components._components.Container import Container
import yfinance as yf
import webbrowser
from datetime import datetime, timedelta

# styles
from assets.styles import NAVBAR_STYLE, LAYOUT_STYLE, INPUT_STYLE, STOCK_INFO_CONTAINER_STYLE, COMPANY_NAME_STYLE, TITLE_STYLE, TICKER_STYLE, STOCK_PRICE_STYLE, CURRENCY_STYLE, INDICATOR_CONTAINER_STYLE, PREDICTION_PLOT_CONTAINER_STYLE, INDICATOR_TERM_STYLE, INDICATOR_STYLE, NEWS_CONTAINER_STYLE, GTREND_CONTAINER_STYLE

# stock prediction models
from stock_pred.LSTM import LSTM_model
from stock_pred.XGboost import XGBoost_model
from stock_pred.LR import LR_model
from stock_pred.ARIMA import ARIMA_model

# google trends
from google_trends.main import get_trends_data, plot_from_csv

# news sentiment
from news_sentiment.news_sentiment_table import fetch_news_for_ticker, combine_sentiment_models
from news_sentiment.news_sentiment_plots import calculate_sentiment_moving_avg

# helper functions
def convert_to_large_format(number):
    if number >= 1_000_000_000_000:
        return f"{number / 1_000_000_000_000:.3f}T"
    elif number >= 1_000_000_000:
        return f"{number / 1_000_000_000:.2f}B"
    elif number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    else:
        return f"{number:,}"

# Stock Indicators
current_ticker = 'AAPL'
ticker_data = yf.Ticker(current_ticker)
history = ticker_data.history(period='1mo')
history.reset_index(inplace=True)
symbol = ticker_data.info.get('symbol')
current_price = ticker_data.info.get('currentPrice')
stock_price = round(current_price, 2) # Rounding the current price to two decimal points
open = ticker_data.info.get('open')
high = ticker_data.info.get('dayHigh')
low = ticker_data.info.get('dayLow')
volume = convert_to_large_format(ticker_data.info.get('volume'))
market_cap = convert_to_large_format(ticker_data.info.get('marketCap'))
trailing_pe = round(ticker_data.info.get('trailingPE'), 2)
avg_vol = convert_to_large_format(ticker_data.info.get('averageVolume'))
wh52 = ticker_data.info.get('fiftyTwoWeekHigh')
wl52 = ticker_data.info.get('fiftyTwoWeekLow')
company_name = ticker_data.info.get('shortName')

# initial stock movement graph
fig = px.line(history, x='Date', y='Close')

# initialize the app
app = dash.Dash(
  external_stylesheets=[
      dbc.themes.BOOTSTRAP,
      'https://cdnjs.cloudflare.com/ajax/libs/bootstrap-icons/1.8.1/font/bootstrap-icons.min.css'  # Include Bootstrap Icons
  ],
  suppress_callback_exceptions=True
)

# start of dashboard components
# ticker dropdown
ticker_input = dcc.Dropdown(
  id='ticker-dropdown',
  options=['AAPL', 'AMZN', 'GOOG', 'MSFT'],
  value='AAPL',
  style=INPUT_STYLE
)

# navigation bar
navbar = dbc.Navbar(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(ticker_input, width='auto'),
                dbc.Col(
                    html.H1('Market Sentiment Dashboard',
                            className='text-white text-center',
                            style={'fontSize': '34px', 'padding-left': '200px',}
                            ),
                    # width={"size": 6, "offset": 1},
                    width=8,
                    className='d-flex justify-content-center',
                ),
                dbc.Col(
                    width='auto',  # Empty column to push the button to the right
                    className='flex-grow-1',
                ),
                dbc.Col(
                    dcc.Link(
                        html.Button([
                            html.I(className='bi bi-house-fill', style={'color': '#444444'}),  # Bootstrap Icon for house
                            html.Span(' Home', style={'color': '#444444'})  # Text color
                        ], className='btn btn-primary', style={'backgroundColor': '#f8f4f4', 'borderColor': '#f8f4f4'}),
                        href='/',
                        style={'textDecoration': 'none'}
                    ),
                    width='auto',
                    className='ml-auto'
                ),
            ],
            align='center',
            className='w-100',
        ),
        fluid=True
    ),
    color='dark',
    dark=True,
    class_name='navbar',
    style=NAVBAR_STYLE
)

# stock indicators container
stock_indicators = dbc.Container(
  id='stock-indicator',
  style=INDICATOR_CONTAINER_STYLE,
  children=[
     dbc.Row(
      children=[
        dbc.Col(
          children=[
            dbc.Row(
              children=[
                dbc.Col(html.P(children='Open', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-open', children=open, style=INDICATOR_STYLE), width=5)
              ]),
            dbc.Row(
              children=[
                dbc.Col(html.P(children='High', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-high', children=high, style=INDICATOR_STYLE), width=5)
              ]),
            dbc.Row(
              children=[
                dbc.Col(html.P(children='Low', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-low', children=low, style=INDICATOR_STYLE), width=5)
              ]),
          ]
        ),
        dbc.Col(
          children=[
            dbc.Row(
              children=[
                dbc.Col(html.P(children='Volume', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-vol', children=volume, style=INDICATOR_STYLE), width=5)
              ]),
            dbc.Row(
              children=[
                dbc.Col(html.P(children='Mkt. Cap', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-mktcap', children=market_cap, style=INDICATOR_STYLE), width=5)
              ]),
            dbc.Row(
              children=[
                dbc.Col(html.P(children='P/E', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-pe', children=trailing_pe, style=INDICATOR_STYLE), width=5)
              ])
          ]
        ),
        dbc.Col(
          children=[
            dbc.Row(
              children=[
                dbc.Col(html.P(children='Avg. Vol.', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-avgvol', children=avg_vol, style=INDICATOR_STYLE), width=5)
              ]),
            dbc.Row(
              children=[
                dbc.Col(html.P(children='52WH', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-52wh', children=wh52, style=INDICATOR_STYLE), width=5)
              ]),
            dbc.Row(
              children=[
                dbc.Col(html.P(children='52WL', className='indicators', style=INDICATOR_TERM_STYLE), width=6),
                dbc.Col(html.P(id='indicator-52wl', children=wl52, style=INDICATOR_STYLE), width=5)
              ])
          ]
        )
      ]
     )
  ]
)

# stock movement graph containers
stock_graph = dbc.Container(
  id='stock-info-container',
  style=STOCK_INFO_CONTAINER_STYLE,
  children=[
    dbc.Row(
      children=[
        dbc.Col(html.H1(id='ticker-name', children=symbol, style=TICKER_STYLE), width=3),
        dbc.Col(html.H4(id='company-name', children=company_name, style=COMPANY_NAME_STYLE), width=5),
        dbc.Col(html.H2(id='stock-price', children=stock_price, style=STOCK_PRICE_STYLE), width=2),
        dbc.Col(html.H4(id='currency', children='USD', style=CURRENCY_STYLE), width=1),
      ]
    ),
    dcc.Graph(id='stock-graph', figure=fig)
  ]
)

stock_pred_lr = dbc.Container(
    id='plot-container-lr',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-lr'),
        dcc.Graph(id='pred-graph-lr')
    ])

stock_pred_lstm = dbc.Container(
    id='plot-container-lstm',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-lstm'),
        dcc.Graph(id='pred-graph-lstm')
    ])

stock_pred_arima = dbc.Container(
    id='plot-container-arima',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-arima'),
        dcc.Graph(id='pred-graph-arima')
    ])

stock_pred_xgboost = dbc.Container(
    id='plot-container-xgboost',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-xgboost'),
        dcc.Graph(id='pred-graph-xgboost')
    ])

prediction_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(stock_pred_lr, label='Linear Regression', tab_id='lr', label_style={'color': '#444444'}),
                dbc.Tab(stock_pred_xgboost, label='XGBoost', tab_id='xgboost', label_style={'color': '#444444'}),
                dbc.Tab(stock_pred_arima, label='ARIMA', tab_id='arima', label_style={'color': '#444444'}),
                dbc.Tab(stock_pred_lstm, label='LSTM', tab_id='lstm', label_style={'color': '#444444'}),
            ],
            id='tabs',
            active_tab='lr',
            style={'backgroundColor': '#F5F5F5'}
        ),
    ],
    style={'backgroundColor': 'white'}
)

# Info button Prediction
info_button_prediction = dbc.Button(
     html.I(className='bi bi-info-lg'), # Bootstrap icon class
     id='info-button-prediction',
     outline=True,
     className='custom-info-button ml-auto' # Apply the custom class for hover effects
)

# Modal for displaying model information
info_modal_prediction = dbc.Modal(
    [
        dbc.ModalHeader('Model Explanations', style={'font-size': '20px', 'font-weight': 'bold'}),
        dbc.ModalBody(
            [
                html.H6('Linear Regression', style={'font-weight': 'bold'}),
                html.P("Linear Regression is a straightforward and interpretable statistical model that predicts stock prices by identifying a linear relationship between the target variable and one or more predictors. It's most effective for short to mid-term forecasting in stable market conditions."),
                html.H6('XGBoost', style={'font-weight': 'bold'}),
                html.P('XGBoost (Extreme Gradient Boosting) is a powerful machine learning algorithm known for its efficiency and performance in predictive tasks. It works by sequentially improving weak learners, making it highly effective in capturing complex patterns in the data for accurate forecasts, but lacks in fluctuated trends.'),
                html.H6('ARIMA', style={'font-weight': 'bold'}),
                html.P('ARIMA (AutoRegressive Integrated Moving Average) is a traditional statistical model tailored for time-series forecasting. It excels in producing reliable predictions in stable and cyclical market environments by modeling the dependencies between observations.'),
                html.H6('LSTM', style={'font-weight': 'bold'}),
                html.P('LSTM (Long Short-Term Memory) networks are a type of recurrent neural network (RNN) designed to handle sequential data. They are particularly effective at capturing long-range dependencies in time-series data, making them ideal for predicting stock prices with more complex temporal patterns.'),
            ]
        ),
    ],
    id='info-modal-prediction',
    centered=True,
    is_open=False,  # Default state is closed
)

prediction_plot_container = dbc.Container(
  id='plot-container',
  style=PREDICTION_PLOT_CONTAINER_STYLE,
  children=[
    dbc.Row(
    children=[
      dbc.Col(html.H1(id='plot-prediction', children='Prediction', style=TITLE_STYLE), width=11),
      dbc.Col(info_button_prediction, width=1),
    ]
    ),
      dbc.Col(children=[
          dcc.Loading(
              id='loading-prediction-plot',
              overlay_style={'visibility': 'visible'},
              delay_show=400,
              color='#444444',
              children=[prediction_tabs]
          )
      ]),
    info_modal_prediction
  ]
)

# google trend keyword input 
keyword_input = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(dbc.Input(id='keyword1', type='text', placeholder='Keyword 1'), width=3),
                dbc.Col(dbc.Input(id='keyword2', type='text', placeholder='Keyword 2'), width=3),
                dbc.Col(dbc.Input(id='keyword3', type='text', placeholder='Keyword 3'), width=3),
                dbc.Col(dbc.Input(id='keyword4', type='text', placeholder='Keyword 4'), width=3),
            ],
            style={'marginTop': '20px'}
        ),
        dbc.Row(
           children=[
              dbc.Button('Submit',
                   id='submit-keywords',
                   style={'backgroundColor': '#444444', 'color': 'white', 'border': 'none'},
                   className='mt-2',
                   )
           ],
           style={'marginTop':'12px', 'marginBottom': '20px'}
        )
    ],
    fluid=True
)

# google trend container 
google_trend_container = dbc.Container(
  id='google-trend-container',
  style=GTREND_CONTAINER_STYLE,
  children=[
    html.H1('Google Trends', style=TITLE_STYLE),
    keyword_input,
    dcc.Loading(
            id='loading-google-trend',
            type='default',
            color='#444444',  
            children=dcc.Graph(id='google-trend-plot', className='mt-2', style={'height': '495px'})
        )
  ],
  fluid=True
)

# News Table
news_table = dash_table.DataTable(
    id='news-table',
    columns=[
        {'name': 'Date', 'id': 'Date'},
        # {'name': 'Time', 'id': 'Time'},
        {'name': 'Headline', 'id': 'Headline', 'presentation': 'markdown'},
        {'name': 'VADER Sentiment Score', 'id': 'VADER Sentiment Score'},
        {'name': 'Twitter RoBERTa Sentiment Score', 'id': 'Twitter RoBERTa Sentiment Score'},
        {'name': 'DistilRoberta Sentiment Score', 'id': 'DistilRoberta Sentiment Score'},
        {'name': 'BART MNLI: Impact on Stock Price', 'id': 'BART MNLI: Impact on Stock Price'},
    ],
    style_table={'overflowX': 'auto'},
    style_cell={
        'whiteSpace': 'normal',
        'height': 'auto',
        'textAlign': 'center',
        'paddingLeft': '10px',
        'paddingRight': '10px',
        'fontFamily': 'Arial, sans-serif',
    },
    style_cell_conditional=[
        {'if': {'column_id': 'Date'}, 'width': '80px'},
        {'if': {'column_id': 'VADER Sentiment Score'}, 'width': '120px'},
        {'if': {'column_id': 'Twitter RoBERTa Sentiment Score'}, 'width': '120px'},
        {'if': {'column_id': 'DistilRoberta Sentiment Score'}, 'width': '120px'},
        {'if': {'column_id': 'BART MNLI: Impact on Stock Price'}, 'width': '120px'},
        {'if': {'column_id': 'Headline'}, 'width': '500px'},
    ],
    style_header={
        'fontWeight': 'bold',  # Makes the column headers bold
        'textAlign': 'center',  # Aligns the text to the left
       # 'borderBottom': '2px solid black',  # Adds a thicker, darker line below the header
    },
    style_data_conditional=[
        {
            'if': {'row_index': 'odd'},
            'backgroundColor': 'rgb(248, 248, 248)'  # Light grey background for odd rows (striped effect)
        }
    ],
)

news_sentiment_plot_vader = dbc.Container(
    id='plot-container-vader',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-vader'),
        dcc.Graph(id='news-vader-plot'),
    ])

news_sentiment_plot_twitter_roberta = dbc.Container(
    id='plot-container-twitter_roberta',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-twitter-roberta'),
        dcc.Graph(id='news-twitter-roberta-plot'),
    ])

news_sentiment_plot_distilroberta = dbc.Container(
    id='plot-container-distilroberta',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-distilroberta'),
        dcc.Graph(id='news-distilroberta-plot'),
    ])

news_sentiment_plot_bart_mnli = dbc.Container(
    id='plot-container-bart-mnli',
    style={'padding': '0px'},
    children=[
        dcc.Store(id='data-store-bart-mnli'),
        dcc.Graph(id='news-bart-mnli-plot'),
    ])

news_sentiment_tabs = html.Div(
    [
        dbc.Tabs(
            [
                dbc.Tab(news_sentiment_plot_vader, label='VADER Lexicon', tab_id='vader', label_style={'color': '#444444'}),
                dbc.Tab(news_sentiment_plot_twitter_roberta, label='Twitter RoBERTa', tab_id='twitter_roberta', label_style={'color': '#444444'}),
                dbc.Tab(news_sentiment_plot_distilroberta, label='DistilRoberta', tab_id='distilroberta', label_style={'color': '#444444'}),
                dbc.Tab(news_sentiment_plot_bart_mnli, label='BART Large MNLI', tab_id='bart_mnli', label_style={'color': '#444444'}),
            ],
            id='news_sentiment_tabs',
            active_tab='vader',
            style={'backgroundColor': '#F5F5F5'}
        ),
    ],
    style={'backgroundColor': 'white'}
)

# Info button News
info_button_news_table = dbc.Button(
     html.I(className='bi bi-info-lg'), # Bootstrap icon class
     id='info-button-news-table',
     outline=True,
     className='custom-info-button ml-auto' # Apply the custom class for hover effects
)

# Modal for displaying model information
info_modal_news_table = dbc.Modal(
    [
        dbc.ModalHeader('Model Explanations', style={'font-size': '20px', 'font-weight': 'bold'}),
        dbc.ModalBody(
            [
                html.P('Sentiment analysis is a technique used to evaluate large volumes of text to determine whether the expressed sentiment is positive, negative, or neutral. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), sentiment analysis interprets text in a manner similar to human understanding.'),
                html.P('The Market Sentiment Dashboard employs one sentiment analysis tool (VADER) and two Large Language Models (LLMs) (Twitter RoBERTa and DistilRoberta) to analyze the sentiment of news article headlines associated with the four tickers featured on the dashboard.'),
                html.H6('Sentiment Scores', style={'font-weight': 'bold'}),
                html.P('The sentiment scores generated by VADER, Twitter RoBERTa, and DistilRoberta reflect the overall sentiment of a news article headline, with values ranging from -1 to 1:', style={'margin-bottom': '6px'}),
                html.P('Strongly positive sentiment: Score closer to 1', style={'margin-bottom': '2px'}),
                html.P('Strongly negative sentiment: Score closer to -1', style={'margin-bottom': '2px'}),
                html.P('Neutral sentiment: Score around 0'),
                html.H5('Model Descriptions', style={'font-weight': 'bold'}),
                html.H6('VADER', style={'font-weight': 'bold'}),
                html.P('VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media, while also applicable to texts from other domains.'),
                html.H6('Twitter RoBERTa', style={'font-weight': 'bold'}),
                html.P('Twitter RoBERTa is a transformer-based model fine-tuned specifically for sentiment analysis. It was trained on approximately 124 million sentiment-annotated tweets from January 2018 to December 2021.'),
                html.H6('DistilRoberta', style={'font-weight': 'bold'}),
                html.P('DistilRoberta is a distilled version of the RoBERTa-base model, which is on average twice as fast. It was trained on sentences from financial news categorized by sentiment, making it particularly suitable for sentiment analysis in financial contexts.'),
                html.H6('BART Large MNLI', style={'font-weight': 'bold'}),
                html.P('BART Large MNLI is a versatile model used for Natural Language Inference (NLI) tasks and zero-shot classification. This model classifies text into categories without requiring task-specific training by converting the input text into a premise and comparing it to user-provided labels as hypotheses. It calculates the probabilities of each hypothesis being an entailment, contradiction, or neutral in relation to the premise. The hypothesis with the highest probability of entailment is chosen as the most likely category.'),
                html.P("The Market Sentiment Dashboard uses this model to determine whether a news headline suggests a positive, negative, or neutral impact on a company's stock price."),
            ]
        ),
    ],
    id='info-modal-news-table',
    centered=True,
    is_open=False,  # Default state is closed
)

# Info button News
info_button_news_plots = dbc.Button(
     html.I(className='bi bi-info-lg'), # Bootstrap icon class
     id='info-button-news-plots',
     outline=True,
     className='custom-info-button ml-auto' # Apply the custom class for hover effects
)

# Modal for displaying model information
info_modal_news_plots = dbc.Modal(
    [
        dbc.ModalHeader('Model Explanations', style={'font-size': '20px', 'font-weight': 'bold'}),
        dbc.ModalBody(
            [
                html.P('Sentiment analysis is a technique used to evaluate large volumes of text to determine whether the expressed sentiment is positive, negative, or neutral. By leveraging Natural Language Processing (NLP) and Machine Learning (ML), sentiment analysis interprets text in a manner similar to human understanding.'),
                html.P('The Market Sentiment Dashboard employs one sentiment analysis tool (VADER) and two Large Language Models (LLMs) (Twitter RoBERTa and DistilRoberta) to analyze the sentiment of news article headlines associated with the four tickers featured on the dashboard.'),
                html.H6('Sentiment Scores', style={'font-weight': 'bold'}),
                html.P('The sentiment scores generated by VADER, Twitter RoBERTa, and DistilRoberta reflect the overall sentiment of a news article headline, with values ranging from -1 to 1:', style={'margin-bottom': '6px'}),
                html.P('Strongly positive sentiment: Score closer to 1', style={'margin-bottom': '2px'}),
                html.P('Strongly negative sentiment: Score closer to -1', style={'margin-bottom': '2px'}),
                html.P('Neutral sentiment: Score around 0'),
                html.H5('Model Descriptions', style={'font-weight': 'bold'}),
                html.H6('VADER', style={'font-weight': 'bold'}),
                html.P('VADER (Valence Aware Dictionary and sEntiment Reasoner) is a lexicon and rule-based sentiment analysis tool specifically attuned to sentiments expressed in social media, while also applicable to texts from other domains.'),
                html.H6('Twitter RoBERTa', style={'font-weight': 'bold'}),
                html.P('Twitter RoBERTa is a transformer-based model fine-tuned specifically for sentiment analysis. It was trained on approximately 124 million sentiment-annotated tweets from January 2018 to December 2021.'),
                html.H6('DistilRoberta', style={'font-weight': 'bold'}),
                html.P('DistilRoberta is a distilled version of the RoBERTa-base model, which is on average twice as fast. It was trained on sentences from financial news categorized by sentiment, making it particularly suitable for sentiment analysis in financial contexts.'),
                html.H6('BART Large MNLI', style={'font-weight': 'bold'}),
                html.P('BART Large MNLI is a versatile model used for Natural Language Inference (NLI) tasks and zero-shot classification. This model classifies text into categories without requiring task-specific training by converting the input text into a premise and comparing it to user-provided labels as hypotheses. It calculates the probabilities of each hypothesis being an entailment, contradiction, or neutral in relation to the premise. The hypothesis with the highest probability of entailment is chosen as the most likely category.'),
                html.P("The Market Sentiment Dashboard uses this model to determine whether a news headline suggests a positive, negative, or neutral impact on a company's stock price."),
            ]
        ),
    ],
    id='info-modal-news-plots',
    centered=True,
    is_open=False,  # Default state is closed
)

news_container = dbc.Container(
  id='news-container',
  style=NEWS_CONTAINER_STYLE,
  children=[
    dbc.Row(
      children=[
        dbc.Col(html.H1(id='news', children='News', style=TITLE_STYLE), width=11),
        dbc.Col(info_button_news_table, width=1),
        dcc.Loading(
            id='loading-news-table',
            #overlay_style={'visibility': 'visible'},
            #style={'position': 'relative', 'zIndex': 0},  # Ensure table headers stay on top
            #parent_style={'zIndex': 1},  # Ensure spinner doesn't cover headers
            delay_show=400,
            color='#444444',
            children=[news_table]
        )
      ]
    ),
    dbc.Row(
      children=[
        dbc.Col(html.H2('Sentiment Analysis Visualization'), style={'margin-top': '20px'}, width=11),
        dbc.Col(info_button_news_plots, style={'margin-top': '20px'}, width=1,),
      ]
    ),
    dbc.Col(children=[
        dcc.Loading(
            id='loading-sentiment-plot',
            overlay_style={'visibility': 'visible'},
            delay_show=400,
            color='#444444',
            children=[news_sentiment_tabs]
            )
    ]),
    info_modal_news_table,
    info_modal_news_plots
  ]
)

# Content for Impressum page
impressum_content = html.Div([
    html.H4('Impressum'),
    html.P('Here goes the Impressum content...', style={'margin-top': '20px'})
], style={'padding-left': '60px', 'margin-top': '40px'})

# Content for Datenschutzerklärung page
datenschutz_content = html.Div([
    html.H4('Datenschutzerklärung'),
    html.P('Here goes the Datenschutzerklärung content...', style={'margin-top': '20px'})
], style={'padding-left': '60px', 'margin-top': '40px'})

footer = html.Footer(
    dbc.Container(
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H5('CONTACT', style={'marginBottom': '18px'}),
                        html.P('Humboldt-Universität zu Berlin', style={'margin': '0', 'line-height': '1.5'}),
                        html.P('School of Business and Economics', style={'margin': '0', 'line-height': '1.5'}),
                        html.P('Spandauer Straße 1,', style={'margin': '0', 'line-height': '1.5'}),
                        html.P('10178 Berlin,', style={'margin': '0', 'line-height': '1.5'}),
                        html.P('Deutschland'),
                        html.P('Tel.: +49 30 2093-99543'),
                        html.P('E-Mail: @hu-berlin.de'),
                    ],
                    width=4
                ),
                dbc.Col(
                    [
                        html.H5('ABOUT US', style={'marginBottom': '18px'}),
                        html.P('Welcome to the Market Sentiment Dashboard, your go-to resource for insights and analysis on the stock market. Designed for informed investors and those with a keen interest in stock trading, our dashboard provides the essential tools and information needed to make well-informed investment decisions. Stay ahead of the market trends and optimize your investment strategies with our expert-driven data and analysis.'),
                    ],
                    width=4
                ),
                dbc.Col(
                    [
                        html.H5('LEGAL', style={'marginBottom': '18px'}),
                        dcc.Link('Datenschutzerklärung', href='/datenschutzerklaerung',
                                 style={'color': 'white', 'display': 'block', 'marginBottom': '10px'}),
                        dcc.Link('Impressum', href='/impressum', style={'color': 'white', 'display': 'block'})
                    ],
                    width=4
                ),
            ]
        ),
        style={'paddingTop': '18px'}
    ),
    style={'textAlign': 'center', 'padding': '10px', 'backgroundColor': '#212529', 'color': 'white'}
)


# Define the main content of the dashboard
main_content = dbc.Container(
    [
        dbc.Row(
            children=[
                dbc.Col(children=[stock_graph, stock_indicators], width=6),
                dbc.Col(children=[google_trend_container], width=6),
                dbc.Col(children=[prediction_plot_container], width=20),
                dbc.Col(children=[news_container], width=20),
            ]
        ),
    ],
    fluid=True,
)

# App layout
app.layout = dbc.Container(
    [
        dcc.Location(id='url', refresh=False),
        navbar,
        html.Div(id='page-content'),
        footer
    ],
    fluid=True,
    style=LAYOUT_STYLE
)


### Callbacks

@app.callback([
    Output('ticker-name', 'children'),
    Output('company-name', 'children'),
    Output('stock-price', 'children'),
    Output('indicator-open', 'children'),
    Output('indicator-high', 'children'),
    Output('indicator-low', 'children'),
    Output('indicator-vol', 'children'),
    Output('indicator-mktcap', 'children'),
    Output('indicator-pe', 'children'),
    Output('indicator-avgvol', 'children'),
    Output('indicator-52wh', 'children'),
    Output('indicator-52wl', 'children'),
    Output('stock-graph', 'figure')],
    [Input('ticker-dropdown', 'value')])
def update_stock_info(ticker):
    print("update_stock_info() called, selected ticker:", ticker)
    ticker_data = yf.Ticker(ticker)
    history = ticker_data.history(period='1mo')
    history.reset_index(inplace=True)

    symbol = ticker
    current_price = ticker_data.info.get('currentPrice')
    stock_price = round(current_price, 2)  # Rounding the current price to two decimal points
    company_name = ticker_data.info.get('shortName')
    open = ticker_data.info.get('open')
    high = ticker_data.info.get('dayHigh')
    low = ticker_data.info.get('dayLow')
    volume = convert_to_large_format(ticker_data.info.get('volume'))
    market_cap = convert_to_large_format(ticker_data.info.get('marketCap'))
    trailing_pe = round(ticker_data.info.get('trailingPE'), 2)
    avg_vol = convert_to_large_format(ticker_data.info.get('averageVolume'))
    wh52 = ticker_data.info.get('fiftyTwoWeekHigh')
    wl52 = ticker_data.info.get('fiftyTwoWeekLow')

    fig = go.Figure(data=[go.Candlestick(
        x=history['Date'],
        open=history['Open'],
        high=history['High'],
        low=history['Low'],
        close=history['Close']
    )])

    fig.update_layout(
        title=f'Stock Price Movement for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False
    )

    return symbol, company_name, stock_price, open, high, low, volume, market_cap, trailing_pe, avg_vol, wh52, wl52, fig


# Callbacks for modal open/close behavior - info button prediction section
@app.callback(
    Output('info-modal-prediction', 'is_open'),
    [Input('info-button-prediction', 'n_clicks')], #Input('close-info-modal-prediction', 'n_clicks')],
    [State('info-modal-prediction', 'is_open')],
)
def toggle_info_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

# Callbacks to load data and update plots
@app.callback(
    Output('data-store-lr', 'data'),
    [Input('tabs', 'active_tab'), Input('ticker-dropdown', 'value')],
    State('data-store-lr', 'data')
)
def load_prediction_LR(active_tab, ticker, data):
    if active_tab == 'lr' and ticker:
      if data is None or data.get('ticker') != ticker:
        current_date = datetime.today().strftime('%Y-%m-%d')
        actual_prices_df, predicted_train_prices_df, predicted_test_prices_df, future_forecasted_prices_df = LR_model(ticker, end=current_date)
        return {
            'ticker': ticker,
            'actual_prices_df': actual_prices_df.to_dict('records'),
            'predicted_train_prices_df': predicted_train_prices_df.to_dict('records'),
            'predicted_test_prices_df': predicted_test_prices_df.to_dict('records'),
            'future_forecasted_prices_df': future_forecasted_prices_df.to_dict('records')
        }
    return data

@app.callback(
    Output('data-store-lstm', 'data'),
    [Input('tabs', 'active_tab'), Input('ticker-dropdown', 'value')],
    State('data-store-lstm', 'data')
)
def load_stock_prediction_LSTM(active_tab, ticker, data):
    if active_tab == 'lstm' and ticker:
      if data is None or data.get('ticker') != ticker:
        current_date = datetime.today().strftime('%Y-%m-%d')
        result_df, future_df = LSTM_model(ticker, end=current_date)
        return  {
            'ticker' : ticker,
            'result_df': result_df.to_dict(),
            'future_df' : future_df.to_dict()
        }
    return data

@app.callback(
    Output('data-store-arima', 'data'),
    [Input('tabs', 'active_tab'), Input('ticker-dropdown', 'value')],
    State('data-store-arima', 'data')
)
def load_stock_prediction_ARIMA(active_tab, ticker, data):
    if active_tab == 'arima' and ticker:
      if data is None or data.get('ticker') != ticker:
        current_date = datetime.today().strftime('%Y-%m-%d')
        historical_data, forecast, conf_int = ARIMA_model(ticker, end=current_date)
        return {
            'ticker' : ticker,
            'historical_data': historical_data.tolist(),
            'historical_data_index': historical_data.index.tolist(),
            'forecast': forecast.tolist(),
            'conf_int': conf_int.tolist()
        }
    return data

@app.callback(
    Output('data-store-xgboost', 'data'),
    [Input('tabs', 'active_tab'), Input('ticker-dropdown', 'value')],
    State('data-store-xgboost', 'data')
)
def load_stock_prediction_XGboost(active_tab, ticker, data):
    if active_tab == 'xgboost' and ticker:
      if data is None or data.get('ticker') != ticker:
        current_date = datetime.today().strftime('%Y-%m-%d')
        result_df, future_df = XGBoost_model(ticker, end=current_date)
        return  {
              'ticker' : ticker,
              'result_df': result_df.to_dict(),
              'future_df' : future_df.to_dict()
          }
    return data

# Callbacks to update the graphs based on stored data
@app.callback(
    Output('pred-graph-lr', 'figure'),
    [Input('data-store-lr', 'data'),
    Input('ticker-dropdown', 'value')]
)
def update_graph_lr(data, ticker):
    if data is None:
        raise PreventUpdate
    
    actual_prices_df = pd.DataFrame(data['actual_prices_df'])
    predicted_train_prices_df = pd.DataFrame(data['predicted_train_prices_df'])
    predicted_test_prices_df = pd.DataFrame(data['predicted_test_prices_df'])
    future_forecasted_prices_df = pd.DataFrame(data['future_forecasted_prices_df'])

    # Merge the dataframes to form a single dataframe for plotting
    data = actual_prices_df.merge(predicted_train_prices_df, on='Date', how='left').merge(predicted_test_prices_df, on='Date', how='left').merge(future_forecasted_prices_df, on='Date', how='outer')

    # Define the initial range for the last 2 years
    last_date = pd.to_datetime(data['Date']).max()
    two_years_ago = last_date - pd.DateOffset(years=2)

    fig = go.Figure()

    # Add actual price line
    fig.add_trace(go.Scatter(
        x=data['Date'],
        y=data['Actual Price'],
        mode='lines',
        name='Actual Price',
        line=dict(color='blue')
    ))

    # Add predicted train price line in green
    fig.add_trace(go.Scatter(
        x=predicted_train_prices_df['Date'],
        y=predicted_train_prices_df['Predicted Price'],
        mode='lines',
        name='Predicted Train Price',
        line=dict(color='green')
    ))

    # Add predicted test price line in yellow
    fig.add_trace(go.Scatter(
        x=predicted_test_prices_df['Date'],
        y=predicted_test_prices_df['Predicted Price'],
        mode='lines',
        name='Predicted Test Price',
        line=dict(color='orange')
    ))

    # Add forecasted price line in red
    fig.add_trace(go.Scatter(
        x=future_forecasted_prices_df['Date'],
        y=future_forecasted_prices_df['Forecasted Price'],
        mode='lines',
        name='Forecast Price',
        line=dict(color='red')
    ))

    fig.update_layout(
        title=f'Stock Prediction & Forecast for {ticker} using Linear Regression',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        legend_title='Legend',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5
        )
    )

    fig.update_xaxes(
        range=[two_years_ago, last_date],
        rangeslider_visible=True  # Adds a range slider for zooming out
    )

    fig.add_shape(
        type="rect",
        x0=future_forecasted_prices_df['Date'].iloc[0],
        x1=future_forecasted_prices_df['Date'].iloc[-1],
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        fillcolor='LightSkyBlue',
        opacity=0.3,
        layer='below',
        line_width=0,
    )
    return fig

@app.callback(
    Output('pred-graph-lstm', 'figure'),
    [Input('data-store-lstm', 'data'),
    Input('ticker-dropdown', 'value')]
)
def update_graph_lstm(data, ticker):
  if data is None:
    raise PreventUpdate

  fig = go.Figure()
  result_df = pd.DataFrame(data['result_df'])
  future_df = pd.DataFrame(data['future_df'])

  result_df['Date'] = pd.to_datetime(result_df['Date'])
  future_df['Date'] = pd.to_datetime(future_df['Date'])

  fig.add_trace(go.Scatter(
      x=result_df['Date'],
      y=result_df['Actual Price'],
      mode='lines',
      name='Actual Price',
      line=dict(color='blue')
  ))

  # Add training predicted price line
  fig.add_trace(go.Scatter(
      x=result_df['Date'],
      y=result_df['Training Predicted Price'],
      mode='lines',
      name='Training Predicted Price',
      line=dict(color='green')
  ))

  # Add testing predicted price line
  fig.add_trace(go.Scatter(
      x=result_df['Date'],
      y=result_df['Testing Predicted Price'],
      mode='lines',
      name='Testing Predicted Price',
      line=dict(color='orange')
  ))

  # Add forecasted price line
  fig.add_trace(go.Scatter(
      x=future_df['Date'],
      y=future_df['Forecasted Price'],
      mode='lines',
      name='Forecasted Price',
      line=dict(color='red')
  ))

  # Add a light blue shaded area for the forecast period
  fig.add_shape(
      type="rect",
      x0=future_df['Date'].iloc[0],
      x1=future_df['Date'].iloc[-1],
      y0=0,
      y1=1,
      xref='x',
      yref='paper',
      fillcolor='LightSkyBlue',
      opacity=0.3,
      layer='below',
      line_width=0,
  )

  # Update layout
  fig.update_layout(
      title=f'Stock Price Prediction & Forecast for {ticker} using LSTM',
      xaxis_title='Date',
      yaxis_title='Price',
      xaxis_showgrid=False,
      yaxis_showgrid=False,
      legend_title='Legend',
      legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5
        )
  )

  # Set the range to include both the historical and forecasted dates
  two_years_ago = result_df['Date'].max() - pd.DateOffset(years=2)
  fig.update_xaxes(
      range=[two_years_ago, future_df['Date'].max()],
      rangeslider_visible=True
  )
  return fig

@app.callback(
    Output('pred-graph-arima', 'figure'),
    [Input('data-store-arima', 'data'),
    Input('ticker-dropdown', 'value')]
)
def update_graph_arima(data, ticker):
    if data is None:
        raise PreventUpdate
    
    historical_data = pd.Series(data['historical_data'], index=pd.to_datetime(data['historical_data_index']))
    forecast = data['forecast']
    conf_int = np.array(data['conf_int'])

    last_date = pd.to_datetime(historical_data.index.union(pd.date_range(start=historical_data.index[-1], periods=91, freq='B'))).max()
    two_years_ago = last_date - pd.DateOffset(years=2)

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=historical_data.index, y=historical_data, mode='lines', name='Actual Data', line=dict(color='blue')))
    future_dates = pd.date_range(historical_data.index[-1] + pd.Timedelta(days=1), periods=90, freq='B')
    fig.add_trace(go.Scatter(x=future_dates, y=forecast, mode='lines', name='Forecast', line=dict(color='red')))
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int[:, 0],
        fill=None,
        mode='lines',
        line=dict(color='pink', width=0),
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=conf_int[:, 1],
        fill='tonexty',  # fill area between trace0 and trace1
        mode='lines', line=dict(color='pink'),
        name='Confidence Interval'
    ))

    # Update layout
    fig.update_layout(
        title=f'Stock Price Prediction & Forecast for {ticker} using ARIMA',
        xaxis_title='Date',
        yaxis_title=' Stock Price',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        legend_title='Legend',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5
        ))
    
    fig.update_xaxes(
        range=[two_years_ago, last_date],
        rangeslider_visible=True  # Adds a range slider for zooming out
    )
    
    # Add a light blue shaded area for the forecast period
    fig.add_shape(
        type='rect',
        x0=future_dates[0],
        x1=future_dates[-1],
        y0=0,
        y1=1,
        xref='x',
        yref='paper',
        fillcolor='LightSkyBlue',
        opacity=0.3,
        layer='below',
        line_width=0,
    )

    # Update layout
    fig.update_layout(
        title=f'Stock Price Prediction & Forecast for {ticker} using ARIMA',
        xaxis_title='Date',
        yaxis_title=' Stock Price',
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        legend_title='Legend',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5
        )
    )

    return fig

@app.callback(
    Output('pred-graph-xgboost', 'figure'),
    [Input('data-store-xgboost', 'data'),
    Input('ticker-dropdown', 'value')]
)
def update_graph_xgboost(data, ticker):
  if data is None:
    raise PreventUpdate

  fig = go.Figure()
  result_df = pd.DataFrame(data['result_df'])
  future_df = pd.DataFrame(data['future_df'])

  # Convert 'Date' columns to datetime
  result_df['Date'] = pd.to_datetime(result_df['Date'])
  future_df['Date'] = pd.to_datetime(future_df['Date'])

  # Add actual price line
  fig.add_trace(go.Scatter(
      x=result_df['Date'],
      y=result_df['Actual Price'],
      mode='lines',
      name='Actual Price',
      line=dict(color='blue')
  ))

  # Add training predicted price line
  fig.add_trace(go.Scatter(
      x=result_df['Date'],
      y=result_df['Training Predicted Price'],
      mode='lines',
      name='Training Predicted Price',
      line=dict(color='green')
  ))

  # Add testing predicted price line
  fig.add_trace(go.Scatter(
      x=result_df['Date'],
      y=result_df['Testing Predicted Price'],
      mode='lines',
      name='Testing Predicted Price',
      line=dict(color='orange')
  ))

  # Add forecasted price line
  fig.add_trace(go.Scatter(
      x=future_df['Date'],
      y=future_df['Forecasted Price'],
      mode='lines',
      name='Forecasted Price',
      line=dict(color='red')
  ))

  # Add a light blue shaded area for the forecast period
  fig.add_shape(
      type="rect",
      x0=future_df['Date'].iloc[0],
      x1=future_df['Date'].iloc[-1],
      y0=0,
      y1=1,
      xref='x',
      yref='paper',
      fillcolor='LightSkyBlue',
      opacity=0.3,
      layer='below',
      line_width=0,
  )

  # Update layout
  fig.update_layout(
      title=f'Stock Price Prediction & Forecast for {ticker} using XGBoost',
      xaxis_title='Date',
      yaxis_title='Price',
      xaxis_showgrid=False,
      yaxis_showgrid=False,
      legend_title='Legend',
      legend=dict(
            orientation='h',
            yanchor='bottom',
            y=0.02,
            xanchor='center',
            x=0.5
        )
  )

  # Set the range to include both the historical and forecasted dates
  two_years_ago = result_df['Date'].max() - pd.DateOffset(years=2)
  fig.update_xaxes(
      range=[two_years_ago, future_df['Date'].max()],
      rangeslider_visible=True
  )
  return fig

# Callback for Google Trend
@callback(
    Output('google-trend-plot', 'figure'),
    [Input('submit-keywords', 'n_clicks'),
     Input('ticker-dropdown', 'value')],
    [State('keyword1', 'value'),
     State('keyword2', 'value'),
     State('keyword3', 'value'),
     State('keyword4', 'value')]
)
def update_google_trends_plot(n_clicks, ticker, kw1, kw2, kw3, kw4):
    keywords = [kw for kw in [kw1, kw2, kw3, kw4] if kw]

    if not keywords:
        if ticker == 'AAPL':
            return plot_from_csv('./google_trends/apple_trend.csv')
        elif ticker == 'GOOG':
            return plot_from_csv('./google_trends/google_trend.csv')
        elif ticker == 'MSFT':
            return plot_from_csv('./google_trends/microsoft_trend.csv')
        elif ticker == 'AMZN':
            return plot_from_csv('./google_trends/amazon_trend.csv')
        else:
            return plot_from_csv('./google_trends/apple_trend.csv')

    # If keywords are entered, retrieve and plot the Google Trends data
    interest_over_time_df = get_trends_data(keywords)

    if interest_over_time_df is not None and not interest_over_time_df.empty:
        fig = go.Figure()
        for keyword in keywords:
            fig.add_trace(
                go.Scatter(x=interest_over_time_df.index, y=interest_over_time_df[keyword], mode='lines', name=keyword))

        fig.update_layout(
            title='Google Trends Data',
            xaxis_title='Date',
            yaxis_title='Search Interest',
            showlegend=True,
            legend=dict(
                yanchor='top',
                y=0.99,
                xanchor='right',
                x=0.99
            )
        )
        return fig
    elif ticker in ['AAPL', 'GOOG', 'MSFT', 'AMZN']:
        return plot_from_csv(f'./google_trends/{ticker.lower()}_trend.csv')
    else:
        return plot_from_csv('./google_trends/apple_trend.csv')

# Callbacks for modal open/close behavior - info button news section
@app.callback(
    Output('info-modal-news-table', 'is_open'),
    [Input('info-button-news-table', 'n_clicks')],
    [State('info-modal-news-table', 'is_open')],
)
def toggle_info_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('news-table', 'data'),
    [Input('ticker-dropdown', 'value'),]
)
def update_news_table(selected_ticker):
    news_df = fetch_news_for_ticker(selected_ticker)
    print('Fetched news data:', news_df)

    news_df = combine_sentiment_models(news_df)

    # Modify headlines to include anchor tags
    news_df = news_df[['Date', 'Time', 'Headline', 'URL', 'VADER Sentiment Score', 'Twitter RoBERTa Sentiment Score', 'DistilRoberta Sentiment Score', 'BART MNLI: Impact on Stock Price']]
    news_df['Headline'] = news_df.apply(lambda row: f"[{row['Headline']}]({row['URL']})", axis=1)

    return news_df.to_dict('records')


# Callbacks for modal open/close behavior - info button news section
@app.callback(
    Output('info-modal-news-plots', 'is_open'),
    [Input('info-button-news-plots', 'n_clicks')],
    [State('info-modal-news-plots', 'is_open')],
)
def toggle_info_modal(n1, is_open):
    if n1:
        return not is_open
    return is_open

@app.callback(
    Output('data-store-vader', 'data'),
    Input('ticker-dropdown', 'value')
)
def update_news_df_vader(selected_ticker):
    news_df_vader = calculate_sentiment_moving_avg(f'./news_sentiment/news_{selected_ticker}.csv', 'VADER Sentiment')
    return news_df_vader.to_dict('records')

@app.callback(
    Output('data-store-twitter-roberta', 'data'),
    Input('ticker-dropdown', 'value')
)
def update_news_df_twitter_roberta(selected_ticker):
    news_df_twitter_roberta = calculate_sentiment_moving_avg(f'./news_sentiment/news_{selected_ticker}_twitter_roberta.csv', 'Twitter RoBERTa Sentiment')
    return news_df_twitter_roberta.to_dict('records')

@app.callback(
    Output('data-store-distilroberta', 'data'),
    Input('ticker-dropdown', 'value')
)
def update_news_df_distilroberta(selected_ticker):
    news_df_distilroberta = calculate_sentiment_moving_avg(f'./news_sentiment/news_{selected_ticker}_distilroberta.csv', 'DistilRoberta Sentiment')
    return news_df_distilroberta.to_dict('records')

@app.callback(
    Output('data-store-bart-mnli', 'data'),
    Input('ticker-dropdown', 'value')
)
def update_news_df_bart_mnli(selected_ticker):
    news_df_bart_mnli = calculate_sentiment_moving_avg(f'./news_sentiment/news_{selected_ticker}_BART_MNLI.csv', 'BART MNLI')
    return news_df_bart_mnli.to_dict('records')

@app.callback(
  Output('news-vader-plot', 'figure'),
  [Input('data-store-vader', 'data'),
   Input('ticker-dropdown', 'value')]
)
def update_scatter_plot_vader(news_data, ticker):
    news_df = pd.DataFrame(news_data)

    # Scatter plot trace for the sentiment scores
    scatter_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['Daily Average VADER Sentiment Score'],
        mode='markers',
        marker=dict(color='grey', opacity=0.5),
        name='',
        showlegend=False,  # Hide the marker from the legend
        hovertemplate='Date = %{x}<br>Daily Average VADER Sentiment Score = %{y:.6f}',
        hoverlabel=dict(font=dict(color='white'))
    )

    # Moving average line trace
    moving_avg_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['10-Day Moving Average'],
        mode='lines',
        line=dict(color='red', width=2),
        name='10-Day Moving Average',
        hoverlabel=dict(namelength=-1)  # Ensure the full name is displayed in hover box
    )

    # Create the figure and add both traces
    fig = go.Figure(data=[scatter_trace, moving_avg_trace])

    # Adjust layout for better aesthetics
    fig.update_layout(
        title=f'Sentiment Analysis of News Headlines Related to {ticker}',
        xaxis_title='Date',
        yaxis_title='VADER Sentiment Score',
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig

@app.callback(
  Output('news-twitter-roberta-plot', 'figure'),
  [Input('data-store-twitter-roberta', 'data'),
   Input('ticker-dropdown', 'value')]
)
def update_scatter_plot_twitter_roberta(news_data, ticker):
    news_df = pd.DataFrame(news_data)

    # Scatter plot trace for the sentiment scores
    scatter_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['Daily Average Twitter RoBERTa Sentiment Score'],
        mode='markers',
        marker=dict(color='grey', opacity=0.5),
        name='',
        showlegend=False,  # Hide the marker from the legend
        hovertemplate='Date = %{x}<br>Daily Average Twitter RoBERTa Sentiment Score = %{y:.6f}',
        hoverlabel=dict(font=dict(color='white'))
    )

    # Moving average line trace
    moving_avg_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['10-Day Moving Average'],
        mode='lines',
        line=dict(color='red', width=2),
        name='10-Day Moving Average',
        hoverlabel=dict(namelength=-1)  # Ensure the full name is displayed in hover box
    )

    # Create the figure and add both traces
    fig = go.Figure(data=[scatter_trace, moving_avg_trace])

    # Adjust layout for better aesthetics
    fig.update_layout(
        title=f'Sentiment Analysis of News Headlines Related to {ticker}',
        xaxis_title='Date',
        yaxis_title='Twitter RoBERTa Sentiment Score',
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig

@app.callback(
  Output('news-distilroberta-plot', 'figure'),
  [Input('data-store-distilroberta', 'data'),
   Input('ticker-dropdown', 'value')]
)
def update_scatter_plot_distilroberta(news_data, ticker):
    news_df = pd.DataFrame(news_data)

    # Scatter plot trace for the sentiment scores
    scatter_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['Daily Average DistilRoberta Sentiment Score'],
        mode='markers',
        marker=dict(color='grey', opacity=0.5),
        name='',
        showlegend=False,  # Hide the marker from the legend
        hovertemplate='Date = %{x}<br>Daily Average DistilRoberta Sentiment Score = %{y:.6f}',
        hoverlabel=dict(font=dict(color='white'))
    )

    # Moving average line trace
    moving_avg_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['10-Day Moving Average'],
        mode='lines',
        line=dict(color='red', width=2),
        name='10-Day Moving Average',
        hoverlabel=dict(namelength=-1)  # Ensure the full name is displayed in hover box
    )

    # Create the figure and add both traces
    fig = go.Figure(data=[scatter_trace, moving_avg_trace])

    # Adjust layout for better aesthetics
    fig.update_layout(
        title=f'Sentiment Analysis of News Headlines Related to {ticker}',
        xaxis_title='Date',
        yaxis_title='DistilRoberta Sentiment Score',
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig

@app.callback(
  Output('news-bart-mnli-plot', 'figure'),
  [Input('data-store-bart-mnli', 'data'),
   Input('ticker-dropdown', 'value')]
)
def update_scatter_plot_bart_mnli(news_data, ticker):
    news_df = pd.DataFrame(news_data)

    # Scatter plot trace for the sentiment scores
    scatter_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['Daily Average BART MNLI Score'],
        mode='markers',
        marker=dict(color='grey', opacity=0.5),
        name='',
        showlegend=False,  # Hide the marker from the legend
        hovertemplate='Date = %{x}<br>Daily Average BART MNLI Score = %{y:.6f}',
        hoverlabel=dict(font=dict(color='white'))
    )

    # Moving average line trace
    moving_avg_trace = go.Scatter(
        x=news_df['Date'],
        y=news_df['10-Day Moving Average'],
        mode='lines',
        line=dict(color='red', width=2),
        name='10-Day Moving Average',
        hoverlabel=dict(namelength=-1)  # Ensure the full name is displayed in hover box
    )

    # Create the figure and add both traces
    fig = go.Figure(data=[scatter_trace, moving_avg_trace])

    # Adjust layout for better aesthetics
    fig.update_layout(
        title=f'Impact on the Stock Price of {ticker}',
        xaxis_title='Date',
        yaxis_title='BART MNLI Score',
        plot_bgcolor='rgba(240, 240, 240, 0.95)',
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    return fig

# Callback to update the page content based on the URL
@app.callback(
    Output('page-content', 'children'),
    [Input('url', 'pathname')]
)
def display_page(pathname):
    if pathname == '/impressum':
        return impressum_content
    elif pathname == '/datenschutzerklaerung':
        return datenschutz_content
    else:
        return main_content  # The main content of your dashboard


# Run the dashboard locally on http://127.0.0.1:8050/
if __name__ == '__main__':
  print('Running the dashboard locally on http://127.0.0.1:8050/')
  port = 8050
  webbrowser.open(f'http://127.0.0.1:{port}')
  app.run(debug=False, port=port)
  
