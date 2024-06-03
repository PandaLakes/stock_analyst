import streamlit as st
import pandas as pd
from prophet import Prophet
import plotly.express as px
import plotly.graph_objects as go
import yfinance as yf
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
import altair as alt
import random
from st_aggrid import AgGrid, GridOptionsBuilder, JsCode, AgGridTheme

# Download VADER lexicon
nltk.download('vader_lexicon')

# Initialize advanced sentiment analysis model
sentiment_model = pipeline('sentiment-analysis')

# Set page configuration
st.set_page_config(page_title="Stock Sentiment and Analysis Dashboard", page_icon="📊", layout="wide")

# Load custom CSS for styling
with open('style.css') as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def create_info_card(header, value):
    card = f"""
    <div class="card">
        <div class="card-body">
            <h5 class="card-title" style="font-weight: bold; font-size: 1.25rem; color: #1f77b4;">{header}</h5>
            <p class="card-text" style="font-size: 1rem; color: #333;">{value}</p>
        </div>
    </div>
    """
    return card

# Function to load data from an Excel file
def load_data(file_path):
    data = pd.read_excel(file_path)
    return data

# Function to clean data
def clean_data(data):
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    data['Title'] = data['Title'].str.replace(r'[^\w\s]', '', regex=True)
    return data

# Function to generate signals based on news titles
def generate_signals(data):
    buy_keywords = ['buy', 'growth', 'positive', 'up', 'increase']
    sell_keywords = ['sell', 'decline', 'negative', 'down', 'decrease']

    data['Signal'] = 'Hold'
    for i, row in data.iterrows():
        title = row['Title'].lower()
        if any(keyword in title for keyword in buy_keywords):
            data.at[i, 'Signal'] = 'Buy'
        elif any(keyword in title for keyword in sell_keywords):
            data.at[i, 'Signal'] = 'Sell'

    return data

# Function to fetch news
def fetch_news(tickers):
    finviz_url = 'https://finviz.com/quote.ashx?t='
    news_tables = {}
    for ticker in tickers:
        url = finviz_url + ticker
        req = Request(url=url, headers={'user-agent': 'my-app'})
        response = urlopen(req)
        html = BeautifulSoup(response, 'html.parser')
        news_table = html.find(id='news-table')
        news_tables[ticker] = news_table
    return news_tables

# Function to parse news
def parse_news(news_tables):
    parsed_data = []
    for ticker, news_table in news_tables.items():
        if news_table:
            for row in news_table.findAll('tr'):
                if row.a:
                    title = row.a.get_text()
                    link = row.a['href']
                else:
                    continue
                date_data = row.td.text.strip().split(' ')
                if len(date_data) == 1:
                    date = datetime.now().strftime('%b-%d-%y')
                    time = date_data[0]
                else:
                    date = date_data[0]
                    time = date_data[1]
                parsed_data.append([ticker, date, time, title, link])
    return parsed_data

# Function to standardize dates
def standardize_date(date_str):
    if date_str == "Today":
        return datetime.now().strftime("%Y-%m-%d")
    else:
        try:
            return datetime.strptime(date_str, "%b-%d-%y").strftime("%Y-%m-%d")
        except ValueError:
            return date_str

# Function to analyze advanced sentiment
def analyze_sentiment_advanced(headline):
    result = sentiment_model(headline)
    sentiment = result[0]['label']
    score = result[0]['score']
    return sentiment, score

# Function to clean news data
def clean_news_data(data):
    data['date'] = pd.to_datetime(data['date'], errors='coerce')
    data['headline'] = data['headline'].str.replace(r'[^\w\s]', '', regex=True)
    return data

# Function to generate signals based on advanced sentiment
def generate_signals_advanced(data):
    data['Signal'] = 'Hold'
    for i, row in data.iterrows():
        if row['sentiment'] == 'POSITIVE' and row['sentiment_score'] > 0.7:
            data.at[i, 'Signal'] = 'Buy'
        elif row['sentiment'] == 'NEGATIVE' and row['sentiment_score'] > 0.7:
            data.at[i, 'Signal'] = 'Sell'
    return data

# Function to perform sentiment analysis and generate signals
def perform_sentiment_analysis_and_signals(tickers):
    news_tables = fetch_news(tickers)
    parsed_data = parse_news(news_tables)

    # Convert to DataFrame
    columns = ['ticker', 'date', 'time', 'headline', 'link']
    news_df = pd.DataFrame(parsed_data, columns=columns)

    # Standardize dates
    news_df['date'] = news_df['date'].apply(standardize_date)

    # Clean news data
    news_df = clean_news_data(news_df)

    # Perform advanced sentiment analysis
    sentiments = [analyze_sentiment_advanced(headline) for headline in news_df['headline']]
    sentiments_df = pd.DataFrame(sentiments, columns=['sentiment', 'sentiment_score'])
    news_df = news_df.join(sentiments_df)

    # Generate signals
    news_df = generate_signals_advanced(news_df)
    return news_df

# Function to create news ticker
def create_news_ticker(news_df):
    news_items = news_df[['ticker', 'headline', 'link']].apply(lambda row: f'<a href="{row["link"]}" target="_blank">{row["ticker"]}: {row["headline"]}</a>', axis=1).tolist()
    news_ticker_html = """
    <div class="news-ticker">
        <div>
            {items}
        </div>
    </div>
    """.format(items=' &bullet; '.join(news_items))
    return news_ticker_html

# Function to create a plotly chart for trading signals
def create_trading_signals_chart(news_df, ticker):
    ticker_news = news_df[news_df['ticker'] == ticker]

    # Group by date to avoid multiple signals on the same date
    ticker_news = ticker_news.groupby('date').agg({
        'sentiment_score': 'mean',
        'Signal': 'first'
    }).reset_index()

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=ticker_news['date'], y=ticker_news['sentiment_score'], mode='lines+markers', name='Sentiment Score', line=dict(color='purple')))

    buy_signals = ticker_news[ticker_news['Signal'] == 'Buy']
    sell_signals = ticker_news[ticker_news['Signal'] == 'Sell']
    hold_signals = ticker_news[ticker_news['Signal'] == 'Hold']

    fig.add_trace(go.Scatter(x=buy_signals['date'], y=buy_signals['sentiment_score'], mode='markers', name='Buy Signal', marker=dict(color='green', symbol='triangle-up', size=10)))
    fig.add_trace(go.Scatter(x=sell_signals['date'], y=sell_signals['sentiment_score'], mode='markers', name='Sell Signal', marker=dict(color='red', symbol='triangle-down', size=10)))
    fig.add_trace(go.Scatter(x=hold_signals['date'], y=hold_signals['sentiment_score'], mode='markers', name='Hold Signal', marker=dict(color='blue', symbol='circle', size=10)))

    # Add trend line
    fig.add_trace(go.Scatter(x=ticker_news['date'], y=ticker_news['sentiment_score'], mode='lines', name='Trend Line', line=dict(dash='dash', color='magenta')))

    # Add annotations
    fig.add_annotation(x=ticker_news['date'].iloc[-1], y=ticker_news['sentiment_score'].iloc[-1],
                       text="Last Point", showarrow=True, arrowhead=1)

    fig.update_layout(
        title=f'{ticker} Stock Trading Signals',
        xaxis_title='Date',
        yaxis_title='Sentiment Score',
        width=1400,
        height=600,
        xaxis_rangeslider_visible=True,
        template='plotly_dark',
        hovermode='x unified'
    )

    fig.update_xaxes(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1M', step='month', stepmode='backward'),
                dict(count=6, label='6M', step='month', stepmode='backward'),
                dict(count=1, label='YTD', step='year', stepmode='todate'),
                dict(count=1, label='1Y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible=True),
        type='date'
    )

    return fig

# Function to get stock price data and predictions
def get_stock_data(ticker):
    df = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
    df.reset_index(inplace=True)

    # Prepare data for Prophet
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    prophet = Prophet(daily_seasonality=True)
    prophet.fit(df_prophet)

    future_dates = prophet.make_future_dataframe(periods=30)
    forecast = prophet.predict(future_dates)

    return df, forecast

# Function to generate recommendations based on forecast and sentiment
def generate_recommendations(forecast, news_df):
    recommendations = []
    sentiment_score = news_df['sentiment_score'].mean()

    # Filter for future dates within the next 30 days
    future_forecast = forecast[forecast['ds'] > datetime.now()]
    future_forecast = future_forecast[future_forecast['ds'] <= datetime.now() + timedelta(days=30)]

    # Simple logic to determine buy, sell, hold
    for i in range(len(future_forecast) - 1):
        if future_forecast['yhat'].iloc[i] < future_forecast['yhat'].iloc[i + 1] and sentiment_score > 0:
            recommendations.append('Buy')
        elif future_forecast['yhat'].iloc[i] > future_forecast['yhat'].iloc[i + 1] and sentiment_score < 0:
            recommendations.append('Sell')
        else:
            recommendations.append('Hold')

    recommendations.append('Hold')  # For the last day, hold as default

    return recommendations

# Function to compare multiple tickers to buy
def compare_tickers_to_buy(tickers):
    comparison_data = []
    for ticker in tickers:
        news_df = perform_sentiment_analysis_and_signals([ticker])
        df, forecast = get_stock_data(ticker)
        sentiment_score = news_df['sentiment_score'].mean()
        future_price_increase = forecast['yhat'].iloc[-1] > df['Close'].iloc[-1]
        
        comparison_data.append({
            'Ticker': ticker,
            'Current Price': df['Close'].iloc[-1],
            'Sentiment Score': sentiment_score,
            'Future Price Increase': future_price_increase,
            'Recommendation': 'Buy' if sentiment_score > 0 and future_price_increase else 'Hold' if sentiment_score >= 0 else 'Sell'
        })

    comparison_df = pd.DataFrame(comparison_data)
    best_ticker = comparison_df[comparison_df['Recommendation'] == 'Buy'].sort_values(by='Sentiment Score', ascending=False).head(1)
    return comparison_df, best_ticker

# Function to compare multiple tickers to sell
def compare_tickers_to_sell(tickers):
    comparison_data = []
    for ticker in tickers:
        news_df = perform_sentiment_analysis_and_signals([ticker])
        df, forecast = get_stock_data(ticker)
        sentiment_score = news_df['sentiment_score'].mean()
        future_price_decrease = forecast['yhat'].iloc[-1] < df['Close'].iloc[-1]
        
        comparison_data.append({
            'Ticker': ticker,
            'Current Price': df['Close'].iloc[-1],
            'Sentiment Score': sentiment_score,
            'Future Price Decrease': future_price_decrease,
            'Recommendation': 'Sell' if sentiment_score < 0 and future_price_decrease else 'Hold' if sentiment_score <= 0 else 'Buy'
        })

    comparison_df = pd.DataFrame(comparison_data)
    best_ticker = comparison_df[comparison_df['Recommendation'] == 'Sell'].sort_values(by=['Sentiment Score', 'Future Price Decrease'], ascending=[True, True]).head(1)
    return comparison_df, best_ticker

# Function to generate Altair chart for signals
def generate_altair_chart(data_with_signals):
    data_with_signals['Date'] = data_with_signals['Date'].dt.strftime('%Y-%m-%d')
    chart = alt.Chart(data_with_signals).mark_circle(size=60).encode(
        x=alt.X('Date:T', title='Date'),
        y=alt.Y('Ticker:N', title='Ticker'),
        color=alt.Color('Signal:N', scale=alt.Scale(domain=['Buy', 'Hold', 'Sell'], range=['green', 'blue', 'red']), title='Signal'),
        tooltip=['Date', 'Ticker', 'Title', 'Signal']
    ).properties(
        title='Stock Trading Signals Based on News Headlines',
        width=800,
        height=400
    ).interactive()

    return chart

# Function to get random tickers
def get_random_tickers(tickers, n=5):
    return random.sample(tickers, n)

# Function to generate recommendation table
def generate_recommendation_table(news_df):
    table_data = []
    for ticker in news_df['ticker'].unique():
        ticker_news = news_df[news_df['ticker'] == ticker]
        sentiment_score = ticker_news['sentiment_score'].mean()
        if sentiment_score > 0.5:
            reco = 'Buy'
        elif sentiment_score < -0.5:
            reco = 'Sell'
        else:
            reco = 'Hold'
        
        # Fetch the stock price
        stock_price = get_stock_data(ticker)[0]['Close'].iloc[-1]
        table_data.append([ticker, reco, sentiment_score, stock_price])
    
    table_df = pd.DataFrame(table_data, columns=['Ticker', 'Reco', 'Sentiment Score', 'Stock Price'])
    return table_df

# Function to display recommendation table in Streamlit
def display_recommendation_table(table_df):
    st.subheader("Recommandations de Trading")
    
    # Configure AgGrid
    gb = GridOptionsBuilder.from_dataframe(table_df)
    gb.configure_columns(["Ticker", "Sentiment Score", "Stock Price"], cellStyle={"textAlign": "center"})
    
    cell_style_jscode = JsCode("""
    function(params) {
        if (params.value == 'Buy') {
            return {'color': 'white', 'backgroundColor': 'green', 'fontWeight': 'bold', 'fontSize': '14px'};
        } else if (params.value == 'Sell') {
            return {'color': 'white', 'backgroundColor': 'red', 'fontWeight': 'bold', 'fontSize': '14px'};
        } else {
            return {'color': 'white', 'backgroundColor': 'blue', 'fontWeight': 'bold', 'fontSize': '14px'};
        }
    }
    """)
    
    gb.configure_column("Reco", cellStyle=cell_style_jscode)
    gridOptions = gb.build()
    
    # Display table with AgGrid
    st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
    AgGrid(
        table_df, 
        gridOptions=gridOptions,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        theme=AgGridTheme.STREAMLIT,
        height=200, # Increase table height
        fit_columns_on_grid_load=True, # Automatically adjust column width
        reload_data=True, # Reload data on update
        width=1200 # Increase table width
    )
    st.markdown("</div>", unsafe_allow_html=True)

# Page: Stock Analysis
def stock_analysis():
    # Fetch and display news sentiment and signals for all tickers
    all_tickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'META', 'NFLX', 'NVDA', 'INTC', 'AMD']
    news_df = perform_sentiment_analysis_and_signals(all_tickers)
    
    # Create and display news ticker
    news_ticker_html = create_news_ticker(news_df)
    st.markdown(news_ticker_html, unsafe_allow_html=True)
    
    st.title("📈 Stock Investment Analyzer")
    
    ticker = st.text_input("Enter Stock Ticker", value='AAPL')
    if st.button("Analyze"):
        # Fetch and display news sentiment and signals for the selected ticker
        ticker_news_df = perform_sentiment_analysis_and_signals([ticker])
        st.subheader("News Sentiment Analysis and Signals")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        st.dataframe(ticker_news_df[['date', 'headline', 'sentiment', 'sentiment_score', 'Signal']])
        st.markdown("</div>", unsafe_allow_html=True)

        # Fetch and display stock data
        df, forecast = get_stock_data(ticker)
        st.subheader("Stock Price Data")
        st.markdown("<div class='centered-plot'>", unsafe_allow_html=True)
        fig = px.line(df, x='Date', y='Close', title=f'{ticker} Stock Price')
        st.plotly_chart(fig)
        st.markdown("</div>", unsafe_allow_html=True)

        st.subheader("Stock Price Prediction")
        st.markdown("<div class='centered-plot'>", unsafe_allow_html=True)
        fig_forecast = px.line(forecast, x='ds', y='yhat', title=f'{ticker} Stock Price Prediction')
        st.plotly_chart(fig_forecast)
        st.markdown("</div>", unsafe_allow_html=True)

        # Generate and display recommendations
        recommendations = generate_recommendations(forecast, ticker_news_df)
        forecast_filtered = forecast[forecast['ds'] > datetime.now()]
        forecast_filtered = forecast_filtered[forecast_filtered['ds'] <= datetime.now() + timedelta(days=30)]
        forecast_filtered['Recommendation'] = recommendations
        st.subheader("Investment Recommendations")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        st.dataframe(forecast_filtered[['ds', 'yhat', 'Recommendation']])
        st.markdown("</div>", unsafe_allow_html=True)

        # Create and display trading signals chart
        st.subheader("Trading Signals")
        st.markdown("<div class='centered-plot'>", unsafe_allow_html=True)
        trading_signals_chart = create_trading_signals_chart(ticker_news_df, ticker)
        st.plotly_chart(trading_signals_chart)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Generate and display Altair chart
        st.subheader("News-based Trading Signals")
        st.markdown("<div class='centered-plot'>", unsafe_allow_html=True)
        data_with_signals = ticker_news_df[['date', 'ticker', 'headline', 'Signal']].rename(columns={'date': 'Date', 'ticker': 'Ticker', 'headline': 'Title'})
        altair_chart = generate_altair_chart(data_with_signals)
        st.altair_chart(altair_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Display random tickers recommendations
        st.subheader("Random Tickers Recommendations")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        random_news_df = perform_sentiment_analysis_and_signals(get_random_tickers(all_tickers))
        table_df = generate_recommendation_table(random_news_df)
        display_recommendation_table(table_df)
        st.markdown("</div>", unsafe_allow_html=True)

# Page: Best Ticker to Buy
def best_ticker_to_buy():
    st.title("💹 Best Ticker to Buy")

    tickers = st.text_input("Enter Stock Tickers (comma separated)", value='AAPL,GOOGL,MSFT').split(',')
    if st.button("Find Best Ticker"):
        comparison_df, best_ticker = compare_tickers_to_buy(tickers)
        
        st.subheader("Best Ticker to Buy Details")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        for index, row in best_ticker.iterrows():
            card_html = ""
            card_html += create_info_card("Ticker", row['Ticker'])
            card_html += create_info_card("Current Price", row['Current Price'])
            card_html += create_info_card("Sentiment Score", row['Sentiment Score'])
            card_html += create_info_card("Recommendation", row['Recommendation'])
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Page: Best Ticker to Sell
def best_ticker_to_sell():
    st.title("📉 Best Ticker to Sell")

    tickers = st.text_input("Enter Stock Tickers (comma separated)", value='AAPL,GOOGL,MSFT').split(',')
    if st.button("Find Best Ticker"):
        comparison_df, best_ticker = compare_tickers_to_sell(tickers)
        
        st.subheader("Best Ticker to Sell Details")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        for index, row in best_ticker.iterrows():
            card_html = ""
            card_html += create_info_card("Ticker", row['Ticker'])
            card_html += create_info_card("Current Price", row['Current Price'])
            card_html += create_info_card("Sentiment Score", row['Sentiment Score'])
            card_html += create_info_card("Recommendation", row['Recommendation'])
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Page: Upload Data and Generate Signals
def upload_and_generate_signals():
    st.title("📊 Upload Data and Generate Signals")

    uploaded_file = st.file_uploader("Choose an Excel file", type="xlsx")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        data = clean_data(data)
        data_with_signals = generate_signals(data)
        st.subheader("Data with Generated Signals")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        st.dataframe(data_with_signals)
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.subheader("Trading Signals Chart")
        st.markdown("<div class='centered-plot'>", unsafe_allow_html=True)
        altair_chart = generate_altair_chart(data_with_signals)
        st.altair_chart(altair_chart, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Page: Date Prediction
def date_prediction():
    st.title("📅 Date-based Prediction")

    ticker = st.text_input("Enter Stock Ticker", value='AAPL')
    date = st.date_input("Enter Date", value=datetime.now().date())
    if st.button("Predict"):
        news_df = perform_sentiment_analysis_and_signals([ticker])
        df, forecast = get_stock_data(ticker)

        sentiment_score = news_df[news_df['date'] == pd.to_datetime(date)]['sentiment_score'].mean()
        prediction = forecast[forecast['ds'] == pd.to_datetime(date)]['yhat'].values[0]

        st.subheader(f"Prediction for {ticker} on {date.strftime('%Y-%m-%d')}")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        card_html = ""
        card_html += create_info_card("Ticker", ticker)
        card_html += create_info_card("Date", date.strftime('%Y-%m-%d'))
        card_html += create_info_card("Predicted Sentiment Score", sentiment_score)
        card_html += create_info_card("Predicted Stock Price", prediction)
        st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Page: Historical Price Comparison
def historical_price_comparison():
    st.title("📜 Historical Price Comparison")

    tickers = st.text_input("Enter Stock Tickers (comma separated)", value='AAPL,GOOGL,MSFT').split(',')
    if st.button("Compare"):
        if tickers:
            fig = go.Figure()
            st.markdown("<div class='centered-plot'>", unsafe_allow_html=True)
            for ticker in tickers:
                df = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime('%Y-%m-%d'))
                df.reset_index(inplace=True)
                fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name=ticker))

            fig.update_layout(
                title="Historical Price Comparison",
                xaxis_title='Date',
                yaxis_title='Stock Price',
                width=1400,
                height=600,
                xaxis_rangeslider_visible=True,
                template='plotly_dark',
                hovermode='x unified'
            )

            st.plotly_chart(fig)
            st.markdown("</div>", unsafe_allow_html=True)

# Page: Interactive News Feed
def interactive_news_feed():
    st.title("📰 Interactive News Feed")

    tickers = st.text_input("Enter Stock Tickers (comma separated)", value='AAPL,GOOGL,MSFT').split(',')
    if st.button("Fetch News"):
        news_tables = fetch_news(tickers)
        parsed_data = parse_news(news_tables)
        news_df = pd.DataFrame(parsed_data, columns=['ticker', 'date', 'time', 'headline', 'link'])

        st.subheader("Latest News")
        st.markdown("<div class='centered-table'>", unsafe_allow_html=True)
        for index, row in news_df.iterrows():
            card_html = f"""
            <div class="card">
                <div class="card-body">
                    <p class="card-text" style="font-size: 1rem; color: #333;">{row['date']} {row['time']}</p>
                    <h5 class="card-title" style="font-size: 1.25rem; color: #1f77b4;"><a href="{row['link']}" target="_blank">{row['headline']}</a></h5>
                </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# Main function to set up the app
def main():
    st.sidebar.image('logo.png', width=150)
    # Custom CSS for sidebar background image
    sidebar_bg_css = """
    <style>
    [data-testid="stSidebar"] {
        background-image: url('background.png');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    </style>
    """
    st.markdown(sidebar_bg_css, unsafe_allow_html=True)
    
    page = st.sidebar.radio("Go to", ["📈 Stock Analysis", "💹 Best Ticker to Buy", "📉 Best Ticker to Sell", "📊 Upload Data and Generate Signals", "📅 Date Prediction", "📜 Historical Price Comparison", "📰 Interactive News Feed"])

    if page == "📈 Stock Analysis":
        stock_analysis()
    elif page == "💹 Best Ticker to Buy":
        best_ticker_to_buy()
    elif page == "📉 Best Ticker to Sell":
        best_ticker_to_sell()
    elif page == "📊 Upload Data and Generate Signals":
        upload_and_generate_signals()
    elif page == "📅 Date Prediction":
        date_prediction()
    elif page == "📜 Historical Price Comparison":
        historical_price_comparison()
    elif page == "📰 Interactive News Feed":
        interactive_news_feed()

if __name__ == '__main__':
    main()
