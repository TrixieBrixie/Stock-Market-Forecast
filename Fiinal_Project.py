import praw
import prawcore 
import yfinance as yf
from transformers import pipeline
from textblob import TextBlob
import psycopg2
import datetime
import re
import time
import requests
from googleapiclient.discovery import build
import pandas as pd

# Open the log file in write mode to overwrite it at the start
log_file = open("script_output.log", "w")
log_file.close()

def log(message):
    with open("script_output.log", "a") as log_file:
        log_file.write(f"{datetime.datetime.now()} - {message}\n")
    print(f"{datetime.datetime.now()} - {message}")

# Create a Reddit instance 
try:
    reddit = praw.Reddit(
        client_id='13wtVcBB17S_GSs0aVZWJA',
        client_secret='4Z_oN2ty4_87Q0tIeQpRZ9CKV40Agg',
        user_agent='script:reddit_scraper:v1.0 (by /u/TrixieBrixie)',
        username='TrixieBrixie',
        password='251077Giorgia'
    )
    log("Reddit instance created successfully")
except Exception as e:
    log(f"Failed to create Reddit instance: {e}")
    reddit = None

# Create database connection 
try:
    conn = psycopg2.connect(
        host='localhost',
        database='redditsentiment',
        user='postgres',
        password='080111'
    )
    cur = conn.cursor()
    log("Database connection successful")
except Exception as e:
    log(f"Failed to connect to the database: {e}")
    conn = None

if conn and reddit:
    try:
        # Initialise a sentiment analysis pipeline using a pre-trained BERT model
        classifier = pipeline('sentiment-analysis')
        log("Sentiment analysis pipeline initialized successfully")
    except Exception as e:
        log(f"Failed to initialize sentiment analysis pipeline: {e}")
        classifier = None 

    # Clean text
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub(r'@\w+', ' ', text)
        text = re.sub(r'\d+', ' ', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Analyze sentiment using BERT
    def analyze_sentiment_bert(text):
        if classifier is None:
            log("Sentiment analysis pipeline not initialized. Exiting sentiment analysis.")
            return None, None, None
        cleaned_text = clean_text(text)
        result = classifier(cleaned_text)
        sentiment_label = result[0]['label']
        sentiment_confidence = result[0]['score']
        polarity = 1 if sentiment_label == 'POSITIVE' else -1
        subjectivity = TextBlob(cleaned_text).sentiment.subjectivity  
        weighted_polarity = polarity * sentiment_confidence
        return weighted_polarity, sentiment_confidence, subjectivity

    # Fetch FTSE 100 data for the specified dates
    def fetch_ftse100_data(date):
        try:
            log(f"Fetching FTSE 100 data for {date}")
            ftse100_intraday = yf.download("^FTSE", start=date, end=(date + datetime.timedelta(days=1)).strftime("%Y-%m-%d"))
            if ftse100_intraday.empty:
                log("No data returned for FTSE 100 on the specified date.")
            else:
                log(f"FTSE 100 data: {ftse100_intraday}")
            return ftse100_intraday
        except Exception as e:
            log(f"Failed to fetch FTSE 100 data: {e}")
            return None

    # Fetch financial news articles using News API
    def fetch_financial_news(api_key, target_date, keywords):
        log(f"Fetching financial news articles for {target_date}")
        url = f"https://newsapi.org/v2/everything?q={' OR '.join(keywords)}&from={target_date}&to={target_date}&sortBy=popularity&apiKey={api_key}"
        try:
            response = requests.get(url)
            articles = response.json().get('articles', [])
            log(f"Fetched {len(articles)} articles from News API")
            return articles
        except Exception as e:
            log(f"Failed to fetch news articles: {e}")
            return []

    # Fetch Google News
    def fetch_google_news(api_key, target_date, keywords):
        log(f"Fetching Google news articles for {target_date}")
        try:
            service = build("customsearch", "v1", developerKey=api_key)
            query = ' OR '.join(keywords)
            res = service.cse().list(
                q=query,
                cx='a6bbc5606e0c34b81',
                sort=f'date:r:{target_date.strftime("%Y%m%d")}:{target_date.strftime("%Y%m%d")}',
                num=50  
            ).execute()
            articles = res.get('items', [])
            log(f"Fetched {len(articles)} articles from Google News API")
            log(f"Response: {res}")
            return articles
        except Exception as e:
            log(f"Failed to fetch Google news articles: {e}")
            return []

    # Fetch and analyze subreddit posts with keyword filtering
    def fetch_and_analyze_subreddit(subreddit_name, target_date, keywords, confidence_threshold=0.75, polarity_threshold=0.5):
        subreddit = reddit.subreddit(subreddit_name)
        posts_fetched = 0
        log(f"Fetching posts from subreddit '{subreddit_name}' on {target_date}")
        try:
            for post in subreddit.new(limit=2000): 
                post_date = datetime.datetime.fromtimestamp(post.created_utc).date()
                if post_date != target_date:
                    # Skip posts outside the target date
                    continue  
                
                cur.execute("SELECT COUNT(*) FROM sentimentdata WHERE post_id = %s", (str(post.id),))
                if cur.fetchone()[0] > 0:
                    # Skip posts that ahve already been processed
                    continue  

                log(f"Processing post from {post_date}: {post.title}")
                # Include both the title & body of the post
                post_content = f"{post.title} {post.selftext}"  
                post_content_lower = post_content.lower()
                if not any(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', post_content_lower) for keyword in keywords):
                    log(f"Skipped post: '{post.title}' (No keywords matched)")
                    continue

                log(f"Content: {post_content}")
                weighted_polarity, sentiment_confidence, subjectivity = analyze_sentiment_bert(post_content)
                # If sentiment result is None skip the article
                if sentiment_confidence is None or weighted_polarity is None:
                    log("Skipping article due to invalid sentiment analysis results.")
                    continue
                if sentiment_confidence >= confidence_threshold and abs(weighted_polarity) > polarity_threshold:
                    log(f"Sentiment: Weighted Polarity = {weighted_polarity}, Confidence = {sentiment_confidence}, Subjectivity = {subjectivity}")
                    timestamp = datetime.datetime.fromtimestamp(post.created_utc)
                    cur.execute(
                        'INSERT INTO sentimentdata (post_id, timestamp, content, title, polarity, confidence, subjectivity, subreddit) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT (post_id) DO NOTHING;',
                        (str(post.id), timestamp, post_content, post.title, weighted_polarity, sentiment_confidence, subjectivity, subreddit_name)
                    )
                    conn.commit()
                    log(f"Stored: {post.title}")
                    posts_fetched += 1
                    if posts_fetched >= 50:
                        log(f"Fetched {posts_fetched} posts, stopping.")
                        break
                else:
                    log(f"Skipped low confidence sentiment for: {post.title}")

        except praw.exceptions.APIException as e:
            if e.error_type == "RATELIMIT":
                delay = int(e.message.split("minute")[0].split()[-1])
                log(f"Rate limited. Sleeping for {delay} minutes.")
                time.sleep(delay * 60)
        except praw.exceptions.ClientException as e:
            log(f"Client error: {e}")

    # Fetch and analyze Financial News
    def fetch_and_analyze_news(api_key, target_date, keywords, confidence_threshold=0.75, polarity_threshold=0.5):
        articles = fetch_financial_news(api_key, target_date, keywords)
        for article in articles:
            title = article['title']
            content = article['content']
            source = article['source']['name']
            if not title or not content:
                continue

            log(f"Processing news article: {title}")
            log(f"Content: {content}")
            weighted_polarity, sentiment_confidence, subjectivity = analyze_sentiment_bert(content)
            # If sentiment result is None skip the article
            if sentiment_confidence is None or weighted_polarity is None:
                log("Skipping article due to invalid sentiment analysis results.")
                continue
            if sentiment_confidence >= confidence_threshold and abs(weighted_polarity) > polarity_threshold:
                log(f"Sentiment: Weighted Polarity = {weighted_polarity}, Confidence = {sentiment_confidence}, Subjectivity = {subjectivity}")
                timestamp = datetime.datetime.strptime(article['publishedAt'], "%Y-%m-%dT%H:%M:%SZ")
                cur.execute(
                    'INSERT INTO newsdata (timestamp, title, content, polarity, confidence, subjectivity, source) VALUES (%s, %s, %s, %s, %s, %s, %s);',
                    (timestamp, title, content, weighted_polarity, sentiment_confidence, subjectivity, source)
                )
                conn.commit()
                log(f"Stored news article: {title}")

    # Function to fetch and analyze Google News
    def fetch_and_analyze_google_news(api_key, target_date, keywords, confidence_threshold=0.75, polarity_threshold=0.5):
        articles = fetch_google_news(api_key, target_date, keywords)
        log(f"Number of articles fetched: {len(articles)}")
        for article in articles:
            title = article.get('title', '')
            snippet = article.get('snippet', '')
            link = article.get('link', '')

            if not title or not snippet:
                log(f"Skipping article due to missing title or snippet. Title: {title}, Snippet: {snippet}")
                continue

            log(f"Processing Google news article: {title}")
            log(f"Snippet: {snippet}")
            weighted_polarity, sentiment_confidence, subjectivity = analyze_sentiment_bert(snippet)
            log(f"Sentiment analysis result - Polarity: {weighted_polarity}, Confidence: {sentiment_confidence}, Subjectivity: {subjectivity}")
            # If sentiment result is None skip the article
            if sentiment_confidence is None or weighted_polarity is None:
                log("Skipping article due to invalid sentiment analysis results.")
                continue

            if sentiment_confidence >= confidence_threshold and abs(weighted_polarity) > polarity_threshold:
                log(f"Article meets confidence and polarity thresholds, storing in database")
                try:
                    timestamp_str = article.get('pagemap', {}).get('metatags', [{}])[0].get('og:updated_time', None)
                    if timestamp_str:
                        timestamp = datetime.datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%fZ")
                    else:
                        # Fallback to current time if timestamp is missing
                        timestamp = datetime.datetime.now() 
                except KeyError:
                    # Fallback to current time if timestamp is missing
                    timestamp = datetime.datetime.now()  

                try:
                    cur.execute(
                        'INSERT INTO googlenewsdata (timestamp, title, snippet, link, polarity, confidence, subjectivity) VALUES (%s, %s, %s, %s, %s, %s, %s);',
                        (timestamp, title, snippet, link, weighted_polarity, sentiment_confidence, subjectivity)
                    )
                    conn.commit()
                    log(f"Stored Google news article: {title}")
                except Exception as e:
                    log(f"Failed to insert data into googlenewsdata table: {e}")
            else:
                log(f"Article does not meet sentiment thresholds. Skipping.")

        log(f"Finished processing Google news articles")

    def correlate_sentiment_with_price_and_volatility():
        cur.execute("SELECT timestamp, polarity, confidence FROM sentimentdata")
        sentiment_data = cur.fetchall()

        cur.execute("SELECT timestamp, polarity, confidence FROM newsdata")
        news_data = cur.fetchall()

        cur.execute("SELECT timestamp, polarity, confidence FROM googlenewsdata")
        google_news_data = cur.fetchall()

        combined_sentiment_data = [
            (data[0].date(), 'Reddit', data[1], data[2]) for data in sentiment_data
        ] + [
            (data[0].date(), 'Financial News', data[1], data[2]) for data in news_data
        ] + [
            (data[0].date(), 'Google News', data[1], data[2]) for data in google_news_data
        ]

        correlation_results = []

        for data in combined_sentiment_data:
            date = data[0]
            source = data[1]
            polarity = data[2]
            confidence = data[3]

            next_date = date + datetime.timedelta(days=1)
            ftse_data = fetch_ftse100_data(next_date)

            # Check if FTSE 100 data is available
            if ftse_data is None or ftse_data.empty:
                log(f"Skipping correlation for {next_date} due to missing FTSE 100 data.")
                continue

            close_price = ftse_data['Close'].iloc[0]
            open_price = ftse_data['Open'].iloc[0]
            high_price = ftse_data['High'].iloc[0]
            low_price = ftse_data['Low'].iloc[0]
            price_change = (close_price - open_price) / open_price
            volatility = (high_price - low_price) / open_price

            # Log the data is being processed
            log(f"Processing data for date: {date}, source: {source}, price change: {price_change}, volatility: {volatility}")

            # Insert combined sentiment data into the new_combined_sentiment_data table
            cur.execute(
                'INSERT INTO new_combined_sentiment_data (date, source, polarity, confidence, price_change, volatility) VALUES (%s, %s, %s, %s, %s, %s);',
                (date, source, polarity, confidence, price_change, volatility)
            )

            # Check if the correlation result already exists
            cur.execute("SELECT COUNT(*) FROM correlation_results WHERE date = %s AND source = %s AND polarity = %s AND confidence = %s", 
                        (date, source, polarity, confidence))
            count = cur.fetchone()[0]

            # Only update if a match is found with all key attributes; otherwise, insert a new record
            if count > 0:
                log(f"Found existing entry for {date}, {source}, updating it.")
                cur.execute('UPDATE correlation_results SET price_change = %s, volatility = %s WHERE date = %s AND polarity = %s AND confidence = %s AND source = %s', 
                            (price_change, volatility, date, polarity, confidence, source))
                log(f"Updated existing correlation result for date: {date}")
            else:
                log(f"No existing entry found for {date}, {source}. Inserting a new record.")
                cur.execute(
                    'INSERT INTO correlation_results (date, source, polarity, confidence, price_change, volatility) VALUES (%s, %s, %s, %s, %s, %s);',
                    (date, source, polarity, confidence, price_change, volatility)
                )
                log(f"Inserted new correlation result for date: {date}")

            conn.commit()

        return correlation_results

# API key for News API
news_api_key = '5db692a659e2427581ca7fa47ebf3619'

# API key for Google News API
google_news_api_key = 'AIzaSyB8MFQIHoW4DZ2HYg7ObBzV8oc4uXwL0DQ'

# List of subreddits to collect data from
subreddits = ['Investing', 'Stocks', 'Finance', 'Markets', 'WallStreetBets', 'StockMarket', 'Economics', 'FinancialPlanning', 'Daytrading', 'CryptoCurrency', 'InvestmentClub', 'PersonalFinance']

if __name__ == '__main__':
    # Set dates for data collection
    today = datetime.datetime.now().date()
    day_before_today = today - datetime.timedelta(days=1)
    two_days_before_today = today - datetime.timedelta(days=2)

    # Fetch FTSE 100 data for the day before today 
    fetch_ftse100_data(day_before_today)

    # Keywords to search for in Reddit posts, financial news, and Google news
    keywords = ['ftse', 'market', 'shares', 'stocks', 'equities', 'london exchange', 'uk stock', 'dividend']

    for subreddit_name in subreddits:
        try:
            fetch_and_analyze_subreddit(subreddit_name, two_days_before_today, keywords, 0.75, 0.5)
        except prawcore.exceptions.NotFound:
            log(f"Subreddit '{subreddit_name}' not found or inaccessible.")
        except Exception as e:
            log(f"An error occurred while processing subreddit '{subreddit_name}': {e}")

    # Fetch and analyze financial news from two days ago
    fetch_and_analyze_news(news_api_key, two_days_before_today, keywords, 0.75, 0.5)

    # Fetch and analyze Google news from two days ago
    fetch_and_analyze_google_news(google_news_api_key, two_days_before_today, keywords, 0.75, 0.5)

    # Correlate sentiment data with next-day price changes and volatility and store results
    correlation_results = correlate_sentiment_with_price_and_volatility()
    for result in correlation_results:
        log(f"Date: {result[0]}, Polarity: {result[1]}, Confidence: {result[2]}, Price Change: {result[3]}, Volatility: {result[4]}")

    # Close database connection
    if conn:
        cur.close()
        conn.close()
        log("Database connection closed")
