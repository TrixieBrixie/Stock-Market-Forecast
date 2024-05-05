import praw # To intercat with Reddit
import yfinance as yf
from transformers import pipeline, BertTokenizer # For sentiment analysis 
import psycopg2 # Database operations
import datetime
import re
import time
from praw.exceptions import APIException, ClientException
import requests

# Create a Reddit instance with credentials to interact with Reddit platform
reddit = praw.Reddit(
    client_id='13wtVcBB17S_GSs0aVZWJA',
    # Create a password-like string used to authenticate the application's identity when accessing the Reddit API
    client_secret='4Z_oN2ty4_87Q0tIeQpRZ9CKV40Agg',
    user_agent='script:reddit_scraper:v1.0 (by /u/TrixieBrixie)',
    username='TrixieBrixie',
    password='251077Giorgia'
)

# Create database connection 
conn = psycopg2.connect(
    host='localhost',
    database='redditsentiment',
    user='postgres',
    password='080111'
)
# Create a control structure to transfer datat to the database.
cur = conn.cursor()

# Initialize a sentiment analysis pipeline using a pre-trained BERT model. 
# This pipeline will be used to analyze the sentiment of texts.
classifier = pipeline('sentiment-analysis') 
'''
pipeline function is designed to be versatile and can be used for different natural language processing (NLP) tasks 
such as text classification, named entity recognition, question answering, etc.
'''

# Create a function to clean text from Reddit by making it lowercase, 
# removing URLs, mentions, numbers, punctuation, and extra spaces
def clean_text(text):
    text = text.lower()  # Lowercase text
    text = re.sub(r'https?://\S+', ' ', text)  # Remove URLs
    text = re.sub(r'@\w+', ' ', text)  # Remove mentions
    text = re.sub(r'\d+', ' ', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

# Load a pre-trained BERT model fine-tuned for sentiment analysis
classifier = pipeline('sentiment-analysis')

# Create a function that cleans the text and then performs sentiment analysis, 
# returning both the weighted polarity (based on sentiment and its confidence) and the sentiment's confidence score.
def analyze_sentiment_bert(text):
    cleaned_text = clean_text(text)
    # Create a dictionary to store the sentiment classification result (label), and the score (score) of it
    # E.g. [{'label': 'POSITIVE', 'score': 0.9997}]
    result = classifier(cleaned_text) 
    # Extract sentiment and confidence
    sentiment_label = result[0]['label']
    sentiment_confidence = result[0]['score']
    
    # Interpret polarity based on label with confidence as weight
    polarity = 1 if sentiment_label == 'POSITIVE' else -1
    weighted_polarity = polarity * sentiment_confidence  # This will weight the polarity by confidence
    '''
    Weighted Polarity: This value represents the sentiment direction (positive or negative). 
    It combines both the direction and the strength (confidence) of the sentiment into a single measure.
    Sentiment Confidence: This score tells us how confident the model is about its sentiment prediction. 
    A high confidence score indicates that the model is very sure about its classification, 
    whereas a lower score suggests uncertainty.'''
    return weighted_polarity, sentiment_confidence

# Fecht FTSE data from the tow days beofore the current date
def fetch_ftse100_data():
    try:
        # Get the rtarget date
        target_date = datetime.datetime.now().date() - datetime.timedelta(days=2)

        # Convert the date to string format
        target_date_str = target_date.strftime("%Y-%m-%d")
        print(f"Fetching FTSE 100 data for: {target_date_str}")

        # Fetch intraday data for the current date
        ftse100_intraday = yf.download("^FTSE", target_date_str)
         # Check if any data was returned
        if ftse100_intraday.empty:
            print("No data returned for FTSE 100 on the specified date.")
        else:
            print(ftse100_intraday)  # Print the data fetched

        return ftse100_intraday
    
    except Exception as e:
        print(f"Failed to fetch FTSE 100 data: {e}")
        return None

# Create a function to fetch 100 posts from two day before running the code, 
# analyze their titles for sentiment, and store the results in the database 
# if the sentiment confidence and polarity_threshold exceed the thresholds
def fetch_and_analyze_subreddit(subreddit_name, keywords, confidence_threshold=0.75, polarity_threshold=0.5):
    reddit = praw.Reddit(client_id='13wtVcBB17S_GSs0aVZWJA', client_secret='4Z_oN2ty4_87Q0tIeQpRZ9CKV40Agg', user_agent='House_Market:v1.0 (by /u/TrixieBrixie)')
    subreddit = reddit.subreddit(subreddit_name)
    remaining_requests = 30
    try:
        for post in subreddit.new(limit=100):
            post_title = post.title.lower()
            if remaining_requests <= 1:
                print("Approaching rate limit, sleeping for 60 seconds.")
                time.sleep(60)
                remaining_requests = 30

            print(f"Title: {post.title}")
            weighted_polarity, sentiment_confidence = analyze_sentiment_bert(post.title)
            if sentiment_confidence >= confidence_threshold and abs(weighted_polarity) > polarity_threshold:
                print(f"Sentiment: Weighted Polarity = {weighted_polarity}, Confidence = {sentiment_confidence}")
                timestamp = datetime.datetime.fromtimestamp(post.created_utc)  # Use created_utc for correct timestamp
                cur.execute(
                    'INSERT INTO sentimentdata (post_id, timestamp, title, polarity, confidence, subreddit) VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (post_id) DO NOTHING;',
                    (str(post.id), timestamp, post.title, weighted_polarity, sentiment_confidence, subreddit_name)
                )
                conn.commit()
                print(f"Stored: {post.title}")
                print(" ")
            else:
                print(f"Skipped low confidence sentiment for: {post.title}")
                print(" ")

            # Monitor rate limits from headers dynamically
            headers = post._reddit._core._requestor._http.headers
            remaining_requests = int(headers.get('x-ratelimit-remaining', 100))
            if remaining_requests < 5:
                print("Approaching rate limit, sleeping for 10 seconds.")
                time.sleep(10)
    except praw.exceptions.APIException as e:
        if e.error_type == "RATELIMIT":
            delay = int(e.message.split("minute")[0].split()[-1])
            print(f"Rate limited. Sleeping for {delay} minutes.")
            time.sleep(delay * 60)
    except praw.exceptions.ClientException as e:
        print(f"Client error: {e}")

if __name__ == '__main__':
    # Fetch and display FTSE 100 data
    ftse_data = fetch_ftse100_data()
    if ftse_data is not None:
        print("FTSE data fetched successfully.")
    else:
        print("Failed to fetch FTSE data.")
    
    # Fetch and analyze Reddit data
    keywords = ['FTSE 100', 'stock market', 'share prices', 'market volatility', 'equities', 'London Stock Exchange', 'UK stocks', 'dividends']
    fetch_and_analyze_subreddit('UKInvesting', keywords, 0.75, 0.5)

    # Close database connection
    cur.close()
    conn.close()


# psql -U postgres -d redditsentiment 

# SELECT * FROM sentimentdata;

# TRUNCATE TABLE sentimentdata;

# CREATE TABLE sentimentdata (
#     post_id VARCHAR NOT NULL,
#     timestamp TIMESTAMP WITHOUT TIME ZONE,
#     title TEXT,
#     polarity DOUBLE PRECISION,
#     subjectivity DOUBLE PRECISION,
#     subreddit VARCHAR,S
#     confidence DOUBLE PRECISION,
#     PRIMARY KEY (post_id)
# );


# Group data based on the polarity:
# SELECT polarity, COUNT(*) as count
# FROM sentimentdata
# GROUP BY polarity
# ORDER BY polarity;

# Calculate the average subjectivity for each sentiment group.
# SELECT polarity, AVG(subjectivity) as average_subjectivity
# FROM sentimentdata
# GROUP BY polarity
# ORDER BY polarity;

