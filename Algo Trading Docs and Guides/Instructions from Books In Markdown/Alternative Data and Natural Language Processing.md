# Alternative Data and Natural Language Processing

**Leveraging Alternative Data Sources for Trading Signals**

---

## Introduction

Alternative data provides informational edge in increasingly efficient markets. This document covers techniques for extracting signals from text, news, social media, and other non-traditional sources.

---

## Sentiment Analysis

### News Sentiment

```python
from textblob import TextBlob
import pandas as pd

def analyze_news_sentiment(headlines):
    """
    Analyze sentiment from news headlines
    
    Parameters:
    -----------
    headlines : pd.Series
        News headlines with datetime index
    
    Returns:
    --------
    pd.DataFrame : Sentiment scores
    """
    sentiments = []
    
    for date, headline in headlines.items():
        blob = TextBlob(headline)
        
        sentiments.append({
            'date': date,
            'headline': headline,
            'polarity': blob.sentiment.polarity,  # -1 to 1
            'subjectivity': blob.sentiment.subjectivity  # 0 to 1
        })
    
    return pd.DataFrame(sentiments)

# Aggregate to daily
def aggregate_daily_sentiment(sentiment_df):
    """Aggregate sentiment to daily frequency"""
    daily = sentiment_df.groupby(sentiment_df['date'].dt.date).agg({
        'polarity': ['mean', 'std', 'min', 'max'],
        'subjectivity': 'mean'
    })
    
    return daily
```

### Advanced NLP with Transformers

```python
from transformers import pipeline

# Load pre-trained sentiment model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert"
)

def finbert_sentiment(texts):
    """
    Analyze sentiment using FinBERT
    
    Parameters:
    -----------
    texts : list
        List of text strings
    
    Returns:
    --------
    pd.DataFrame : Sentiment predictions
    """
    results = sentiment_pipeline(texts)
    
    df = pd.DataFrame(results)
    df['score'] = df.apply(
        lambda x: x['score'] if x['label'] == 'positive' else -x['score'],
        axis=1
    )
    
    return df
```

---

## Social Media Signals

### Twitter Sentiment

```python
import tweepy

def fetch_twitter_sentiment(symbol, n_tweets=100):
    """
    Fetch and analyze Twitter sentiment
    
    Parameters:
    -----------
    symbol : str
        Stock symbol
    n_tweets : int
        Number of tweets to fetch
    
    Returns:
    --------
    dict : Sentiment statistics
    """
    # Fetch tweets (requires API credentials)
    tweets = api.search_tweets(q=f"${symbol}", count=n_tweets)
    
    # Analyze sentiment
    sentiments = []
    for tweet in tweets:
        blob = TextBlob(tweet.text)
        sentiments.append(blob.sentiment.polarity)
    
    return {
        'mean_sentiment': np.mean(sentiments),
        'sentiment_std': np.std(sentiments),
        'positive_ratio': sum(s > 0 for s in sentiments) / len(sentiments),
        'tweet_volume': len(tweets)
    }
```

### Reddit Analysis

```python
import praw

def analyze_reddit_sentiment(subreddit_name, symbol):
    """
    Analyze Reddit sentiment for a stock
    
    Parameters:
    -----------
    subreddit_name : str
        Subreddit name (e.g., 'wallstreetbets')
    symbol : str
        Stock symbol
    
    Returns:
    --------
    dict : Sentiment and engagement metrics
    """
    reddit = praw.Reddit(...)  # Initialize with credentials
    
    subreddit = reddit.subreddit(subreddit_name)
    
    posts = []
    for post in subreddit.search(symbol, limit=100):
        posts.append({
            'title': post.title,
            'score': post.score,
            'num_comments': post.num_comments,
            'created': post.created_utc
        })
    
    df = pd.DataFrame(posts)
    
    # Analyze sentiment
    df['sentiment'] = df['title'].apply(
        lambda x: TextBlob(x).sentiment.polarity
    )
    
    return {
        'mean_sentiment': df['sentiment'].mean(),
        'total_engagement': df['score'].sum() + df['num_comments'].sum(),
        'post_volume': len(df)
    }
```

---

## Web Scraping

### SEC Filings

```python
import requests
from bs4 import BeautifulSoup

def scrape_sec_filing(cik, filing_type='10-K'):
    """
    Scrape SEC filing
    
    Parameters:
    -----------
    cik : str
        Company CIK number
    filing_type : str
        Filing type (10-K, 10-Q, 8-K, etc.)
    
    Returns:
    --------
    str : Filing text
    """
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type={filing_type}"
    
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract filing text
    # (Implementation depends on specific filing format)
    
    return filing_text
```

### Earnings Call Transcripts

```python
def analyze_earnings_call(transcript):
    """
    Analyze earnings call transcript
    
    Parameters:
    -----------
    transcript : str
        Earnings call transcript text
    
    Returns:
    --------
    dict : Analysis results
    """
    # Sentiment analysis
    sentiment = TextBlob(transcript).sentiment.polarity
    
    # Keyword extraction
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vectorizer = TfidfVectorizer(max_features=20, stop_words='english')
    tfidf = vectorizer.fit_transform([transcript])
    keywords = vectorizer.get_feature_names_out()
    
    # Tone analysis (uncertainty, confidence)
    uncertainty_words = ['uncertain', 'maybe', 'possibly', 'might']
    confidence_words = ['confident', 'certain', 'definitely', 'strong']
    
    uncertainty_score = sum(transcript.lower().count(w) for w in uncertainty_words)
    confidence_score = sum(transcript.lower().count(w) for w in confidence_words)
    
    return {
        'sentiment': sentiment,
        'keywords': keywords.tolist(),
        'uncertainty': uncertainty_score,
        'confidence': confidence_score
    }
```

---

## Satellite Imagery

### Parking Lot Analysis

```python
from PIL import Image
import numpy as np

def count_cars_in_parking_lot(image_path):
    """
    Estimate car count from satellite image
    
    Parameters:
    -----------
    image_path : str
        Path to satellite image
    
    Returns:
    --------
    int : Estimated car count
    """
    # Load image
    img = Image.open(image_path)
    img_array = np.array(img)
    
    # Use computer vision model to detect cars
    # (Requires pre-trained object detection model)
    
    # Example using a hypothetical detector
    # car_count = car_detector.detect(img_array)
    
    return car_count

def parking_lot_occupancy_signal(car_counts, historical_avg):
    """
    Generate trading signal from parking lot occupancy
    
    Parameters:
    -----------
    car_counts : pd.Series
        Time series of car counts
    historical_avg : float
        Historical average car count
    
    Returns:
    --------
    pd.Series : Trading signals
    """
    # Calculate deviation from average
    deviation = (car_counts - historical_avg) / historical_avg
    
    # Generate signal
    signals = pd.Series(index=car_counts.index)
    signals[deviation > 0.1] = 1  # Buy signal
    signals[deviation < -0.1] = -1  # Sell signal
    signals.fillna(0, inplace=True)
    
    return signals
```

---

## Best Practices

1. **Validate alternative data** - Check for quality and consistency
2. **Combine multiple sources** - Don't rely on single data source
3. **Account for lag** - Alternative data may not be real-time
4. **Test for alpha decay** - Signals may become less effective over time
5. **Consider costs** - Alternative data can be expensive
6. **Legal compliance** - Ensure data usage is legal and ethical

---

## References

1. Jansen, S. (2020). *Machine Learning for Algorithmic Trading*. Packt Publishing. Chapters 3-4.
2. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
