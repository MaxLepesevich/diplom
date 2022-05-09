import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.algorithm import sentiment

def test_SentimentAnalyzer():
    model = sentiment.SentimentAnalyzer()

    boost = sentiment.SentimentAnalyzer.load_boost('data/vader_boost.json')
    model = sentiment.SentimentAnalyzer(boost, default_sentiment=0.123)
