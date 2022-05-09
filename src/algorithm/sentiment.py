import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import dataclasses
from dataclasses import dataclass

# nltk.download('vader_lexicon')
@dataclass
class SentimentAnalyzer:
    def __init__(self, boost=None, default_sentiment=0.5) -> None:
        self.model = SIA()
        self.default_sentiment = default_sentiment
        if boost:
            self.model.lexicon.update(boost)

    @classmethod
    def load_boost(cls, path):
        with open(path, 'r') as f:
            return json.load(f)["boost"]

    @classmethod
    def null_sentiment(cls, scores):
        return scores['neg'] + scores['neu'] + scores['pos'] == 0

    def sentiment(self, sent):
        scores = self.model.polarity_scores(sent)

        if self.null_sentiment(scores):
            return self.default_sentiment

        return scores['compound']