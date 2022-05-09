
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import nltk
import pandas as pd
from dataclasses import dataclass, field
from algorithm.sentiment import SentimentAnalyzer

@dataclass
class Preprocess():

    def __init__(self) -> None:
        pass

    def make_stop_words():
        stop_words = nltk.corpus.stopwords.words('english')
        stop_words += [ "'s", "uh","would", "um", "ha","ah", 
                        ".","?","’",",","'m","oh","ot",":",
                        "u","'re","a","um","into","to",'say',
                        "um","overall","!","&","=","(",")",
                        "'d","[","]"]
        good_words = [ "no", "not", "only","aren't","couldn't", "didn't","doesn't", "can't",
                       "hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't",
                       "shouldn't", "wasn't", "weren't", "won't", "wouldn't","more", 
                       "most", "other", "some", "such", "very"]
        stop_words = [i for i in stop_words if i not in good_words]
        return stop_words
    
    def preprocess_sentences(sentence:list, stop_words:list):
        output = []
        full_sentences = []
        for sent in nltk.sent_tokenize(sentence):
            full_sentences.append(sent)
            temp = [word for word in nltk.word_tokenize(sent) 
                            if word.lower() not in stop_words]
            output.append(temp)
        return output, full_sentences

@dataclass
class Sentence():
    sentence_text: str = ''
    sentiment: float = 0.0
    list_of_words: list = field(default_factory=[])
    
@dataclass
class Analiz():
    total_sum: float = 0.0
    answer_text: str = ''
    sentences: list = field(default_factory=[])

@dataclass
class ParameterCount():
    def __init__(self, df: pd.DataFrame):
        self.df = df
    

    def make_user_id(self):
        output = {}
        for i, val in self.df.iterrows():
            output[val["UserId"]] = Analiz(0,val["Text"],[])
        return output  
    
    def make_sentiment(text: str, model):
        return model.sentiment(text)


    def make_word_list(self, id_dict:dict):
        model = SentimentAnalyzer()
        boost = SentimentAnalyzer.load_boost('data/vader_boost.json')
        model = SentimentAnalyzer(boost, default_sentiment=0.123)
        for id in id_dict:
            stop_words = Preprocess.make_stop_words()
            output, full_sentences = Preprocess.preprocess_sentences(id_dict[id].answer_text, stop_words)
            
            list_of_sentences = []
            sum = 0
            for index, list_of_words in enumerate(output):
                sentiment = ParameterCount.make_sentiment(full_sentences[index], model)
                sum += sentiment
                list_of_sentences.append(Sentence(full_sentences[index],sentiment,list_of_words=list_of_words))
            id_dict[id].sentences = list_of_sentences
            id_dict[id].total_sum = sum/len(output)
        return id_dict 

    