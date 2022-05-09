import pandas as pd
import nltk
import re
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# nltk.download('vader_lexicon')
# nltk.download("maxent_ne_chunker")
# nltk.download('words')
# nltk.download('treebank')
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
def preprocess_without_lem(sentence):
    """
    Преобразует предложение в список слов без stop_words    
    """
    stop_words = nltk.corpus.stopwords.words('english') + ["'s", 'uh','would', "um", "ha",'ah', 
                                                           ".","n't","?","’",",","'m","oh",'ot',
                                                           ":","u","'re", 'a', 'um', 'into', 'to',
                                                           'say', "um", "overall","!","&",""]
    a = ['no', 'not', 'only',"aren't", "couldn't", "didn't","doesn't",
         "hadn't","hasn't","haven't","isn't","mightn't","mustn't","needn't",
         "shouldn't", "wasn't", "weren't", "won't", "wouldn't",'more', 
         'most', 'other', 'some', 'such', 'very']
    stop_words = [i for i in stop_words if i not in a]
    output = []
    sen = []
    for sent in nltk.sent_tokenize(sentence):
        sen.append(sent)
        temp = []
        for word in nltk.word_tokenize(sent):
            if word.lower() not in stop_words:
                temp.append(word)
        output.append(temp)
    return output, sen

def putDataInJson(path, output):
    # Запись данных в json файл
    with open(path, 'w') as outfile:
        json.dump(output, outfile, indent = 2)

raw_data = pd.read_csv("D:\Programming\education\Reviews.csv")

dict_id = dict(raw_data["ProductId"].value_counts()[2:3])

list_id = [id for id in dict_id]

title_data = raw_data[['ProductId', 'UserId','Summary', 'Text']].dropna()
df = title_data.loc[title_data["ProductId"] == list_id[0]]

df["Text"] = df["Text"].apply(lambda x:re.sub('<br .>|<a.+>','',x))

df["Text"] = df["Text"].apply(lambda x:re.sub(r"http\S+", "", x))

# df['Summary'] = df['Summary'].apply(lambda x:re.sub('<br .>|<a.+>','',x))

# df['Summary'] = df['Summary'].apply(lambda x:re.sub(r"http\S+", "", x))


df = df[["Text",'UserId']]

slow = {}

sid = SIA()


for i, val in df.iterrows():
    text = val["Text"].replace(".",". ")
    slow[val["UserId"]] = {"sum":0, "text":text}
    text, sen = preprocess_without_lem(text)
    
    slow[val["UserId"]]["info"] = []
    sum = 0
    for index, sent in enumerate(text):
        sum+=sid.polarity_scores(sen[index])["compound"]

        slow[val["UserId"]]["info"] += [{"text_list":sent,
                                 "text":sen[index],
                                 "sentiment":sid.polarity_scores(sen[index])["compound"]}]
    slow[val["UserId"]]["sum"] = sum
    if sum > 0:
        slow[val["UserId"]] = 0 

# print(slow)
putDataInJson("sent.json", slow)

# sid = SIA()
# body_data['sentiments']           = body_data['body'].apply(lambda x: sid.polarity_scores(' '.join(re.findall(r'\w+',x.lower()))))
# body_data['Positive Sentiment']   = body_data['sentiments'].apply(lambda x: x['pos']+1*(10**-6)) 
# body_data['Neutral Sentiment']    = body_data['sentiments'].apply(lambda x: x['neu']+1*(10**-6))
# body_data['Negative Sentiment']   = body_data['sentiments'].apply(lambda x: x['neg']+1*(10**-6))

