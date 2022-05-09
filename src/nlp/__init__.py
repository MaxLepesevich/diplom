import pandas as pd
import re
import dataclasses
from dataclasses import dataclass
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
# nltk.download('vader_lexicon')
# nltk.download('stopwords')

@dataclass
class DfParser():
    def __init__(self, path: str):
        self.raw_data = pd.read_csv(path)
    
    def get_df_for_productId(self, id_in_list: int = 0):
        title_data = self.raw_data[['ProductId', 'UserId','Summary', 'Text']].dropna()
        dict_id = dict(self.raw_data["ProductId"].value_counts()[:100])

        list_id = [id for id in dict_id]
        df = title_data.loc[title_data["ProductId"] == list_id[id_in_list]]
        
        return df
    
    @classmethod
    def df_processing(cls, df: pd.DataFrame):
        df['Text'] = df['Text'].apply(lambda x:re.sub(r"<br .>|<a.+>","",x))
        df['Text'] = df['Text'].apply(lambda x:re.sub(r"http\S+", "", x))
        df['Text'] = df['Text'].apply(lambda x:re.sub(r"[0-9]+", "", x))
        df['Text'] = df['Text'].apply(lambda x:re.sub(r"/|$|(|)|[|]", "", x))
        df['Text'] = df['Text'].apply(lambda x:re.sub(r'.\".+\"', "", x))
        df['Text'] = df['Text'].apply(lambda x:x.replace("...",""))
        df['Text'] = df['Text'].apply(lambda x:x.replace(".",". "))
        df['Text'] = df['Text'].apply(lambda x:x.replace("=",""))
        
        return df
    @classmethod
    def make_df_with_optional_columns(cls, df: pd.DataFrame, 
                                            columns_list: list = ['ProductId', 'UserId',
                                                                  'Summary', 'Text']):
    
        return df[columns_list]
