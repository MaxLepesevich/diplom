import os
import sys
import json
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.algorithm import sentiment, make_vader_sentiment
from src import nlp
from json import JSONEncoder
import dataclasses


class DataclassJSONEncoder(JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)

df_parser = nlp.DfParser("data/Reviews.csv")

df = df_parser.get_df_for_productId(78)
df = df_parser.df_processing(df)
df = df_parser.make_df_with_optional_columns(df, ['UserId','Text'])

vader_sent = make_vader_sentiment.ParameterCount(df)
id_dict = vader_sent.make_user_id()
without_sentiment = vader_sent.make_word_list(id_dict)


save_path = 'without_sentiment.json'

with open(save_path, 'w', encoding='utf8') as f:
    json.dump(without_sentiment, f,
        indent=4,
        ensure_ascii=False,
        cls=DataclassJSONEncoder)