
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src.algorithm import db_connection, make_vader_sentiment
from src import nlp

df_parser = nlp.DfParser("data/Reviews.csv")

df = df_parser.get_df_for_productId(78)
df = df_parser.df_processing(df)
df = df_parser.make_df_with_optional_columns(df, ['UserId','Text', "ProductId"])

productId = str(list(df["ProductId"])[0])

postgreDB = db_connection.PostgreConnection()
conn, cur = postgreDB.connection()
id = postgreDB.put_product_id(conn, cur, productId)

vader_sent = make_vader_sentiment.ParameterCount(df)
id_dict = vader_sent.make_user_id()

without_sentiment = vader_sent.make_word_list(id_dict)

list_of_sentences_id = postgreDB.put_sentiment_data(conn, cur, without_sentiment, id)

postgreDB.put_sentences_info(conn, cur, without_sentiment, list_of_sentences_id)