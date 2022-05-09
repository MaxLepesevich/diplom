import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
import psycopg2
from dataclasses import dataclass


@dataclass
class PostgreConnection():
    def __init__(self, dbname ='diplom', user ='postgres', 
                password = '*****', host = 'localhost'):
        self.dbname = dbname
        self.user = user 
        self.password = password
        self.host = host

    def connection(self):
        conn = psycopg2.connect(dbname = self.dbname, user = self.user,
                                password = self.password, host = self.host)
        
        cur = conn.cursor()
        return conn, cur

    @classmethod
    def preprocess_line(cls, sentence):
        list_words = "'"
        if len(sentence.list_of_words) == 0:
            return list_words + "'"
        for ind, word in enumerate(sentence.list_of_words):
            list_words += word.replace("'","’")

            if ind == len(sentence.list_of_words)-1:
                list_words += "'"
            else: 
                list_words += "','"

        return list_words

    @classmethod
    def put_product_id(cls, conn, cur, productId):
        list_of_product_id = []
        id = 0
        
        cur.execute('SELECT * FROM PRODUCT')
        rows = cur.fetchall()
        
        if len(rows) != 0:
            list_of_product_id = [x[1] for x in rows]
            id = rows[-1][0]+1
        
        if productId not in list_of_product_id:
            cur.execute("INSERT INTO PRODUCT VALUES ({},'{}')".format(id, productId))
            conn.commit()
        else:
            for prod_name in rows:
                if productId == prod_name[1]:
                    return prod_name[0] 
        return id
    
    @classmethod
    def put_sentiment_data(cls, conn, cur, without_sentiment, id):
        list_of_product_id = []
        sentiment_data_id = 0

        cur.execute('SELECT * FROM SENTIMENT_DATA')
        rows = cur.fetchall()

        if len(rows) != 0:
            sentiment_data_id = rows[-1][0]+1
            list_of_product_id = [x[1] for x in rows]
        list_of_sentences_id = []
        if id not in list_of_product_id:
            for val in without_sentiment:
                cur.execute("INSERT INTO SENTIMENT_DATA VALUES ({0},{1},'{2}',{3},'{4}')".format(sentiment_data_id,
                                                                                                 id,
                                                                                                 val,
                                                                                                 without_sentiment[val].total_sum,
                                                                                                 without_sentiment[val].answer_text.replace("'","’"))
                    )
                list_of_sentences_id.append(sentiment_data_id)
                sentiment_data_id+=1
            conn.commit()
        return list_of_sentences_id

    @classmethod
    def put_sentences_info(cls, conn, cur, without_sentiment, list_of_sentences_id):
        list_of_product_id = []
        sentences_info_id = 0

        cur.execute('SELECT * FROM SENTENCES_INFO')
        rows = cur.fetchall()

        if len(rows) != 0:
            sentences_info_id = rows[-1][0]+1
            list_of_product_id = [x[1] for x in rows]

        for index, val in enumerate(without_sentiment):
            for sentence in without_sentiment[val].sentences:    
                list_of_words = PostgreConnection.preprocess_line(sentence)                    
                sql_line = "INSERT INTO SENTENCES_INFO VALUES ({},{},'{}',{}, ARRAY [".format(sentences_info_id,
                                                                                              list_of_sentences_id[index],
                                                                                              sentence.sentence_text.replace("'","’"),
                                                                                              sentence.sentiment)+list_of_words+"])"
                sentences_info_id+=1
                cur.execute(sql_line)
        conn.commit()