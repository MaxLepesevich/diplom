import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from src import nlp



def test_DfParser():
    df_parser = nlp.DfParser("data/Reviews.csv")

    df = df_parser.get_df_for_productId(5)
    df = df_parser.df_processing(df)
    df = df_parser.make_df_with_optional_columns(df,['Text'])
    print(df)

test_DfParser()