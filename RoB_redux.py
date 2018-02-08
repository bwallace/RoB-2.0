

import pandas as pd 
import spacy
nlp = spacy.load('en')


def get_col_names(domain_str):
    return (domain + "-judgment", domain + "-rationale")    

def load_RoB_df(path_to_csv):
    tp = pd.read_csv(path_to_csv, chunksize=10000)
    df = pd.concat(tp, ignore_index=True)
    return df 

'''
Consume RoB data in the CSV; convert to RA-CNN style data.
'''
def convert_df_to_training_data(df, domain="Random sequence generation"):
    judgment_col, rationale_col = get_col_names(domain)

    for index, row in df.iterrows():
        full_text = row["fulltext"]
        document = nlp(full_text)
        sentences = list(document.sents)
        rationale_lbls = []
        for sent_idx, sent in enumerate(sentences):
            is_rationale = 


path_to_csv = "RoB_data.csv"
df = load_RoB_df(path_to_csv)