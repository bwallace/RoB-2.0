
# coding: utf-8

import pandas as pd
import spacy
nlp = spacy.load('en')
import fuzzywuzzy
from fuzzywuzzy import fuzz

data_path = "data/RoB-data.csv"
tp = pd.read_csv(data_path, chunksize=10000)
df = pd.concat(tp, ignore_index=True)
df = df.dropna(subset=["fulltext"]) # drop rows where we don't have full texts
df = df.drop(['Unnamed: 0'], axis=1)

def get_col_names(domain_str):
    return (domain_str + "-judgment", domain_str + "-rationale")  

def get_quote(rationale_str):
    Q = "Quote:"
    if Q in rationale_str:
        # annoying but sometimes there are different quote chars used (?!)
        # so we normalize here. hacky.
        rationale_str = rationale_str.replace('“', '"').replace('”', '"')
        if not '"' in rationale_str:
            print("no rationale string! {0}".format(rationale_str))
            return None 
        try:
            return rationale_str[rationale_str.index(Q):].split('"')[1]
        except:
            import pdb; pdb.set_trace()
    return None 

def is_sent_match(rationale, candidate, threshold=90, min_k=5):
    # assume rationales need to be at least k words.
    if len(candidate.split(" ")) < min_k:
        return False 
    
    return fuzz.token_sort_ratio(rationale, candidate) >= threshold


'''
 'Random sequence generation-judgment',
 'Random sequence generation-rationale',
 'Allocation concealment-judgment',
 'Allocation concealment-rationale',
 'Blinding of participants and personnel-mortality-judgment',
 'Blinding of participants and personnel-mortality-rationale',
 'Blinding of participants and personnel-objective-judgment',
 'Blinding of participants and personnel-objective-rationale',
 'Blinding of participants and personnel-subjective-judgment',
 'Blinding of participants and personnel-subjective-rationale',
 'Blinding of participants and personnel-all-judgment',
 'Blinding of participants and personnel-all-rationale',
 'Blinding of outcome assessment-mortality-judgment',
 'Blinding of outcome assessment-mortality-rationale',
 'Blinding of outcome assessment-objective-judgment',
 'Blinding of outcome assessment-objective-rationale',
 'Blinding of outcome assessment-subjective-judgment',
 'Blinding of outcome assessment-subjective-rationale',
 'Blinding of outcome assessment-all-judgment',
 'Blinding of outcome assessment-all-rationale',
'''


'''
Consume RoB data in the CSV; convert to RA-CNN style data.
'''
MAX_FT_LEN = 15000 # covers 99%+ of cases; there is one outlier with 2902523, which breaks things...
def convert_df_to_training_data(path="RoB_data.csv", study_range=None):
    

    domain_name_map = {"bpp":"Blinding of participants and personnel", 
                       "rsg":"Random sequence generation",
                       "ac":"Allocation concealment",
                       "boa":"Blinding of outcome assessment"}
                       
    
    # here we construct a dictionary to be converted to a DataFrame
    # for output.
    # note that RSG and AC are only overall.
    d = {"pmid":[], "doi":[], "doc_id": [], "sentence":[], 
         "rsg-rationale":[], "rsg-doc-judgment":[],
         "ac-rationale":[], "ac-doc-judgment":[], 
         "bpp-rationale-all":[], "bpp-doc-judgment-all":[],
         "bpp-rationale-mortality":[], "bpp-doc-judgment-mortality":[],
         "bpp-rationale-objective":[], "bpp-doc-judgment-objective":[],
         "bpp-rationale-subjective":[], "bpp-doc-judgment-subjective":[],
         "boa-rationale-all":[], "boa-doc-judgment-all":[],
         "boa-rationale-mortality":[], "boa-doc-judgment-mortality":[],
         "boa-rationale-objective":[], "boa-doc-judgment-objective":[],
         "boa-rationale-subjective":[], "boa-doc-judgment-subjective":[]}
         
    outcome_categories = ["mortality", "objective", "subjective", "all"]
    
    rows_to_process = list(df.iterrows())
    for index, row in rows_to_process:
        if (index % 50) == 0:
            print ("on study {0}".format(index))
            
        full_text = row["fulltext"][:MAX_FT_LEN]
 
        document = nlp(full_text)
        sentences = list(document.sents)
        is_rationale = []
        for sent in sentences:
            sent = sent.string
            cur_pmid = row["pmid"]
            if pd.isnull(cur_pmid):
                cur_pmid = 0

            cur_doi = row["doi"]
            if pd.isnull(cur_doi):
                cur_doi = "missing"

            if cur_pmid == 0 and cur_doi == "missing":
                import pdb; pdb.set_trace()
            elif cur_pmid != 0:
                doc_id = str(int(cur_pmid))
            else:
                doc_id = cur_doi 

            d["doi"].append(cur_doi)
            d["pmid"].append(cur_pmid) 
            d["sentence"].append(sent)
            d["doc_id"].append(doc_id)

            for abbrv, domain in list(domain_name_map.items()):
                domain_rationale, rationale_field_key = None, None
                if abbrv in ["rsg", "ac"]:
                    # simple case; only overall judgment
                    judgment_col, rationale_col = get_col_names(domain)
                    if not pd.isnull(row[rationale_col]):
                        domain_rationale = get_quote(row[rationale_col])
                    domain_judgment = row[judgment_col]
                    
                    domain_field_key = abbrv + "-doc-judgment"
                    d[domain_field_key].append(domain_judgment)
                    
                    rationale_field_key = abbrv + "-rationale"
                    if not (domain_rationale is None) and is_sent_match(domain_rationale, sent):
                        d[rationale_field_key].append(1)
                    else:
                        d[rationale_field_key].append(0)
                                
                else:
                    # more complicated, need to loop over outcome
                    # categories/types
                    for outcome_type in outcome_categories:
                        domain_rationale, rationale_field_key = None, None
                        domain_plus_outcome = domain + "-" + outcome_type
                        judgment_col, rationale_col = get_col_names(domain_plus_outcome)
                        #if "Patients were assessed" in sent:
                        #    import pdb; pdb.set_trace()
                        if not pd.isnull(row[rationale_col]):
                            domain_rationale = get_quote(row[rationale_col])
                        domain_judgment = row[judgment_col]

                        domain_field_key = abbrv + "-doc-judgment-" + outcome_type
                        d[domain_field_key].append(domain_judgment)
                    
                        rationale_field_key = abbrv + "-rationale-" + outcome_type
                        if not (domain_rationale is None) and is_sent_match(domain_rationale, sent):
                            d[rationale_field_key].append(1)
                        else:
                            d[rationale_field_key].append(0)
                            
    
    return pd.DataFrame(d)



def put_together(outpath="data/RoB-data-2-all.csv"):
    increment_by = 500
    cur_start, cur_end = 0, 500
    N = 28428

    df = None
    while cur_start < N: 
        # print ("cur_start: {0}; cur_end: {1}")
        cur_df = pd.read_csv("data/RoB-data-2-{0}--{1}.csv".format(cur_start, cur_end))

        if df is None:
            df = cur_df
        else:
            df = pd.concat([df, cur_df])
        
        cur_start += increment_by
        cur_end += increment_by

    df.to_csv(outpath)

def main():
    formatted_data = convert_df_to_training_data()
    formatted_data.to_csv("RoB-data-4.csv")
    

    # create a doc_id category: this is PMID where available, else DOI
    # if both unavailable (???) drop
    # ASSUMPTION: no PMID -> not in PMID
    formatted_data.to_csv("RoB-data-4.csv")

    '''
    increment_by = 500
    cur_start, cur_end = 0, 500
    N = 28432

    while cur_start < N: 
        # print ("cur_start: {0}; cur_end: {1}")
        formatted_data = convert_df_to_training_data(study_range=[cur_start, cur_end])
        formatted_data.to_csv("data/RoB-data-2-{0}--{1}.csv".format(cur_start, cur_end))
        cur_start += increment_by
        cur_end += increment_by
    '''



def train_dev_test_split(RA_CNN_data_path="data/RoB-data-3-all.csv"):
    ####
    #  First pull out duplicates. Priority is to identify based on PMIDs, then default to DOI. 
    ####
    # df is the original data file that we formatted! 
    all_data = pd.read_csv(RA_CNN_data_path) 



def get_duplicate_ids():
    '''
    Assemble data for testing
    '''
    orig_df = pd.read_csv("data/RoB-data-w-uids.csv")
    duplicate_uids = []
    for idx, row in orig_df.iterrows():
        cur_cdno = row['cdno']
        cur_uid = row['uid']
        # duplicates are studies w/same uid but in *different*
        # review
        cur_duplicates = orig_df[(orig_df['uid'] == cur_uid) & (orig_df['cdno'] != cur_cdno)]
        if len(cur_duplicates) > 0:
            duplicate_uids.append(cur_uid)

    return list(set(duplicate_uids))

if __name__ == "__main__": 
    main()

