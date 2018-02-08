
# coding: utf-8

import pandas as pd
import spacy
nlp = spacy.load('en')
import fuzzywuzzy
from fuzzywuzzy import fuzz

data_path = "/Users/byron/Dropbox/cochranetech/rob2/RoB_data.csv"
tp = pd.read_csv(data_path, chunksize=10000)
df = pd.concat(tp, ignore_index=True)

doc = nlp(df['fulltext'].values[0])
sentences = list(doc.sents)

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
def convert_df_to_training_data(path="RoB_data.csv", domain="Random sequence generation"):
    
    domain_name_map = {"bpp":"Blinding of participants and personnel", 
                       "rsg":"Random sequence generation",
                       "ac":"Allocation concealment",
                       "boa":"Blinding of outcome assessment"}
                       
    
    # here we construct a dictionary to be converted to a DataFrame
    # for output.
    # note that RSG and AC are only overall.
    d = {"pmid":[], "sentence":[], 
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
    
    for index, row in list(df.iterrows()):
        if (index % 10) == 0:
            print ("on study {0}".format(index))
            
        full_text = row["fulltext"]
 
        document = nlp(full_text)
        sentences = list(document.sents)
        is_rationale = []
        for sent in sentences:
            sent = sent.string
            d["pmid"].append(row["pmid"])
            d["sentence"].append(sent)
            
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


formatted_data = convert_df_to_training_data()
formatted_data.to_csv("RoB-data-2.csv")



