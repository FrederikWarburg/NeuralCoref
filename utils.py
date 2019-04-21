from nltk.tokenize import word_tokenize
import pandas as pd
import numpy as np
import spacy

def custom_tokenizer(txt):
    tokens= word_tokenize(txt)
    new_tokens = []
    for token in tokens:
        if '-' in token:
            print(token)
            tmp = token.replace(' ','')
            tmp = tmp.split('-')
            for i, word in enumerate(tmp):
                if i == 0:
                    new_tokens.append(tmp[0])
                else:
                    new_tokens.append('-')
                    new_tokens.append(tmp[i])
        else:
            new_tokens.append(token)
    
    return new_tokens


def get_cluster_num(span_int, prediction):
    cluster_belongs = []
    for i, cluster in enumerate(prediction['clusters']):
        for span in cluster:
            if span_int == span[0]:
                cluster_belongs.append(i)
    return cluster_belongs

def get_span_from_offset(txt, offset):
    
    for token in spans(txt):
        
        if offset == token[2]:
            return token[0]

def spans(txt):
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(txt)
    offset = 0
    for token_count, token in enumerate(doc):
        token = str(token)
        #for token_count, token in enumerate(tokens):
        offset = txt.find(token, offset)
        yield token_count, token, offset, offset+len(token)
        offset += len(token)