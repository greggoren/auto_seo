import xml.etree.ElementTree as ET
import nltk.data
import pyndri
import numpy as np
from collections import Counter
import math
import params

index = pyndri.Index(params.path_to_index)


def load_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    docs={}
    for doc in root:
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                name=att.text
            else:
                docs[name]=att.text
    return docs



def retrieve_sentences(doc):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(doc)
    return sentences

def get_Dinit_for_query(query):
    Dinit=[]
    with open(params.path_to_data_set) as data_set:
        for data_row in data_set:
            if data_row.split()[0]==query:
                doc = data_row.split(" # ")[1]
                Dinit.append(doc)
        return Dinit

def get_tdidf_value_of_word(word, counts, N,number_of_terms, token2id, id2df):
    stemmed = pyndri.krovetz_stem(word)
    id = token2id[stemmed]
    return  get_tfidf_value(id,counts,N,number_of_terms,id2df),id

def get_tfidf_value(id,counts,N,number_of_terms,id2df):
    df = id2df[id]
    tf = counts[id]/number_of_terms
    idf = math.log(float(N) / df)
    return tf*idf

def create_tfidf_vectors(sentences):
    all_sentences=[]
    token2id, id2token, id2df = index.get_dictionary()
    N = index.document_count()
    for doc in sentences:
        terms = index.document(doc)[1]
        counts = Counter(terms)
        for sentence in sentence[doc]:
            words = sentence.split()
            sentence_vector = np.zeros(len(token2id))
            for word in words:
                tfidf,id = get_tdidf_value_of_word(word, counts, N, token2id, id2df)
                sentence_vector[id]+=tfidf
            sentence_vector/=len(words)
            all_sentences.append(sentence)
    return all_sentences

def transform_terms_to_counts(Dinit):
    Dinit_counts={}
    for d_i in Dinit:
        terms = index.document(d_i)[1]
        counts = Counter(terms)
        Dinit_counts[d_i]=counts
    return Dinit_counts

def retrieve_ranked_lists(ranked_lists_file):
    chosen_docs={}
    with open(ranked_lists_file) as ranked_lists:
        for ranked_doc in ranked_lists:
            query,doc = ranked_doc.split()[0],ranked_doc.split()[2]
            if not chosen_docs.get(query,False):
                chosen_docs[query]=[]
            chosen_docs[query].append(doc)
        return chosen_docs

def create_document_tf_id_vector(doc):
    terms = index.document(doc)[1]
    token2id, id2token, id2df = index.get_dictionary()
    doc_vector = np.zeros(len(token2id))
    number_docs = index.document_count()
    number_of_terms = len(terms)
    counts = Counter(terms)
    for term in set(terms):
        tfidf = get_tfidf_value(term,counts,number_docs,number_of_terms,id2df)
        doc_vector[term]+=tfidf
    return doc_vector

