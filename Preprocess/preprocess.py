import xml.etree.ElementTree as ET
import nltk.data
import pyndri
import numpy as np
from collections import Counter
import math
import params
from params import beta

index = pyndri.Index(params.path_to_index)
token2id, id2token, id2df = index.get_dictionary()
id2tf = index.get_term_frequencies()
dic={}
total_corpus_term_count=0
doc_length = {}
for document_id in range(index.document_base(), index.maximum_document()):
    dic[index.document(document_id)[0]] = document_id
    doc_length[index.document(document_id)[0]] = index.document_length(document_id)
    total_corpus_term_count+=len(index.document(document_id))


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

def turn_sentence_into_terms(sentence):
    result = []
    words = index.tokenize(sentence)
    for word in words:
        result.append(token2id[pyndri.krovetz_stem(word)])
    return result




def retrieve_sentences(doc):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(doc)
    return sentences

def get_Dinit_for_query(query):
    Dinit=[]
    with open(params.path_to_data_set) as data_set:
        for data_row in data_set:
            if data_row.split()[1].split(":")[1]==query:
                doc = data_row.split(" # ")[1]
                Dinit.append(doc.rstrip())
        return Dinit

def get_tdidf_value_of_word(word, counts, N,number_of_terms):
    stemmed = pyndri.krovetz_stem(word)
    id = token2id[stemmed]
    return  get_tfidf_value(id,counts,N,number_of_terms),id

def get_tfidf_value(id,counts,N,number_of_terms):
    df = id2df[id]
    tf = counts[id]/number_of_terms
    idf = math.log(float(N) / df)
    return tf*idf

def split_Dinit_to_sentences(Dinit):
    token2id, id2token, id2df = index.get_dictionary()
    tree = ET.parse(params.trec_text_file)
    root = tree.getroot()
    Dinit_text = {}
    for doc in root:
        name = ""
        for att in doc:
            if att.tag == "DOCNO":
                name = att.text
            else:
                if name in Dinit:
                    Dinit_text[name] = Counter([token2id[pyndri.krovetz_stem(i)] for i in  retrieve_sentences(att.text)])
    return Dinit_text

def query_likelihood():
    pass

def query_probability_given_docs(query,Dinit_counts):
    query_to_doc_probability={}
    id2tf=index.get_term_frequencies()
    for d_i in Dinit_counts:
        counts = Dinit_counts[d_i]
        doc_id = dic.get(d_i,dic[list(dic.keys())[0]])#TODO: erase all these tests
        document_length = index.document_length(doc_id)
        tmp=1
        for i in [beta*(counts[q] / document_length) + (1-beta)*id2tf[token2id[pyndri.krovetz_stem(q)]]/total_corpus_term_count for q in pyndri.tokenize(query)]:
            tmp *= i
        query_to_doc_probability[d_i]=tmp

    return query_to_doc_probability



def create_tfidf_vectors(sentences):
    all_sentences=[]
    N = index.document_count()
    for doc in sentences:
        terms = index.document(doc)[1]
        counts = Counter(terms)
        for sentence in sentence[doc]:
            words = sentence.split()
            sentence_vector = np.zeros(len(token2id))
            for word in words:
                tfidf,id = get_tdidf_value_of_word(word, counts, N)
                sentence_vector[id]+=tfidf
            sentence_vector/=len(words)
            all_sentences.append(sentence)
    return all_sentences


def transform_terms_to_counts(Dinit):
    Dinit_counts={}
    for d_i in Dinit:
        doc_id = dic.get(d_i,dic[list(dic.keys())[0]])
        terms = index.document(doc_id)[1]
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
    terms = index.document(dic[doc])[1]
    doc_vector = np.zeros(len(token2id))
    number_docs = index.document_count()
    number_of_terms = len(terms)
    counts = Counter(terms)
    for term in set(terms):
        tfidf = get_tfidf_value(term,counts,number_docs,number_of_terms)
        doc_vector[term]+=tfidf
    return doc_vector

