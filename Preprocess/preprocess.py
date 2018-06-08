import xml.etree.ElementTree as ET
import nltk.data
import pyndri
import numpy as np
from collections import Counter
import math
import params
from params import beta
import re



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

def turn_sentence_into_terms(sentence,index,token2id):
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

def get_tdidf_value_of_word(word, counts, N,number_of_terms,token2id,id2df):
    stemmed = pyndri.krovetz_stem(word)
    id = token2id[stemmed]
    return  get_tfidf_value(id,counts,N,number_of_terms,id2df),id

def get_tfidf_value(id,counts,N,number_of_terms,id2df):
    df = id2df[id]
    tf = counts[id]/number_of_terms
    idf = math.log(float(N) / df)
    return tf*idf

def get_tfidf_value(id,counts,N,number_of_terms,id2df):
    df = id2df[id]
    tf = counts[id]/number_of_terms
    idf = math.log(float(N) / df)
    return tf*idf

# def split_Dinit_to_sentences(Dinit):
#     tree = ET.parse(params.trec_text_file)
#     root = tree.getroot()
#     Dinit_text = {}
#     for doc in root:
#         name = ""
#         for att in doc:
#             if att.tag == "DOCNO":
#                 name = att.text
#             else:
#                 if name in Dinit:
#                     Dinit_text[name] = Counter([token2id[pyndri.krovetz_stem(i)] for i in  retrieve_sentences(att.text)])
#     return Dinit_text

def query_probability_given_docs(query,Dinit_counts,index,dic,token2id,total_corpus_term_count):
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



def create_tfidf_vectors(sentences,index,token2id,id2df):
    all_sentences=[]
    N = index.document_count()
    for doc in sentences:
        terms = index.document(doc)[1]
        counts = Counter(terms)
        for sentence in sentence[doc]:
            words = sentence.split()
            sentence_vector = np.zeros(len(token2id))
            for word in words:
                tfidf,id = get_tdidf_value_of_word(word, counts, N,id2df)
                sentence_vector[id]+=tfidf
            #sentence_vector/=len(words)
            all_sentences.append(sentence)
    return all_sentences


def transform_terms_to_counts(Dinit,dic,index):
    Dinit_counts={}
    for d_i in Dinit:
        doc_id = dic.get(d_i,dic[list(dic.keys())[0]])
        terms = index.document(doc_id)[1]
        counts = Counter(terms)
        Dinit_counts[d_i]=counts
    return Dinit_counts

def convert_sentence_to_tfidf_vector(sentence,index,token2id,id2df):
    N = index.document_count()
    sentence=sentence.rstrip()
    sentence = re.sub('[!,?:]',"",sentence)
    sentence = re.sub('[.]'," ",sentence)
    sentence = re.sub("’ll"," will",sentence)
    sentence = re.sub("'ll"," will",sentence)
    sentence = re.sub("’s","",sentence)
    sentence = re.sub("'s","",sentence)
    words = sentence.split()
    tokens=[]
    for word in words:
        modified = re.sub(' ','',word)
        tokens.extend(pyndri.tokenize(modified))
    #words = [re.sub('[.,?:]',"",w) for w in words]
    sentence_vector = {}
    #sentence_vector = np.zeros(len(token2id))
    counts = Counter([token2id[pyndri.krovetz_stem(word)] for word in tokens])
    for word in set(tokens):
        tfidf, id = get_tdidf_value_of_word(word, counts, N,len(words),id2df)
        sentence_vector[id-1]=tfidf
    # sentence_vector/=len(tokens)
    return sentence_vector



def retrieve_ranked_lists(ranked_lists_file):
    chosen_docs={}
    with open(ranked_lists_file) as ranked_lists:
        for ranked_doc in ranked_lists:
            query,doc = ranked_doc.split()[0],ranked_doc.split()[2]
            if not chosen_docs.get(query,False):
                chosen_docs[query]=[]
            chosen_docs[query].append(doc)
        return chosen_docs



def create_document_tf_idf_vector(doc,index,token2id,dic,id2df):
    terms = index.document(dic[doc.replace("EPOCH","ROUND")])[1]
    # doc_vector = np.zeros(len(token2id))
    doc_vector = {}
    number_docs = index.document_count()
    number_of_terms = len(terms)
    counts = Counter(terms)
    for term in set(terms):
        tfidf = get_tfidf_value(term,counts,number_docs,number_of_terms,id2df)
        doc_vector[term-1]=tfidf
    return doc_vector


def create_sentence_indexes(document_texts,chosen_docs_for_summary,_index,token2id,id2df):
    sentence_texts={}
    sentence_vectors={}
    for document in chosen_docs_for_summary:
        document_for_id = document.replace("EPOCH","ROUND")
        sentence_texts[document_for_id]={}
        text = document_texts[document_for_id]
        sentences = retrieve_sentences(text)
        for index,sentence in enumerate(sentences):
            sentence_texts[document_for_id][str(index)] = sentence
            sentence_vectors[document_for_id+"_"+str(index)] = convert_sentence_to_tfidf_vector(sentence,_index,token2id,id2df)
    return sentence_texts,sentence_vectors