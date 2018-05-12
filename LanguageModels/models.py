from Preprocess.preprocess import id2token
from Preprocess.preprocess import id2tf
from Preprocess.preprocess import total_corpus_term_count
from Preprocess.preprocess import doc_length

import math
from params import beta


def KL_divergence(sentance,Dinit_counts,query_to_doc_probability):
    result = 0
    all_words = len(id2token)
    processed_words = 0
    for word in id2token:
        if processed_words%1000 == 0:
            print("processed ",processed_words,"out of",all_words)
        print()
        r = relevance_model(Dinit_counts,query_to_doc_probability,word)
        if r==0.0:
            continue
        result += r*math.log(r/word_probability_given_sentence(word,sentance))
        processed_words+=1

    return result

def relevance_model(Dinit_counts,query_to_doc_probability,word):
    sum=0
    denominator=0
    P_d = 1/len(Dinit_counts)
    for d_i in Dinit_counts:
        document_length = doc_length[d_i]
        counts = Dinit_counts[d_i]
        tf = counts[word]
        P_w = tf/document_length
        sum+=P_w*query_to_doc_probability[d_i]
        denominator+=query_to_doc_probability[d_i]
    denominator*=P_d
    sum*=P_d
    return sum/denominator

def word_probability_given_sentence(word,sentence):
    tf = len([i for i in sentence if i==word])
    length = len(sentence)
    result = beta*(tf/length)+ (1-beta)*(id2tf[word])/total_corpus_term_count
    return result

