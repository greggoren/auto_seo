from Preprocess.preprocess import index
from Preprocess.preprocess import dic
from Preprocess.preprocess import id2token
from Preprocess.preprocess import id2tf
from Preprocess.preprocess import total_corpus_term_count
import math
from params import beta


def KL_divergence(sentance,Dinit_counts,query_to_doc_probability):
    result = 0
    for word in id2token:
        r = relevance_model(Dinit_counts,query_to_doc_probability,word)
        result += r*math.log(r/word_probability_given_sentence(word,sentance))
    return result

def relevance_model(Dinit_counts,query_to_doc_probability,word):
    sum=0
    denominator=0
    P_d = 1/len(Dinit_counts)
    for d_i in Dinit_counts:
        doc_id = dic.get(d_i, dic[list(dic.keys())[0]])#TODO: remove it...only for tests...
        document_length = index.document_length(doc_id)
        counts = Dinit_counts[d_i]
        tf = counts[word]
        P_w = tf/document_length
        sum+=P_d*P_w*query_to_doc_probability[d_i]
        denominator+=query_to_doc_probability[d_i]
    denominator*=P_d
    return sum/denominator

def word_probability_given_sentence(word,sentence):
    tf = len([i for i in sentence if i==word])
    length = len(sentence)
    result = beta*(tf/length)+ (1-beta)*(id2tf[word])/total_corpus_term_count
    return result

