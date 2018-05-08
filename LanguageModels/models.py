from Preprocess.preprocess import index
import math
from params import beta


def KL_divergence(sentance,Dinit_counts,query_to_doc_probability):
    result = 0
    token2id, id2token, id2df = index.get_dictionary()
    for word in id2token:
        r = relevance_model(Dinit_counts,query_to_doc_probability,word)
        result += r*math.log(r/word_probability_given_sentence(word,sentance))




def relevance_model(Dinit_counts,query_to_doc_probability,word):
    sum=0
    P_d = 1/len(Dinit_counts)
    for d_i in Dinit_counts:
        document_length = index.document_length(d_i)
        counts = Dinit_counts[d_i]
        tf = counts[word]
        P_w = tf/document_length
        sum+=P_d*P_w*query_to_doc_probability[d_i]
    return sum

def word_probability_given_sentence(word,sentence):
    tf = len([i for i in sentence if i==word])
    length = len(sentence)
    return beta*(tf/length)+ (1-beta)*(index.get_term_frequencies()[word])

