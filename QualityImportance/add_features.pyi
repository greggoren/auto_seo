from collections import Counter
from math import log
import pyphen
import pyndri
import params
import pickle
def document_entropy(terms):
    """

    :param terms:
    :return: entropy of a document -sum_{w} p(w|d)*log(p(w|d))
    """
    total_frequencies=0
    counts = Counter(terms)
    for term in terms:
        total_frequencies +=counts[term]
    ent = 0
    for term in terms:
        p = counts[term]/total_frequencies
        ent-=p*log(p,2)
    return str(ent)

def stop_ratio(terms,stop_words_tokens):
    # stop_words_tokens = [token2id[pyndri.krovetz_stem(stop_word)] for stop_word in stop_words]
    stop_words_count=0
    non_stop_words_count=0
    for term in terms:
        if term in stop_words_tokens:
            stop_words_count+=1
        else:
            non_stop_words_count+=1
    return str(stop_words_count/non_stop_words_count)

def stop_cover(terms,stop_words_tokens):
    return str(len(set(terms).intersection(set(stop_words_tokens)))/len(stop_words_tokens))

def average_term_length(terms,id2token):
    sum_of_lengths = 0
    for term in terms:
        word = id2token[term]
        sum_of_lengths+=len(word)
    return str(sum_of_lengths/len(terms))


def average_df(terms,id2df):
    total_df=0
    for term in set(terms):
        total_df+=id2df[term]
    return str(total_df/len(set(terms)))

def unique_terms_ratio(terms):
   return str(len(set(terms))/len(terms))


def difficult_words_ratio(terms,popular_terms):
    return str(len(set(terms).difference(popular_terms))/len(set(terms)))


def average_syllables(terms,id2token):
    sum_of_syllables = 0
    d = pyphen.Pyphen(lang='en')
    for term in terms:
        word = id2token[term]
        sum_of_syllables+=len(d.inserted(word).split("-"))
    return str(sum_of_syllables/len(terms))

def top_freq_lists(tfs):
    tops = sorted(list(tfs.keys()),key=lambda x:tfs[x],reverse=True)
    return tops[100],tops[10000]


def create_features_line(terms,popular_terms,stop_words,id2token,id2df):
    new_line=""
    new_line+="26:"+document_entropy(terms)+" "
    new_line+="27:"+stop_cover(terms,stop_words)+" "
    new_line+="28:"+stop_ratio(terms,stop_words)+" "
    new_line+="29:"+average_df(terms,id2df)+" "
    new_line+="30:"+unique_terms_ratio(terms)+" "
    new_line+="31:"+difficult_words_ratio(terms,popular_terms)+" "
    new_line+="32:"+average_term_length(terms,id2token)+" "
    new_line+="33:"+average_syllables(terms,id2token)+" "
    return new_line

def create_features_file(features_file):
    f = open("new_dic.pickle",'rb')
    dic = pickle.load(f)
    f.close()
    index=pyndri.Index(params.path_to_index)
    token2id, id2token, id2df = index.get_dictionary()
    del token2id
    term_frequencies=index.get_term_frequencies()
    stop_words,popular_terms = top_freq_lists(term_frequencies)
    new_features_file=open("Quality_Features","w")
    with open(features_file) as file:
        for line in file:
            splited=line.split(" # ")
            features,doc_name =splited[0], splited[1]
            did=dic[doc_name]
            terms = index.document(did)[1]
            add_to_line = create_features_line(terms,popular_terms,stop_words,id2token,id2df)
            final_line = features+" "+add_to_line+" # "+doc_name+"\n"
            new_features_file.write(final_line)
        new_features_file.close()

create_features_file(params.data_set_file)

