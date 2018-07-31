from collections import Counter
from math import log
import pyphen
import pyndri
import params
import pickle
def document_entropy(terms,max=None,min=None):
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
    if max is None:
        return ent
    value = (ent-min)/(max-min)
    return str(value)

def stop_ratio(terms,stop_words_tokens,max=None,min=None):
    # stop_words_tokens = [token2id[pyndri.krovetz_stem(stop_word)] for stop_word in stop_words]
    stop_words_count=0
    non_stop_words_count=0
    for term in terms:
        if term in stop_words_tokens:
            stop_words_count+=1
        else:
            non_stop_words_count+=1
    if max is None:
        return stop_words_count/non_stop_words_count
    value=stop_words_count/non_stop_words_count
    value = (value-min)/(max-min)
    return str(value)

def stop_cover(terms,stop_words_tokens,max=None,min=None):
    if max is None:
        return len(set(terms).intersection(set(stop_words_tokens)))/len(stop_words_tokens)
    value = len(set(terms).intersection(set(stop_words_tokens)))/len(stop_words_tokens)
    value = (value-min)/(max-min)
    return str(value)

def average_term_length(terms,id2token,max=None,min=None):
    sum_of_lengths = 0
    for term in terms:
        word = id2token[term]
        sum_of_lengths+=len(word)
    if max is None:
        return sum_of_lengths/len(terms)
    value = sum_of_lengths/len(terms)
    value = (value-min)/(max-min)
    return str(value)

def average_df(terms,id2df,max=None,min=None):
    total_df=0
    for term in set(terms):
        total_df+=id2df.get(term,0)
    if max is None:
        return total_df/len(set(terms))
    value = total_df/len(set(terms))
    value = (value-min)/(max-min)
    return str(value)

def unique_terms_ratio(terms,max=None,min=None):
    if max is None:
        return len(set(terms))/len(terms)
    value = len(set(terms))/len(terms)
    value = (value-min)/(max-min)
    return str(value)


def difficult_words_ratio(terms,popular_terms,max=None,min=None):
    if max is None:
        return len(set(terms).difference(popular_terms))/len(set(terms))
    value = len(set(terms).difference(popular_terms))/len(set(terms))
    value = (value-min)/(max-min)
    return str(value)



def average_syllables(terms,id2token,max=None,min=None):
    sum_of_syllables = 0
    d = pyphen.Pyphen(lang='en')
    for term in terms:
        word = id2token[term]
        sum_of_syllables+=len(d.inserted(word).split("-"))
    if max is None:
        return sum_of_syllables/len(terms)
    value = sum_of_syllables/len(terms)
    value = (value-min)/(max-min)
    return str(value)

def top_freq_lists(tfs):
    tops = sorted(list(tfs.keys()),key=lambda x:tfs[x],reverse=True)
    return tops[:100],tops[:10000]


def create_features_line(terms,popular_terms,stop_words,id2token,id2df,maximum,minimum):
    new_line=""
    new_line+="26:"+document_entropy(terms,maximum[0],minimum[0])+" "
    new_line+="27:"+stop_cover(terms,stop_words,maximum[1],minimum[1])+" "
    new_line+="28:"+stop_ratio(terms,stop_words,maximum[2],minimum[2])+" "
    new_line+="29:"+average_df(terms,id2df,maximum[3],minimum[3])+" "
    new_line+="30:"+unique_terms_ratio(terms,maximum[4],minimum[4])+" "
    new_line+="31:"+difficult_words_ratio(terms,popular_terms,maximum[5],minimum[5])+" "
    new_line+="32:"+average_term_length(terms,id2token,maximum[6],minimum[6])+" "
    new_line+="33:"+average_syllables(terms,id2token,maximum[7],minimum[7])
    return new_line

def get_values(terms,popular_terms,stop_words,id2token,id2df):
    stats=[]
    stats.append(document_entropy(terms))
    stats.append(stop_cover(terms,stop_words))
    stats.append(stop_ratio(terms,stop_words))
    stats.append(average_df(terms,id2df))
    stats.append(unique_terms_ratio(terms))
    stats.append(difficult_words_ratio(terms,popular_terms))
    stats.append(average_term_length(terms,id2token))
    stats.append(average_syllables(terms,id2token))

    return stats

def get_max_min_stats(features_file, id2token, id2df,term_frequencies,stop_words,popular_terms,dic,index):


    values=[[] for i in range(8)]
    maximum=[]
    minimum=[]
    with open(features_file) as file:
        for line in file:
            splited=line.split(" # ")
            features,doc_name =splited[0], splited[1]
            doc_name=doc_name.split("\n")[0]
            did=dic[doc_name]
            terms = index.document(did)[1]
            stats = get_values(terms,popular_terms,stop_words,id2token,id2df)
            for i,l in enumerate(values):
                l.append(stats[i])
        for val in values:
            maximum.append(max(val))
            minimum.append(min(val))
        return maximum,minimum





def create_features_file(features_file):
    f = open("new_dic.pickle",'rb')
    dic = pickle.load(f)
    f.close()
    index=pyndri.Index(params.path_to_index)
    token2id, id2token, id2df = index.get_dictionary()
    del token2id
    term_frequencies=index.get_term_frequencies()
    stop_words,popular_terms = top_freq_lists(term_frequencies)
    maximum,minimum = get_max_min_stats(features_file, id2token, id2df,term_frequencies,stop_words,popular_terms,dic,index)
    new_features_file=open("Quality_Features","w")
    with open(features_file) as file:
        for line in file:
            splited=line.split(" # ")
            features,doc_name =splited[0], splited[1]
            doc_name=doc_name.split("\n")[0]
            did=dic[doc_name]
            terms = index.document(did)[1]
            add_to_line = create_features_line(terms,popular_terms,stop_words,id2token,id2df,maximum,minimum)
            final_line = features+" "+add_to_line+" # "+doc_name+"\n"
            new_features_file.write(final_line)
        new_features_file.close()

create_features_file("features")

