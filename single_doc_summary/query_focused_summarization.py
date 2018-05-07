from Preprocess.preprocess import create_document_tf_id_vector
from Preprocess.preprocess import index
from Preprocess.preprocess import retrieve_sentences
import math

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)



def get_top_k_most_similar_docs_ranked_above(k,ranked_lists,query,reference_doc,index):
    ranked_list = ranked_lists[query]
    index_of_reference = ranked_list.index(reference_doc)
    if index_of_reference <=k:
        return ranked_list[:k]
    subset_docs = ranked_list[:index_of_reference]
    reference_vector = create_document_tf_id_vector(reference_doc,index)
    similarities = [(doc,cosine_similarity(create_document_tf_id_vector(doc,index),reference_vector)) for doc in subset_docs]
    top_k_docs = [doc[0] for doc in sorted(similarities,key=lambda x:x[1],reverse=True)[:k]]
    return top_k_docs

def get_top_m_sentences(doc,query):
    sentences = retrieve_sentences(doc)
    query_words = query.split()
    for i in range(len(sentences)):
        sentence=sentences[i]
        words = sentence.split()
        




