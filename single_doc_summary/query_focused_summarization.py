from Preprocess.preprocess import create_document_tf_id_vector
from Preprocess.preprocess import index
from Preprocess.preprocess import turn_sentence_into_terms
from Preprocess.preprocess import retrieve_sentences
from Preprocess.preprocess import get_Dinit_for_query
from Preprocess.preprocess import retrieve_ranked_lists
from Preprocess.preprocess import transform_terms_to_counts
from Preprocess.preprocess import query_probability_given_docs
import params
from LanguageModels.models import KL_divergence
import math

def cosine_similarity(v1,v2):
    sumxx, sumxy, sumyy = 0, 0, 0
    for i in range(len(v1)):
        x = v1[i]; y = v2[i]
        sumxx += x*x
        sumyy += y*y
        sumxy += x*y
    return sumxy/math.sqrt(sumxx*sumyy)



def get_top_k_most_similar_docs_ranked_above(k,ranked_lists,query,reference_doc):
    ranked_list = ranked_lists[query]
    index_of_reference = ranked_list.index(reference_doc)
    if index_of_reference <=k:
        return ranked_list[:k]
    subset_docs = ranked_list[:index_of_reference]
    reference_vector = create_document_tf_id_vector(reference_doc)
    similarities = [(doc,cosine_similarity(create_document_tf_id_vector(doc),reference_vector)) for doc in subset_docs]
    top_k_docs = [doc[0] for doc in sorted(similarities,key=lambda x:x[1],reverse=True)[:k]]
    return top_k_docs

def get_top_m_sentences(m,doc,Dinit_counts,query_to_doc_probability):
    final = []
    sentences = retrieve_sentences(doc)
    for i in range(len(sentences)):
        sentence=sentences[i]
        sentence_terms = turn_sentence_into_terms(sentence)
        final.append((i,KL_divergence(sentence_terms,Dinit_counts,query_to_doc_probability)))
    return [s[0] for s in sorted(final,key=lambda x:x[1],reverse=True)[:m]]




def summarize_docs_for_query(queries,k,m,reference_docs):

    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    summaries = {}
    for query in reference_docs:
        summaries[query]={}
        Dinit = get_Dinit_for_query(query)
        Dinit_counts = transform_terms_to_counts(Dinit)
        top_k = get_top_k_most_similar_docs_ranked_above(k,ranked_lists,query,reference_docs[query])
        for doc in top_k:
            query_to_doc_probability=query_probability_given_docs(queries[query],Dinit_counts)
            summaries[query][doc]=get_top_m_sentences(m,doc,Dinit_counts,query_to_doc_probability)
    return summaries



