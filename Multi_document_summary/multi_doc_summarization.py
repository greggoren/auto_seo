from Preprocess.preprocess import create_document_tf_idf_vector
from Preprocess.preprocess import convert_sentence_to_tfidf_vector
from Preprocess.preprocess import create_sentence_indexes
from Preprocess.preprocess import create_document_tf_idf_vector
from Multi_document_summary.page_rank import page_rank,create_transition_graph
from utils import cosine_similarity
import params
from Multi_document_summary.diversification import diversify

def get_top_k_most_similar_docs_ranked_above(k,ranked_lists,query,reference_doc):
    ranked_list = ranked_lists[query]
    index_of_reference = ranked_list.index(reference_doc)
    if index_of_reference <=k:
        return ranked_list[:k]
    subset_docs = ranked_list[:index_of_reference]
    reference_vector = create_document_tf_idf_vector(reference_doc)
    similarities = [(doc,cosine_similarity(create_document_tf_idf_vector(doc), reference_vector)) for doc in subset_docs]
    top_k_docs = [doc[0] for doc in sorted(similarities,key=lambda x:x[1],reverse=True)[:k]]
    return top_k_docs





def create_multi_document_summarization(ranked_lists, query_number,query_text, reference_doc, k_docs_above, doc_texts):
    top_k_docs = get_top_k_most_similar_docs_ranked_above(k_docs_above, ranked_lists, query_number, reference_doc)
    top_k_docs.append(reference_doc)
    sentence_texts, sentence_vectors=create_sentence_indexes(doc_texts,top_k_docs)
    transition_matrix = create_transition_graph(sentence_vectors)
    scores=page_rank(params.alpha,transition_matrix)
    query_vector = convert_sentence_to_tfidf_vector(query_text)
    original_doc_vector = create_document_tf_idf_vector(reference_doc)
    summary=diversify(scores,transition_matrix,params.number_of_sentences,query_vector,sentence_vectors,original_doc_vector,params.gamma)
    return summary