from Preprocess.preprocess import convert_sentence_to_tfidf_vector
from Preprocess.preprocess import create_sentence_indexes
from Preprocess.preprocess import create_document_tf_idf_vector
from Multi_document_summary.page_rank import page_rank,create_transition_graph
from utils import cosine_similarity
import params
from Multi_document_summary.diversification import diversify
import sys
def get_top_k_most_similar_docs_ranked_above(k,ranked_lists,query,reference_doc,index,token2id,dic,id2df):
    ranked_list = ranked_lists[query]
    index_of_reference = ranked_list.index(reference_doc)
    if index_of_reference <=k:
        return ranked_list[:k]
    subset_docs = ranked_list[:index_of_reference]
    reference_vector = create_document_tf_idf_vector(reference_doc,index,token2id,dic,id2df)
    similarities = [(doc,cosine_similarity(create_document_tf_idf_vector(doc,index,token2id,dic,id2df), reference_vector)) for doc in subset_docs]
    top_k_docs = [doc[0] for doc in sorted(similarities,key=lambda x:x[1],reverse=True)[:k]]
    return top_k_docs





def create_multi_document_summarization(ranked_lists, query_number,query_text, reference_doc, k_docs_above, doc_texts,index,token2id,dic,id2df):
    print("get top",k_docs_above,"most similar documents")
    sys.stdout.flush()
    top_k_docs = get_top_k_most_similar_docs_ranked_above(k_docs_above, ranked_lists, query_number, reference_doc,index,token2id,dic,id2df)
    top_k_docs.append(reference_doc)
    print("get sentence data")
    sys.stdout.flush()
    sentence_texts, sentence_vectors=create_sentence_indexes(doc_texts,top_k_docs)
    print("create transition matrix")
    sys.stdout.flush()
    transition_matrix = create_transition_graph(sentence_vectors)
    print("starting PageRank")
    sys.stdout.flush()
    scores=page_rank(params.alpha,transition_matrix)
    query_vector = convert_sentence_to_tfidf_vector(query_text,index,token2id,id2df)
    original_doc_vector = create_document_tf_idf_vector(reference_doc,index,token2id,dic,id2df)
    print("diverity algorithm applying")
    sys.stdout.flush()
    summary=diversify(scores,transition_matrix,params.number_of_sentences,query_vector,sentence_vectors,original_doc_vector,params.gamma)
    text = "\n".join([sentence_texts[sentence] for sentence in summary])
    return text