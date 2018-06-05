from Multi_document_summary.multi_doc_summarization import create_multi_document_summarization
from Preprocess.preprocess import retrieve_ranked_lists
from Preprocess.preprocess import load_file
import params
import pickle

def retrieve_query_names():
    query_mapper = {}
    with open("/home/student/data/queris.txt",'r') as file:
        for line in file:
            data = line.split(":")
            query_mapper[data[0]]=data[1].rstrip()
    return query_mapper




k_docs_above=1
ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
queries = retrieve_query_names()
doc_texts = load_file(params.trec_text_file)

summaries={}
for query in reference_docs:
    reference_doc=reference_docs[query]
    summaries[query] = create_multi_document_summarization(ranked_lists,query,reference_doc,k_docs_above,doc_texts)
pickle.dump(summaries,open("summaries","wb"))
