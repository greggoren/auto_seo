from Single_doc_summary.query_focused_summarization import summarize_docs_for_query
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




k=1
m=2
d = retrieve_ranked_lists(params.ranked_lists_file)
reference_docs = {q:d[q][-1] for q in d}
queries = retrieve_query_names()
doc_texts = load_file(params.trec_text_file)
summaries = summarize_docs_for_query(queries,k,m,reference_docs,doc_texts)
pickle.dump(summaries,open("summaries","wb"))
