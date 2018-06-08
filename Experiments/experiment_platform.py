from Multi_document_summary.multi_doc_summarization import create_multi_document_summarization
from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext
from Experiments.experiment_data_processor import create_index
from Experiments.experiment_data_processor import merge_indices
from Experiments.experiment_data_processor import create_features_file
from Experiments.model_handler import run_model
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
import params
import pickle
import pyndri
import sys


def retrieve_query_names():
    query_mapper = {}
    with open(params.query_description_file,'r') as file:
        for line in file:
            data = line.split(":")
            query_mapper[data[0]]=data[1].rstrip()
    return query_mapper


index = pyndri.Index(params.path_to_index)
token2id, id2token, id2df = index.get_dictionary()
id2tf = index.get_term_frequencies()
dic={}
total_corpus_term_count=0
doc_length = {}
for document_id in range(index.document_base(), index.maximum_document()):
    dic[index.document(document_id)[0]] = document_id
    doc_length[index.document(document_id)[0]] = index.document_length(document_id)
    total_corpus_term_count+=len(index.document(document_id))



ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
queries = retrieve_query_names()
doc_texts = load_file(params.trec_text_file)

summaries={}
print("starting summarization")
for query in reference_docs:
    print("in",query )
    sys.stdout.flush()
    reference_doc=reference_docs[query]
    summaries[query] = create_multi_document_summarization(ranked_lists,query,queries[query],reference_doc,params.number_of_documents_above,doc_texts,index,token2id,dic)
print("finished summarization")
summary_file = open("summaries","wb")
pickle.dump(summaries,summary_file)
summary_file.close()

reference_docs_list = list(reference_docs.values())
create_trectext(doc_texts,reference_docs_list,summaries)
index_path = create_index()
print("merging indices")
sys.stdout.flush()
merge_indices(index_path)
create_features_file()
index_doc_name = create_index_to_doc_name_dict("features")
scores_file=run_model("features")
results=retrieve_scores(index_doc_name,scores_file)
results_file = open("scores_of_model","wb")
pickle.dump(results,results_file)
results_file.close()