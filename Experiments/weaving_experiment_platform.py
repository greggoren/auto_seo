from SpamTechniques.weaving import create_new_document_by_weaving
from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext
from Experiments.experiment_data_processor import create_index
from Experiments.experiment_data_processor import merge_indices
from Experiments.experiment_data_processor import create_features_file
from Experiments.experiment_data_processor import wait_for_feature_file_to_be_deleted
from Experiments.experiment_data_processor import move_feature_file
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


if __name__=="__main__":

    threshold = float(sys.argv[1])
    run_name = "_"+str(threshold).replace(".","")

    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

    reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
    queries = retrieve_query_names()
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-01"):
            doc_texts[doc]=a_doc_texts[doc]

    new_texts={}
    print("starting summarization")
    for query in reference_docs:
        print("in",query )
        sys.stdout.flush()
        reference_doc=reference_docs[query]
        new_texts[query] = create_new_document_by_weaving(doc_texts[reference_doc],queries[query],threshold)
    print("finished summarization")
    summary_file = open("new_texts"+run_name,"wb")
    pickle.dump(new_texts, summary_file)
    summary_file.close()

    reference_docs_list = list(reference_docs.values())
    create_trectext(doc_texts, reference_docs_list, new_texts, run_name)
    index_path = create_index(run_name)
    print("merging indices")
    sys.stdout.flush()
    new_index_name = merge_indices(index_path,run_name)
    features_dir = "Features"+run_name
    feature_file="features"
    wait_for_feature_file_to_be_deleted(feature_file)
    create_features_file(features_dir,new_index_name,params.query_description_file)
    move_feature_file(feature_file,run_name)
    index_doc_name = create_index_to_doc_name_dict(feature_file+run_name)
    scores_file=run_model(feature_file+run_name)
    results=retrieve_scores(index_doc_name,scores_file)
    results_file = open("scores_of_model"+run_name,"wb")
    pickle.dump(results,results_file)
    results_file.close()