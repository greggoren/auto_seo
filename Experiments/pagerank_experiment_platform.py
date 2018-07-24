from Multi_document_summary.multi_doc_summarization import create_multi_document_summarization
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
    number_of_documents_above = int(sys.argv[1])
    gamma = float(sys.argv[2])
    run_name = "_"+str(number_of_documents_above)+"_"+str(gamma).replace(".","")
    print("uploading index")

    index = pyndri.Index(params.path_to_index)
    token2id, id2token, id2df = index.get_dictionary()
    del id2token
    f = open("dic4.pickle","rb")
    dic = pickle.load(f)
    f.close()

    print("loading index finished")

    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

    reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
    queries = retrieve_query_names()
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc]=a_doc_texts[doc]

    summaries={}
    print("starting summarization")
    for query in reference_docs:
        print("in",query )
        sys.stdout.flush()
        reference_doc=reference_docs[query]
        summaries[reference_docs[query]] = create_multi_document_summarization(ranked_lists,query,queries[query],reference_doc,number_of_documents_above,gamma,doc_texts,index,token2id,dic,id2df,run_name)
    print("finished summarization")
    summary_file = open("summaries"+run_name,"wb")
    pickle.dump(summaries,summary_file)
    summary_file.close()
    del index
    del token2id
    del id2df
    del dic
    reference_docs_list = list(reference_docs.values())
    print(reference_docs_list)
    create_trectext(doc_texts,reference_docs_list,summaries,run_name)
    index_path = create_index(run_name)
    print("merging indices")
    sys.stdout.flush()
    new_index_name = merge_indices(index_path,run_name)
    features_dir = "Features"+run_name
    feature_file="features"
    wait_for_feature_file_to_be_deleted(feature_file)
    create_features_file(features_dir,new_index_name,params.queries_xml,run_name)
    move_feature_file(feature_file,run_name)
    index_doc_name = create_index_to_doc_name_dict(feature_file+run_name)
    scores_file=run_model(feature_file+run_name,run_name)
    results=retrieve_scores(index_doc_name,scores_file)
    results_file = open("scores_of_model"+run_name,"wb")
    pickle.dump(results,results_file)
    results_file.close()
    f = open("stop.stop" + run_name, 'w')
    f.close()