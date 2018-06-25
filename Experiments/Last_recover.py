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
    # runs_weaving = ["00", "01", "02", "030000000000000004", "04", "05", "06", "07"]
    runs_weaving = ["00", "01", "030000000000000004", "04", "05", "06"]
    for run_name in runs_weaving:
        # features_dir = "Features"+run_name
        feature_file="features_"
        # wait_for_feature_file_to_be_deleted(feature_file)
        # create_features_file(features_dir,"/lv_local/home/sgregory/auto_seo/new_merged_index_1_05",params.queries_xml,run_name)
        # move_feature_file(feature_file,run_name)
        index_doc_name = create_index_to_doc_name_dict(feature_file+run_name)
        scores_file=run_model(feature_file+run_name,run_name)
        results=retrieve_scores(index_doc_name,scores_file)
        results_file = open("scores_of_model_"+run_name,"wb")
        pickle.dump(results,results_file)
        results_file.close()