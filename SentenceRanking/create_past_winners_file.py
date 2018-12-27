from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import  create_features_file_original, \
    create_trectext_original
from utils import run_command
from utils import run_bash_command
from Experiments.experiment_data_processor import merge_indices
from Experiments.experiment_data_processor import create_index
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
import params
import os
def get_docs(doc_texts,round):
    result = {}
    index = str(round).zfill(2)
    for doc in doc_texts:
        if doc.__contains__("ROUND-"+index):
            result[doc]=doc_texts[doc]
    return result

def create_trec_eval_file(results,run_name):
    trec_file = "/home/greg/auto_seo/data/trec_file"+run_name+".txt"
    trec_file_access = open(trec_file, 'a')
    for doc in results:
        query = doc.split("-")[2]
        trec_file_access.write(query
             + " Q0 " + doc + " " + str(0) + " " + str(
                results[doc]) + " seo\n")
    trec_file_access.close()
    return trec_file

def order_trec_file(trec_file):
    final = trec_file.replace(".txt", "")
    command = "sort -k1,1 -k5nr -k2,1 " + trec_file + " > " + final
    for line in run_command(command):
        print(line)
    return final

def run_model(test_file,run_name=""):
    java_path = "/home/greg/jdk1.8.0_181/bin/java"
    jar_path = "/home/greg/SEO_CODE/model_running/RankLib.jar"
    score_file = "scores_winners/scores_of_seo_run"+run_name
    if not os.path.exists("scores_winners/"):
        os.makedirs("scores_winners/")
    features = test_file
    model_path = "/home/greg/auto_seo/CrossValidationUtils/model_bot_group"
    run_bash_command('touch ' + score_file)
    command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
    out = run_bash_command(command)
    print(out)
    return score_file





doc_texts = load_file(params.trec_text_file)
merged_index=""
for index in range(1,6):
    print("in epoch",index)
    doc_text_for_round = get_docs(doc_texts, round=index)
    trec_text_file = create_trectext_original(document_text=doc_text_for_round, summaries = [],run_name= str(index),avoid=[])
    new_index = create_index(trec_text_file,str(index))
    if merged_index:
        run_bash_command("rm -r "+merged_index)
    merged_index = merge_indices(new_index=new_index,run_name=str(index),new_index_name="merged_index")
    feature_file = "features"+ "_" + str(index)
    features_dir = "Features"
    queries_file = "/home/greg/auto_seo/data/queries.xml"
    create_features_file_original(features_dir=features_dir, index_path=merged_index, new_features_file=feature_file , run_name=str(index),queries_file=queries_file)
    index_doc_name = create_index_to_doc_name_dict(feature_file)
    scores_file = run_model(feature_file,str(index))
    results = retrieve_scores(index_doc_name, scores_file)
    trec_file = create_trec_eval_file(results,str(index))
    order_trec_file(trec_file)


