from Experiments.experiment_data_processor import create_features_file_sentence_exp
from utils import run_command
from utils import run_bash_command
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
import os
from sys import argv
import params

def get_docs(doc_texts,round):
    result = {}
    index = str(round).zfill(2)
    for doc in doc_texts:
        if doc.__contains__("ROUND-"+index):
            result[doc]=doc_texts[doc]
    return result

def create_trec_eval_file(results,run_name):
    trec_file = "/home/greg/auto_seo/SentenceRanking/trec_file"+run_name+".txt"
    trec_file_access = open(trec_file, 'w')
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
    java_path = "/lv_local/home/sgregory/jdk1.8.0_181/bin/java"
    jar_path = "/lv_local/home/sgregory/RankLib.jar"
    score_file = "scores_winners/scores_of_seo_run"+run_name
    if not os.path.exists("scores_winners/"):
        os.makedirs("scores_winners/")
    features = test_file
    model_path = "/lv_local/home/sgregory/auto_seo/CrossValidationUtils/model_bot_group"
    run_bash_command('touch ' + score_file)
    command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
    out = run_bash_command(command)
    print(out)
    return score_file

def create_features_file_sentence_exp(features_dir,index_path,queries_file,new_features_file,working_set):
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    command= params.ltr_features_script+" "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile='+ working_set + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl "+params.features_generator_script_path+" "+features_dir+" "+working_set
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)


if __name__=="__main__":
    feature_file = "features"
    features_dir = "Features"
    queries_file = "/lv_local/home/sgregory/auto_seo/data/queries.xml"
    merged_index = "/lv_local/home/sgregory/Bots/mergedindices"
    working_set = "working_set_passive_bots"
    create_features_file_sentence_exp(features_dir=features_dir,index_path=merged_index,queries_file=queries_file,new_features_file=feature_file,working_set=working_set)

    index_doc_name = create_index_to_doc_name_dict(feature_file)
    scores_file = run_model(feature_file,"")
    results = retrieve_scores(index_doc_name, scores_file)
    trec_file = create_trec_eval_file(results,"")
    order_trec_file(trec_file)