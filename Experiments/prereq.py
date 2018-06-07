import os
import params
from utils import run_bash_command
from utils import run_command



def create_model_LambdaMart(train_file,
                            query_relevance_file, args):
    java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
    jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
    number_of_leaves, number_of_trees = args
    models_path = "/lv_local/home/sgregory/auto_seo/models/"
    try:
        if not os.path.exists(models_path):
            os.makedirs(models_path)
    except:
        print("collision")
    command = java_path + ' -jar ' + jar_path + ' -train ' + train_file + ' -ranker 6 -qrel ' + query_relevance_file + ' -metric2t NDCG@20' \
                                                                                                                                 ' -tree ' + str(
        number_of_trees) + ' -leaf ' + str(number_of_leaves) + ' -save ' + models_path + 'model_' + str(
        number_of_trees) + "_" + str(number_of_leaves)
    print("command = ", command)
    run_bash_command(command)


def create_index_to_doc_name_dict(data_set_file):
    doc_name_index={}
    index = 0
    with open(data_set_file) as ds:
        for line in ds:
            rec = line.split("# ")
            doc_name = rec[1].rstrip()
            doc_name_index[index] = doc_name
            index += 1
        return doc_name_index


def order_trec_file(trec_file):
    final = trec_file.replace(".txt", "")
    command = "sort -k1,1 -k5nr -k2,1 " + trec_file + " > " + final
    for line in run_command(command):
        print(line)
    return final


def run_model(test_file):
    java_path = "/lv_local/home/sgregory/jdk1.8.0_121/bin/java"
    jar_path = "/lv_local/home/sgregory/SEO_CODE/model_running/RankLib.jar"
    score_file = "scores/scores_of_seo_run"
    if not os.path.exists("scores/"):
        os.makedirs("scores/")
    features = test_file
    model_path = params.model_path
    run_bash_command('touch ' + score_file)
    command = java_path + " -jar " + jar_path + " -load " + model_path + " -rank " + features + " -score " + score_file
    run_bash_command(command)
    return score_file

def create_trec_eval_file(results):
    trec_file = "/lv_local/home/sgregory/auto_seo/data/trec_file.txt"
    trec_file_access = open(trec_file, 'a')
    for doc in results:
        query = doc.split("-")[2]
        trec_file_access.write(query
             + " Q0 " + doc + " " + str(0) + " " + str(
                results[doc]) + " seo\n")
    trec_file_access.close()
    return trec_file


def retrieve_scores(doc_name_index, score_file):
    with open(score_file) as scores:
        results = {doc_name_index[i]: float(score.split()[2].rstrip()) for i, score in enumerate(scores)}
        return results


doc_name_index = create_index_to_doc_name_dict("/lv_local/home/sgregory/auto_seo/data/features_asr")
#create_model_LambdaMart("/lv_local/home/sgregory/auto_seo/data/featuresCB_asr","/lv_local/home/sgregory/auto_seo/data/qrels",(50,250))
score_file = run_model("/lv_local/home/sgregory/auto_seo/data/features_asr")
results = retrieve_scores(doc_name_index,score_file)
trec_file = create_trec_eval_file(results,doc_name_index)
order_trec_file(trec_file)
