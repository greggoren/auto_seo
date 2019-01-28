from Experiments.experiment_data_processor import create_features_file_sentence_exp
from CompetitionBot.create_ds_for_annotations import get_reference_documents,ASR_MONGO_PORT,ASR_MONGO_HOST
from utils import run_command
from utils import run_bash_command
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
import os
import params
import numpy as np
from pymongo import MongoClient
def get_docs(doc_texts,round):
    result = {}
    index = str(round).zfill(2)
    for doc in doc_texts:
        if doc.__contains__("ROUND-"+index):
            result[doc]=doc_texts[doc]
    return result

def create_trec_eval_file(results,run_name):
    trec_file = "trec_file"+run_name+".txt"
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
    command = "mv features_ "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)


def get_lists(trec_file):
    results = {}
    with open(trec_file) as file:
        for line in file:
            query = line.split()[0]
            doc = line.split()[2]
            if query not in results:
                results[query]=[]
            results[query].append(doc)
    return results



def get_average_bot_ranking(reference_docs,method_index,group,result_passive):
    results = {}

    for query_id in reference_docs:
        query_group = query_id.split("_")[1]
        if query_group!=group:
            continue
        for doc in reference_docs[query_id]:

            bot_method = method_index[query_id+"_"+doc]#document["bot_method"]
            position = result_passive[query_id].index(query_id+"-"+doc)+1
            if bot_method not in results:
                results[bot_method]=[]
            results[bot_method].append(position)
    for bot_method in results:
        results[bot_method]= np.mean(results[bot_method])
    return results

def get_static_bot_positions(reference_docs,results,result_passive,index):
    results[index]={}
    for query_id in reference_docs:

        query_group = query_id.split("_")[1]
        if query_group=="0":
            continue

        for doc in reference_docs[query_id]:

            position = result_passive[query_id].index("ROUND-06-"+query_id+"-"+doc)+1
            results[index][query_id]=position
    return results

def get_method_index():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    docs = db.documents.find({})
    index = {}
    for doc in docs:
        if "bot_method" in doc:
            index[doc["query_id"]+"_"+doc["username"]]=doc["bot_method"]
    return index



def create_working_sets(reference_docs):
    base_names ={}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = db.archive.distinct("iteration")
    iteration = sorted(list(iterations))[7]

    docs = db.archive.find({"iteration":iteration,"query_id":{"$regex":".*_2"}})
    for doc in docs:
        username = doc["username"]
        query_id = doc["query_id"]
        if query_id not in base_names:
            base_names[query_id]=[]
        base_names[query_id].append(username)

    for r in range(6,11):
        f=open("ws_"+str(r),"w")
        for query_id in base_names:
            for i,username in enumerate(base_names[query_id],start=1):
                current = str(r).zfill(2)
                if username in reference_docs[query_id]:
                    current = "06"
                docname = "ROUND-"+current+"-"+query_id+"-"+username
                f.write(query_id+" Q0 "+docname+" 0 "+str(-i)+" static\n")
        f.close()

def analyze_positions(positions):
    average_positions ={}
    average_potential_data={}
    raw_position_data = {}
    for index in positions:
        average_value = np.mean([positions[index][q] for q in positions[index]])
        average_positions[index]=average_value
        for query in positions[index]:
            if index > 6:
                if index not in raw_position_data:
                    raw_position_data[index]=[]
                    average_potential_data[index]=[]
                old_position = positions[index-1][query]
                new_position = positions[index][query]
                if new_position == 1 and old_position == 1:
                    break
                if new_position >= old_position:
                    denominator = 5 - old_position
                else:
                    denominator = old_position - 1
                if denominator == 0:
                    potential = 0
                else:
                    potential = (old_position - new_position) / (denominator)
                overall_promotion = old_position - new_position
                average_potential_data[index].append(potential)
                raw_position_data[index].append(overall_promotion)
    return average_positions,average_potential_data,raw_position_data


def write_data_file(stats,filename):
    f = open(filename,"w")
    for iteration in sorted(list(stats.keys())):
        f.write(str(iteration)+" "+str(stats[iteration])+"\n")
    f.close()

if __name__=="__main__":
    feature_file = "features_bot"
    features_dir = "Features"
    queries_file = "queries.xml"
    merged_index = "/home/greg/ASR18/Collections/competitionindex"
    ref_docs = get_reference_documents()
    create_working_sets(ref_docs)
    positions = {}
    for i in range(6,11):
        working_set = "ws_"+str(i)
        create_features_file_sentence_exp(features_dir=features_dir,index_path=merged_index,queries_file=queries_file,new_features_file=feature_file,working_set=working_set)
        index_doc_name = create_index_to_doc_name_dict(feature_file)
        scores_file = run_model(feature_file,"")
        results = retrieve_scores(index_doc_name, scores_file)
        trec_file = create_trec_eval_file(results,str(i).zfill(2))
        final_file = order_trec_file(trec_file)
        lists = get_lists(final_file)
        positions = get_static_bot_positions(ref_docs,positions,lists,i)
    average_positions, average_potential_data, raw_position_data = analyze_positions(positions)
    write_data_file(average_positions,"static_average")
    write_data_file(raw_position_data,"static_raw")
    write_data_file(average_potential_data,"static_potential")
