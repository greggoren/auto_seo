from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext_original
from Experiments.experiment_data_processor import create_features_file
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
from SentenceRanking.sentence_parse import map_sentences, map_set_of_sentences
from SentenceRanking.sentence_parse import create_lists
from Preprocess.preprocess import retrieve_sentences
import params
import sys
import time
import os
from utils import run_bash_command

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

def retrieve_query_names():
    query_mapper = {}
    with open(params.query_description_file,'r') as file:
        for line in file:
            data = line.split(":")
            query_mapper[data[0]]=data[1].rstrip()
    return query_mapper

def avoid_docs_for_working_set(reference_doc,reference_docs):
    diffenrece = set(reference_docs).difference(set([reference_doc]))
    return diffenrece

def determine_indexes(doc,ranked_list):
    return min(ranked_list.index(doc),3)

if __name__=="__main__":
    # new_ranked_list ="trec_file04"
    round = sys.argv[1].zfill(2)
    new_ranked_list ="trec_file"+round
    reference_index = int(sys.argv[2])
    # ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    ranked_lists_new = retrieve_ranked_lists(new_ranked_list)
    reference_docs = {q:ranked_lists_new[q][reference_index].replace("EPOCH","ROUND") for q in ranked_lists_new}

    winner_docs = {q:ranked_lists_new[q][:determine_indexes(reference_docs[q],ranked_lists_new[q])] for q in ranked_lists_new}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-"+round):
            doc_texts[doc]=a_doc_texts[doc]
    sentence_map=map_set_of_sentences(doc_texts,winner_docs)
    summaries = {}
    labels_file=open("labels_new", 'w')
    addition_to_file_names = round+"_"+str(reference_index)
    sentence_data_file = open("sentences_add_remove_"+addition_to_file_names, "w")
    index=1
    for query in sentence_map:
        print("in query",index, "out of",len(sentence_map))
        sys.stdout.flush()
        reference_doc = reference_docs[query].replace("EPOCH","ROUND")
        reference_text = doc_texts[reference_doc]
        reference_sentences = retrieve_sentences(reference_text)
        for sentence in sentence_map[query]:
            r_index = 1
            new_sentence = sentence_map[query][sentence].replace("\n", "")
            if not new_sentence:
                continue
            for reference_sentence in reference_sentences:
                run_name = sentence+"_"+str(r_index)
                reference_sentence=reference_sentence.replace("\n", "")
                if not reference_sentence:
                    continue
                modified_doc=reference_doc+"\n"+new_sentence
                summaries[reference_doc]=modified_doc
                add = open("/home/greg/auto_seo/scripts/add_remove_"+addition_to_file_names,'w',encoding="utf8")
                add.write(reference_doc+"@@@"+new_sentence.rstrip()+"@@@"+reference_sentence.rstrip()+"\n")
                sentence_data_file.write(run_name + "@@@" + new_sentence.rstrip() + "@@@" + reference_sentence.rstrip() + "\n")
                add.close()
                time.sleep(1)
                trec_text_file = create_trectext_original(doc_texts, summaries, "",[])
                features_dir = "Features"+addition_to_file_names
                feature_file = "features_"+run_name+"_"+addition_to_file_names
                create_features_file(features_dir, params.path_to_index, params.queries_xml,feature_file,"/home/greg/auto_seo/scripts/add_remove_"+addition_to_file_names,addition_to_file_names,new_ranked_list)
                index_doc_name = create_index_to_doc_name_dict(feature_file)
                scores_file = run_model(feature_file)
                results = retrieve_scores(index_doc_name, scores_file)
                lists=create_lists(results)
                addition = abs(lists[query].index(reference_doc))
                query = sentence.split("-")[2]
                labels_file.write(query + " 1 " + run_name + " " + str(addition) + "\n")
                r_index+=1
        index+=1
    labels_file.close()
    sentence_data_file.close()



