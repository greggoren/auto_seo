from Crowdflower import create_full_ds_per_task as mturk_ds_creator
from utils import cosine_similarity
from SentenceRanking.sentence_features_experiment import get_sentence_vector
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import params
from w2v.train_word2vec import WordToVec
from CrossValidationUtils.rankSVM_crossvalidation import cross_validation
from Crowdflower.ban_non_coherent_docs import get_scores,sort_files_by_date,retrieve_initial_documents,ban_non_coherent_docs,get_dataset_stas,get_banned_queries
import numpy as np
from utils import run_bash_command
from Experiments.experiment_data_processor import create_trectext_original
from Crowdflower.seo_utils import *
from Experiments.experiment_data_processor import create_features_file
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
from SentenceRanking.sentence_parse import create_lists
import os



def read_seo_score(labels):
    scores = {}
    with open(labels) as labels_file:
        for line in labels_file:
            id = line.split()[2]
            score = int(line.split()[3].rstrip())
            scores[id]=score
    return scores


def get_level(score):
    demotion_level = 0
    if score >=0 and score <2:
        demotion_level=2
    elif score >=2 and score <4:
        demotion_level=1
    return demotion_level


def modify_seo_score_by_demotion(seo_scores, coherency_scores):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        demotion_level = get_level(coherency_score)
        new_score = max(current_score-demotion_level,0)
        new_scores[id] = new_score
    return new_scores


def create_harmonic_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    epsilon = 0.0001
    for id in seo_scores:
        current_score = seo_scores[id]+epsilon
        coherency_score = coherency_scores[id]
        new_coherency_score = coherency_score*(4.0/5)
        numerator = (1+beta**2)*new_coherency_score*current_score
        denominator = (beta**2)*new_coherency_score+current_score
        if denominator!=0:
            harmonic_mean = float(numerator)/denominator
        else:
            harmonic_mean = 0
        new_scores[id]=harmonic_mean
    return new_scores


def create_weighted_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        new_coherency_score = coherency_score * (4.0 / 5)
        new_score = current_score*beta+new_coherency_score*(1-beta)
        new_scores[id]=new_score
    return new_scores


def save_max_mix_stats(stats,row,query):
    features = list(row.keys())
    if query not in stats:
        stats[query]={}
    for feature in features:
        if feature not in stats[query]:
            stats[query][feature]={}
            stats[query][feature]["max"]  = row[feature]
            stats[query][feature]["min"] = row[feature]
        if row[feature]>stats[query][feature]["max"]:
            stats[query][feature]["max"] = row[feature]
        if row[feature]<stats[query][feature]["min"]:
            stats[query][feature]["min"] = row[feature]
    return stats


def normalize_feature(feature_value,max_min_stats,query,feature):
    if max_min_stats[query][feature]["max"]==max_min_stats[query][feature]["min"]:
        return 0
    denominator =max_min_stats[query][feature]["max"]-max_min_stats[query][feature]["min"]
    value = (feature_value-max_min_stats[query][feature]["min"])/denominator
    return value


def create_ws():
    lists_file = params.ranked_lists_file
    run_bash_command("cp "+lists_file+" /home/greg/auto_seo/scripts/workingSet")



def rewrite_fetures(new_scores, coherency_features_set, old_features_file, new_features_filename, coherency_features_names,qrels_name,max_min_stats):
    f = open(new_features_filename,"w")
    qrels = open(qrels_name,"w")
    with open(old_features_file) as file:
        for line in file:
            qid = line.split()[1]
            query = qid.split(":")[1]
            features = line.split()[2:-2]
            number_of_features = len(features)
            id = line.split(" # ")[1].rstrip()
            if id not in new_scores or id not in coherency_features_set:
                continue
            coherency_features = [str(i)+":"+str(normalize_feature(coherency_features_set[id][feature],max_min_stats,query,feature)) for i,feature in enumerate(coherency_features_names,start=number_of_features+1)]
            new_line = str(new_scores[id]) + " " + qid + " " + " ".join(features) + " " + " ".join(coherency_features) + " # " + id + "\n"
            f.write(new_line)
            qrels.write(query+" 0 "+id+" "+str(new_scores[id])+"\n")
    f.close()
    qrels.close()


def get_histogram(dataset):
    hist ={}
    for id in dataset:
        if dataset[id]<1:
            bucket =0
        elif dataset[id]<2:
            bucket =1
        elif dataset[id]<3:
            bucket =2
        elif dataset[id]<4:
            bucket =3
        elif dataset[id]<5:
            bucket=4
        else:
            bucket=5
        if bucket not in hist:
            hist[bucket]=0
        hist[bucket]+=1
    total_examples = sum([hist[b] for b in hist])
    for bucket in hist:
        hist[bucket]=round(hist[bucket]/total_examples,3)
    return hist

def get_average_score_increase_for_initial_rank(seo_scores, ranked_lists_file,initial_ranks):

    lists={}
    initial_ranks_stats ={}
    seen=[]
    with open(ranked_lists_file) as file:
        for line in file:
            query = line.split()[0]
            run_name = line.split()[2]
            key = initial_ranks[query]
            if query not in lists:
                lists[query]=[]
            if len(lists[query])>=5:
                if query in seen:
                    continue
                if key not in initial_ranks_stats:
                    initial_ranks_stats[key]={}
                    initial_ranks_stats[key][1]=[]
                    initial_ranks_stats[key][2]=[]
                    initial_ranks_stats[key][5]=[]
                initial_ranks_stats[key][1].append(np.mean(lists[query][:1]))
                initial_ranks_stats[key][2].append(np.mean(lists[query][:2]))
                initial_ranks_stats[key][5].append(np.mean(lists[query]))
                seen.append(query)
            lists[query].append(seo_scores[run_name])

    for key in initial_ranks_stats:
        for top in initial_ranks_stats[key]:

            initial_ranks_stats[key][top]=np.mean(initial_ranks_stats[key][top])
    for query in lists:
        key = initial_ranks[query]
        if "ge" not in initial_ranks_stats[key]:
            initial_ranks_stats[key]["ge"]=[]
            initial_ranks_stats[key]["eq"]=[]
            initial_ranks_stats[key]["le"]=[]
        if lists[query][0]>lists[query][1]:
            initial_ranks_stats[key]["ge"].append(1)
            initial_ranks_stats[key]["eq"].append(0)
            initial_ranks_stats[key]["le"].append(0)
        elif lists[query][0]==lists[query][1]:
            initial_ranks_stats[key]["ge"].append(0)
            initial_ranks_stats[key]["eq"].append(1)
            initial_ranks_stats[key]["le"].append(0)
        else:
            initial_ranks_stats[key]["ge"].append(0)
            initial_ranks_stats[key]["eq"].append(0)
            initial_ranks_stats[key]["le"].append(1)
    for key in initial_ranks_stats:
        for stat in ["ge","le","eq"]:
            initial_ranks_stats[key][stat]=np.mean(initial_ranks_stats[key][stat])
    return initial_ranks_stats


def read_sentences(filename):
    stats={}
    with open(filename) as file:
        for line in file:
            comb = line.split("@@@")[0]
            stats[comb]=[]
            sentence_in = line.split("@@@")[1]
            sentence_out = line.split("@@@")[2].rstrip()
            stats[comb].append(sentence_in)
            stats[comb].append(sentence_out)
    return stats


# def pick_best_sentences(score_file):
#     stats={}
#     with open(score_file) as scores:
#         for line in scores:
#             query = line.split()[0]
#             comb = line.split()[2]
#             if query not in stats:
#                 stats[query]=[]
#             if len(stats[query])<2:
#                 replacement_index = comb.split("_")[-1]
#                 prefix = "_".join(comb.split("_")[:-1])
#                 flag = True
#                 for existing_comb in stats[query]:
#                     e_replacement_index = existing_comb.split("_")[-1]
#                     e_prefix ="_".join(existing_comb.split("_")[:-1])
#
#                     if e_replacement_index==replacement_index or e_prefix==prefix:
#                         flag=False
#                 if flag:
#                     stats[query].append(comb)
#     return stats

def write_add_remove_file(file,combs,query,sentences,reference_doc):
    f = open(file,"w")
    combs_of_query = combs[query]
    new_sentence_in = sentences[combs_of_query[0]][0]+" "+sentences[combs_of_query[1]][0]
    new_sentence_out = sentences[combs_of_query[0]][1]+" "+sentences[combs_of_query[1]][1]
    line = reference_doc+"@@@"+new_sentence_in+"@@@"+new_sentence_out+"\n"
    f.write(line)
    f.close()


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

def run_reranking(reference_doc,query,labels_file,add_remove_file,beta="-"):
    features_dir = "Features"
    feature_file = "features_" +query
    create_features_file(features_dir, params.path_to_index, params.queries_xml, feature_file,
                         add_remove_file, "")
    index_doc_name = create_index_to_doc_name_dict(feature_file)
    scores_file = run_model(feature_file)
    results = retrieve_scores(index_doc_name, scores_file)
    lists = create_lists(results)
    addition = abs(lists[query].index(reference_doc))
    labels_file.write(query + " 1 " +beta+ " " + str(addition) + "\n")


def run_bots_and_rerank(method, doc_texts, new_features_file,new_qrels_file,sentences,reference_docs,seo_features_file,dummy_scores,labels_file,beta="-"):
    chosen_models_file_name = "chosen_models_"+method
    chosen_models = read_chosen_model_file(chosen_models_file_name)

    # final_trec_file = run_chosen_model_for_stats(chosen_models, method, new_features_file, doc_name_index,
    #                                              new_features_file)
    final_trec_file=cross_validation(new_features_file, new_qrels_file, "summary_labels_"+method+".tex",
                     "svm_rank",
                     ["map", "ndcg", "P.2", "P.5"], "")
    best_sentences = pick_best_sentences(final_trec_file)

    for query in reference_docs:
        if query in banned_queries or query not in best_sentences:
            continue

        doc = reference_docs[query]
        if query=="180":
            print(doc_texts[doc],flush=True)
        chosen_comb = best_sentences[query]
        doc_texts = save_modified_file(doc_texts,sentences, chosen_comb, doc)
        if query=="180":
            print(doc_texts[doc],flush=True)

    new_coherence_features_set, max_min_stats = create_coherency_features(ref_index=-1,
                                                                          ranked_list_new_file="ranked_lists/trec_file04",
                                                                          doc_text_modified=doc_texts)
    rewrite_fetures(dummy_scores, new_coherence_features_set, seo_features_file, new_features_file+"_exp",
                    coherency_features, "dummy_q", max_min_stats)
    doc_name_index = create_index_to_doc_name_dict(new_features_file+"_exp")
    final_trec_file = run_chosen_model_for_stats(chosen_models, method, new_features_file+"_exp", doc_name_index,
                                                 new_features_file,str(beta))


    new_best_sentences = pick_best_sentences(final_trec_file, best_sentences)

    print(new_best_sentences,flush=True)
    for query in reference_docs:

        if query in banned_queries or query not in best_sentences:
            continue
        reference_doc = reference_docs[query]
        write_add_remove_file(add_remove_file, new_best_sentences, query, sentences, reference_doc)
        run_reranking(reference_doc, query, labels_file, add_remove_file,beta)



if __name__=="__main__":
    ranked_lists_old = retrieve_ranked_lists(params.ranked_lists_file)
    ranked_lists_new = retrieve_ranked_lists("ranked_lists/trec_file04")
    sentences = read_sentences("/home/greg/auto_seo/SentenceRanking/sentences_add_remove")
    reference_docs = {q: ranked_lists_old[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists_old}
    initial_ranks = {q:ranked_lists_new[q].index(reference_docs[q])+1 for q in reference_docs}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts = {}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc] = a_doc_texts[doc]
    dir = "nimo_annotations"
    sorted_files = sort_files_by_date(dir)
    add_remove_file = "/home/greg/auto_seo/scripts/add_remove"
    original_docs = retrieve_initial_documents()
    scores={}
    for k in range(4):
        needed_file = sorted_files[k]
        scores = get_scores(scores,dir + "/" + needed_file,original_docs)
    banned_queries = get_banned_queries(scores,reference_docs)

    ident_filename_fe = "figure-eight/ident_current.csv"
    ident_filename_mturk = "Mturk/Manipulated_Document_Identification.csv"
    ident_filename_mturk_addition = "Mturk/Manipulated_Document_Identification_add.csv"
    ident_fe = mturk_ds_creator.read_ds_fe(ident_filename_fe, True)
    ident_mturk = mturk_ds_creator.read_ds_mturk(ident_filename_mturk, True)
    ident_mturk_addition = mturk_ds_creator.read_ds_mturk(ident_filename_mturk_addition, True)
    ident_mturk = mturk_ds_creator.update_dict(ident_mturk, ident_mturk_addition)
    ident_results = mturk_ds_creator.combine_results(ident_fe, ident_mturk)
    sentence_filename_fe = "figure-eight/sentence_current.csv"
    sentence_filename_mturk = "Mturk/Sentence_Identification.csv"
    sentence_filename_mturk_addition = "Mturk/Sentence_Identification_add.csv"
    sentence_filename_mturk_new = "Mturk/Sentence_Identification11.csv"
    sentence_fe = mturk_ds_creator.read_ds_fe(sentence_filename_fe)
    sentence_mturk = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk)
    sentence_mturk_addtion = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk_addition)
    sentence_mturk_new = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk_new)
    sentence_mturk = mturk_ds_creator.update_dict(sentence_mturk, sentence_mturk_new)
    sentence_mturk = mturk_ds_creator.update_dict(sentence_mturk, sentence_mturk_addtion)
    sentence_results = mturk_ds_creator.combine_results(sentence_fe, sentence_mturk)

    sentence_tags = mturk_ds_creator.get_tags(sentence_results)
    ident_tags = mturk_ds_creator.get_tags(ident_results)
    tmp_aggregated_results = mturk_ds_creator.aggregate_results(sentence_tags,ident_tags)
    aggregated_results = ban_non_coherent_docs(banned_queries,tmp_aggregated_results)
    # dummy_scores = {run_name:"0" for run_name in sentences}
    coherency_features = ["similarity_to_prev", "similarity_to_ref_sentence", "similarity_to_pred",
                          "similarity_to_prev_ref", "similarity_to_pred_ref"]
    seo_scores_file = "labels_new_final"
    tmp_seo_scores = read_seo_score(seo_scores_file)
    seo_scores = ban_non_coherent_docs(banned_queries,tmp_seo_scores)
    modified_scores= modify_seo_score_by_demotion(seo_scores,aggregated_results)
    seo_features_file = "new_sentence_features"
    coherency_features_set,max_min_stats = create_coherency_features(ranked_list_new_file="ranked_lists/trec_file04",ref_index=-1,doc_text_modified=doc_texts)
    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"
    rewrite_fetures(modified_scores,coherency_features_set,seo_features_file,new_features_with_demotion_file,coherency_features,new_qrels_with_demotion_file,max_min_stats)
    labels_file = "labels_demotion"
    f=open(labels_file,"w")
    run_bots_and_rerank("demotion",doc_texts.copy(),new_features_with_demotion_file,new_qrels_with_demotion_file,sentences,reference_docs,seo_features_file,modified_scores,f,"")
    f.close()



    stats_harmonic={}
    betas = [0,0.5,1,2]
    label_file = "labels_harmonic"
    f = open(label_file, "w")
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_"+str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_"+str(beta)
        harmonic_mean_scores={}
        harmonic_mean_scores = create_harmonic_mean_score(seo_scores,aggregated_results,beta)
        rewrite_fetures(harmonic_mean_scores, coherency_features_set, seo_features_file, new_features_with_harmonic_file,
                        coherency_features, new_qrels_with_harmonic_file,max_min_stats)
        run_bots_and_rerank("harmonic", doc_texts.copy(), new_features_with_harmonic_file,new_qrels_with_harmonic_file,sentences ,reference_docs, seo_features_file,
                            harmonic_mean_scores, f,str(beta))
    f.close()

    stats_weighted = {}
    label_file = "labels_weighted"
    f = open(label_file, "w")
    betas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_"+str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_"+str(beta)
        weighted_mean_scores={}
        weighted_mean_scores = create_weighted_mean_score(seo_scores, aggregated_results,beta)
        rewrite_fetures(weighted_mean_scores, coherency_features_set, seo_features_file, new_features_with_weighted_file,
                        coherency_features, new_qrels_with_weighted_file,max_min_stats)
        run_bots_and_rerank("weighted", doc_texts.copy(), new_features_with_weighted_file, new_qrels_with_weighted_file,sentences,reference_docs, seo_features_file,
                            weighted_mean_scores, f, str(beta))
    f.close()
