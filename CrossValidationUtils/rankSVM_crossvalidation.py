from CrossValidationUtils import preprocess_clueweb as p
from CrossValidationUtils import evaluator as e

import numpy as np
import os
import subprocess
import sys
from CrossValidationUtils import svm_handler as s
import operator
import pickle
from sklearn.datasets import dump_svmlight_file
from scipy.stats import ttest_rel
from itertools import product
from SentenceRanking.new_data_experiment.perm import permutation_test
import random
def run_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT,
                         shell=True)
    return iter(p.stdout.readline, b'')

def run_bash_command(command):
    p = subprocess.Popen(command,
                         stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, shell=True)

    out, err = p.communicate()
    return out

def upload_models(models_dir):

    models = []
    for root, dirs, files in os.walk(models_dir):
        print(root, files, dirs)
        for file in files:
            model_file = root + "/" + file
            w = recover_model(model_file)
            if np.linalg.norm(w) < 15:
                model = float(model_file.split("svm_model")[1])
                models.append(model)

    return models


def random_items(iterator, items_wanted=1):
    selected_items = [None] * items_wanted

    for item_index, item in enumerate(iterator):
        for selected_item_index in range(items_wanted):
            if not random.randint(0, item_index):
                selected_items[selected_item_index] = item

    return selected_items

# from mlxtend.evaluate import permutation_test
def permutation_test1(sample_a, sample_b):
    real_diff = abs(np.mean(sample_a) - np.mean(sample_b))
    x = product([1, -1], repeat=len(sample_a))
    if len(sample_a)>15:
        n_perm=10000
    else:
        n_perm = len(x)

    total = 0
    random_items_x  = random_items(iter(x),n_perm)
    for row in random_items_x:

        a_mean = []
        b_mean = []
        for index, val in enumerate(row):
            if val > 0:
                a_mean.append(sample_a[index])
                b_mean.append(sample_b[index])
            else:
                a_mean.append(sample_b[index])
                b_mean.append(sample_a[index])
        current_diff = abs(np.mean(a_mean) - np.mean(b_mean))
        if current_diff >= real_diff:
            total += 1
    return total / n_perm


# def learn_svm(C, train_file, fold):
#     if not os.path.exists("models/" + str(fold)):
#         try:
#             os.makedirs("models/" + str(fold))
#         except:
#             print("weird behaviour")
#             print(C, train_file, fold)
#
#     learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + "models/" + str(
#         fold) + "/svm_model" + str(C) + ".txt"
#     for output_line in run_command(learning_command):
#         print(output_line)
#     return "models/" + str(fold) + "/svm_model" + str(C) + ".txt"
#
#
# def run_svm(C, model_file, test_file, fold):
#     score_path = "scores/" + str(fold)
#     if not os.path.exists(score_path):
#         try:
#             os.makedirs(score_path)
#         except:
#             print("collition")
#     rank_command = "./svm_rank_classify " + test_file + " " + model_file + " " + score_path + "/" + str(C)
#     for output_line in run_command(rank_command):
#         print(output_line)
#     return score_path + "/" + str(C)
#
#
# def retrieve_scores(test_indices, score_file):
#     with open(score_file) as scores:
#         results = {test_indices[i]: score.rstrip() for i, score in enumerate(scores)}
#         return results


def get_average_query_rank_promotion(seo_scores, ranked_lists_file):
    lists = {}
    stats = {}
    with open(ranked_lists_file) as file:
        for line in file:
            query = line.split()[0]
            run_name = line.split()[2]
            if query not in lists:
                lists[query] = []
            if len(lists[query]) >= 5:
                continue
            lists[query].append(seo_scores[run_name])
    stats[1]={q:np.mean(lists[q][:1]) for q in lists}
    stats[2]={q:np.mean(lists[q][:2]) for q in lists}
    stats[5]={q:np.mean(lists[q]) for q in lists}
    return stats

def get_average_score_increase(seo_scores, ranked_lists_file,write=False):
    lists={}
    stats={}
    with open(ranked_lists_file) as file:
        for line in file:
            query = line.split()[0]
            key = query[3:]
            run_name = line.split()[2]
            if query not in lists:
                lists[query]=[]
            if len(lists[query])>=5:
                continue
            lists[query].append(seo_scores[key][run_name])
    if write:
        f = open("weighted_dubug","w")
        f.write(str(lists))
        f.close()
    stats[1] = np.mean([np.mean(lists[q][:1]) for q in lists])
    stats[2] = np.mean([np.mean(lists[q][:2]) for q in lists])
    stats[5] = np.mean([np.mean(lists[q]) for q in lists])
    stats["ge"]=sum([1 for q in lists if lists[q][0]>lists[q][1]])/len(lists)
    stats["eq"]=sum([1 for q in lists if lists[q][0]==lists[q][1]])/len(lists)
    stats["le"]=sum([1 for q in lists if lists[q][0]<lists[q][1]])/len(lists)
    firsts = np.array([lists[q][0] for q in sorted(lists.keys())])
    return stats,firsts

def recover_model(model):
    indexes_covered = []
    weights = []
    with open(model) as model_file:
        for line in model_file:
            if line.__contains__(":"):
                wheights = line.split()
                wheights_length = len(wheights)

                for index in range(1, wheights_length - 1):

                    feature_id = int(wheights[index].split(":")[0])
                    if index < feature_id:
                        for repair in range(index, feature_id):
                            if repair in indexes_covered:
                                continue
                            weights.append(0)
                            indexes_covered.append(repair)
                    weights.append(float(wheights[index].split(":")[1]))
                    indexes_covered.append(feature_id)
    return np.array(weights)


# def train_score(train_file, test_file, fold_number, C):
#     model_file = learn_svm(C, train_file, fold_number)
#     score_file = run_svm(C, model_file, test_file, fold_number)
#     return score_file

def read_qrels(qrels):
    result ={}


def discover_significance_relevance(cv_stats,random_stats):
    metric_significance_sign = {}
    for metric in cv_stats:
        cv_values_vector =[cv_stats[metric][q] for q in sorted(list(cv_stats[metric].keys()))]
        random_values_vector =[random_stats[metric][q] for q in sorted(list(cv_stats[metric].keys()))]
        ttest_value = ttest_rel(cv_values_vector,random_values_vector)
        # ttest_value = permutation_test(cv_values_vector,random_values_vector,method='approximate',num_rounds=50000,seed=9011)
        sign =""
        if ttest_value[1]<=0.1:
            sign="^*"
        # print("ttest = ",ttest_value[1],sign)
        metric_significance_sign[metric]=sign
    return metric_significance_sign


def discover_significance_rank_promotior(cv_vector,random_vector,metric_significance_sign):
    ttest_value = ttest_rel(cv_vector,random_vector)
    # ttest_value = permutation_test(cv_vector,random_vector,method='approximate',num_rounds=50000,seed=9011)
    sign =""
    if ttest_value[1]<=0.1:
            sign="^*"
    metric_significance_sign[1]=sign
    return metric_significance_sign

def cross_validation(features_file,qrels_file,summary_file,method,metrics,append_file = "",seo_scores=False,run_random_for_significance=None):
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(features_file, True)
    number_of_queries = len(set(queries))
    print("there are ", number_of_queries, 'queries')
    evaluator = e.eval(metrics)
    evaluator.create_index_to_doc_name_dict(features_file)

    folds = preprocess.create_folds(X, y, queries, 5)
    fold_number = 1
    # C_array = [0.1, 0.01, 0.0001,1,10,100,10000]
    C_array = [0.1, 0.01, 0.0001]
    validated = set()
    scores = {}
    total_models = {}
    svm = s.svm_handler()
    evaluator.empty_validation_files(method)
    for train, test in folds:
        models = {}
        validated, validation_set, train_set = preprocess.create_validation_set(5, validated,
                                                                                set(train),
                                                                                number_of_queries, queries)
        train_set = sorted(list(train_set))
        validation_set=sorted(list(validation_set))
        test_set = sorted(list(test))
        train_file = preprocess.create_train_file(X[train_set], y[train_set], queries[train_set], fold_number,method)
        validation_file = preprocess.create_train_file(X[validation_set], y[validation_set], queries[validation_set], fold_number,method,True)
        test_file = preprocess.create_train_file_cv(X[test_set], y[test_set], queries[test_set], fold_number,method,True)
        # if append_file:
        #     print("appending train features")
        #     run_bash_command("cat " + append_file + " >> " + train_file)
        for C in C_array:

            model_file = svm.learn_svm_rank_model(train_file, fold_number,C)
            weights = recover_model(model_file)

            svm.w = weights
            scores_file = svm.run_svm_rank_model(validation_file,model_file,fold_number)
            results = svm.retrieve_scores(validation_set,scores_file)
            score_file = evaluator.create_trec_eval_file(validation_set, queries, results, str(C),method, fold_number, True)
            score = evaluator.run_trec_eval(score_file, qrels_file)
            scores[C] = score
            models[C] = model_file
        max_C = max(scores.items(), key=operator.itemgetter(1))[0]
        print("on fold",fold_number,"chosen model:",max_C)
        chosen_model = models[max_C]
        total_models[fold_number]=chosen_model
        test_scores_file=svm.run_svm_rank_model(test_file,chosen_model,fold_number)
        results = svm.retrieve_scores(test_set, test_scores_file)
        trec_file = evaluator.create_trec_eval_file(test_set, queries, results, "", method, fold_number)
        fold_number += 1
    final_trec_file = evaluator.order_trec_file(trec_file)
    run_bash_command("rm " + trec_file)
    # sum=[]
    # for i in total_models:
    #     w = recover_model(total_models[i])
    #     print(w)
    #     if sum==[]:
    #         sum=w
    #     else:
    #         sum+=w
    #     print(sum)
    #
    # average = sum/len(total_models)
    # print(average)
    # f = open(qrels_file+"_averaged_weights.pkl","wb")
    # pickle.dump(average,f)
    # f.close()
    if seo_scores:
        increase_rank_stats,cv_firsts = get_average_score_increase(seo_scores,final_trec_file)
    else:
        increase_rank_stats=False
    stats ,significance_data_cv = evaluator.run_trec_eval_by_query(qrels_file,final_trec_file)
    random_significance_data,random_firsts = run_random_for_significance(features_file,qrels_file,"sig_test",seo_scores=seo_scores)
    sig_signs = discover_significance_relevance(significance_data_cv,random_significance_data)
    sig_signs = discover_significance_rank_promotior(cv_firsts,random_firsts,sig_signs)
    evaluator.run_trec_eval_on_test(qrels_file,summary_file,method,None,increase_rank_stats,sig_signs)
    del X
    del y
    del queries
    return final_trec_file

if __name__ == "__main__":
    features_file = sys.argv[1]
    print("features file=",features_file)
    qrels_file = sys.argv[2]
    print("qrels file=", qrels_file)
    if len(sys.argv)<4:
        summary_file = "svm_summary.tex"
    else:
        summary_file = sys.argv[3]
    if len(sys.argv) < 5:
        append_features = ""
    else:
        append_features = sys.argv[4]
    metrics = ["map","ndcg_cut.5","P.1","P.5"]
    cross_validation(features_file,qrels_file,summary_file,"svm_rank",metrics,append_features)