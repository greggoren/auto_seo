from CrossValidationUtils import preprocess_clueweb as p
from CrossValidationUtils import evaluator as e
import numpy as np
import os
import subprocess
import sys
from CrossValidationUtils import svm_handler as s
import operator
from sklearn.datasets import dump_svmlight_file

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


def cross_validation(features_file,qrels_file,summary_file,append_file = ""):
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(features_file, True)
    number_of_queries = len(set(queries))
    print("there are ", number_of_queries, 'queries')
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict(features_file)

    folds = preprocess.create_folds(X, y, queries, 5)
    fold_number = 1
    C_array = [0.1, 0.01, 0.0001]
    # C_array = [0.1, 0.01, 0.0001,1,10,100,10000]
    validated = set()
    scores = {}
    models = {}
    method ="svm_light"
    svm = s.svm_handler()
    for train, test in folds:

        evaluator.empty_validation_files(method)
        validated, validation_set, train_set = preprocess.create_validation_set(5, validated,
                                                                                set(train),
                                                                                number_of_queries, queries)
        number_of_queries_in_fold = len(set(queries[train_set]))
        train_file = preprocess.create_train_file(X[train_set], y[train_set], queries[train_set], fold_number,method)
        validation_file = preprocess.create_train_file(X[validation_set], y[validation_set], queries[validation_set], fold_number,method,True)
        test_file = preprocess.create_train_file_cv(X[test], y[test], queries[test], fold_number,method,True)
        if append_file:
            print("appending train features")
            run_bash_command("cat " + append_file + " >> " + train_file)
        for C in C_array:

            model_file = svm.learn_svm_light_model(train_file, fold_number,C,number_of_queries_in_fold)
            weights = recover_model(model_file)

            svm.w = weights
            scores_file = svm.run_svm_light_model(validation_file,model_file,fold_number)
            results = svm.retrieve_scores(validation_set,scores_file)
            score_file = evaluator.create_trec_eval_file(validation_set, queries, results, str(C),method, fold_number, True)
            score = evaluator.run_trec_eval(score_file, qrels_file)
            scores[C] = score
            models[C] = model_file
        max_C = max(scores.items(), key=operator.itemgetter(1))[0]
        print("on fold",fold_number,"chosen model:",max_C)
        chosen_model = models[max_C]
        test_scores_file=svm.run_svm_light_model(test_file,chosen_model,fold_number)
        results = svm.retrieve_scores(test, test_scores_file)
        trec_file = evaluator.create_trec_eval_file(test, queries, results, "", method, fold_number)
        fold_number += 1
    evaluator.order_trec_file(trec_file)
    run_bash_command("rm " + trec_file)
    evaluator.run_trec_eval_on_test(qrels_file,summary_file,method)

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
    cross_validation(features_file,qrels_file,summary_file,append_features)