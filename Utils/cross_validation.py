from Utils import preprocess_clueweb as p
from Utils import evaluator as e
import numpy as np
import os
import subprocess
import sys
from Utils import SVM_SGD as s
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


def learn_svm(C, train_file, fold):
    if not os.path.exists("models/" + str(fold)):
        try:
            os.makedirs("models/" + str(fold))
        except:
            print("weird behaviour")
            print(C, train_file, fold)

    learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + "models/" + str(
        fold) + "/svm_model" + str(C) + ".txt"
    for output_line in run_command(learning_command):
        print(output_line)
    return "models/" + str(fold) + "/svm_model" + str(C) + ".txt"


def run_svm(C, model_file, test_file, fold):
    score_path = "scores/" + str(fold)
    if not os.path.exists(score_path):
        try:
            os.makedirs(score_path)
        except:
            print("collition")
    rank_command = "./svm_rank_classify " + test_file + " " + model_file + " " + score_path + "/" + str(C)
    for output_line in run_command(rank_command):
        print(output_line)
    return score_path + "/" + str(C)


def retrieve_scores(test_indices, score_file):
    with open(score_file) as scores:
        results = {test_indices[i]: score.rstrip() for i, score in enumerate(scores)}
        return results


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


def train_score(train_file, test_file, fold_number, C):
    model_file = learn_svm(C, train_file, fold_number)
    score_file = run_svm(C, model_file, test_file, fold_number)
    return score_file

if __name__ == "__main__":
    features_file = sys.argv[1]
    print("features file=",features_file)
    qrels_file = sys.argv[2]
    print("qrels file=", qrels_file)
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(features_file, True)
    number_of_queries = len(set(queries))
    print("there are ",number_of_queries,'queries')
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict(features_file)

    folds = preprocess.create_folds(X, y, queries, 5)
    fold_number = 1
    C_array = [0.1, 0.01, 0.001]
    validated = set()
    scores = {}
    models = {}
    for train, test in folds:
        evaluator.empty_validation_files()
        validated, validation_set, train_set = preprocess.create_validation_set(5, validated,
                                                                                set(train),
                                                                                number_of_queries, queries)
        train_file = "train" + str(fold_number) + ".txt"
        run_bash_command("rm " + train_file)
        dump_svmlight_file(X[train], y[train], train_file, query_id=queries[train], zero_based=False)
        for C in C_array:
            model_file = learn_svm(C, train_file, fold_number)
            weights = recover_model(model_file)
            svm = s.svm_sgd(C)
            svm.w = weights
            score_file = svm.predict(X, queries, validation_set, evaluator, True)
            score = evaluator.run_trec_eval(score_file,qrels_file)
            scores[svm.C] = score
            models[svm.C] = svm
        max_C = max(scores.items(), key=operator.itemgetter(1))[0]
        chosen_model = models[max_C]
        chosen_model.predict(X, queries, test, evaluator)
        fold_number += 1
    evaluator.run_trec_eval_on_test(qrels_file)