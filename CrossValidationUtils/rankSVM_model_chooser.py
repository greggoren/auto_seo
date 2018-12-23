import CrossValidationUtils.preprocess_clueweb as p
from CrossValidationUtils import svm_handler as s
import CrossValidationUtils.evaluator as e
from utils import run_bash_command
import sys
import os


def choose_model(features_file,qrels_file,label_method,beta=""):
    number_of_folds = 5
    preprocess = p.preprocess()
    X, y, queries = preprocess.retrieve_data_from_file(features_file, True)
    number_of_queries = len(set(queries))
    metrics = ["map", "ndcg", "P.2", "P.5"]
    evaluator = e.eval(metrics)
    evaluator.create_index_to_doc_name_dict(features_file)
    evaluator.remove_score_file_from_last_run("svm_rank")
    folds = preprocess.create_folds(X, y, queries, number_of_folds)
    fold_number = 1
    C = [0.1, 0.01, 0.001]
    model_handler = s.svm_handler()
    evaluator.empty_validation_files("svm_rank")
    trecs = []
    for train, test in folds:
        # model_handler.set_queries_to_folds(queries, test, fold_number)
        train_file = preprocess.create_train_file(X[train], y[train], queries[train], fold_number, "svm_rank")
        test_file = preprocess.create_train_file(X[test], y[test], queries[test], fold_number, "svm_rank", True)
        for c_value in C:
            model_file = model_handler.learn_svm_rank_model(train_file, fold_number, c_value)
            model_name = os.path.basename(model_file).replace(".txt", "")
            scores_file = model_handler.run_svm_rank_model(test_file, model_file, fold_number)
            results = model_handler.retrieve_scores(test, scores_file)
            trec_file = evaluator.create_trec_eval_file(test_indices=test, queries=queries, results=results,
                                                        model=model_name, method="svm_rank", fold=0, validation=True)
            trecs.append(trec_file)
            trecs = list(set(trecs))
        fold_number += 1
    scores = {}
    for trec_file in trecs:
        print("working on ", trec_file)
        score = evaluator.run_trec_eval(trec_file, qrels_file)
        model = os.path.basename(trec_file)
        scores[model] = score

    sorted_models = sorted(list(scores.keys()), key=lambda x: scores[x], reverse=True)
    for file in sorted_models:
        print(file, scores[file])
    f = open("chosen_models_" + label_method, "a")
    add=""
    if beta:
       add= "_" + beta
    f.write(label_method+add+" "+sorted_models[0]+"\n")
    f.close()


#
if __name__=="__main__":
    features_file = sys.argv[1]
    print("features file=", features_file)
    qrels_file = sys.argv[2]
    print("qrels file=", qrels_file)
    label_method = sys.argv[3]
    print("label method=", label_method)
    choose_model(features_file,qrels_file,label_method)




