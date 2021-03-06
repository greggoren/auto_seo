import CrossValidationUtils.preprocess_clueweb as p
import CrossValidationUtils.lambdaMART_models_handler as mh
import CrossValidationUtils.evaluator as e
from utils import run_bash_command
import sys


def get_results(score_file,test_indices):
    results={}
    with open(score_file) as scores:
        for index,score in enumerate(scores):
            results[test_indices[index]]=score
    return results


#
if __name__=="__main__":
    features_file = sys.argv[1]
    print("features file=", features_file)
    qrels_file = sys.argv[2]
    print("qrels file=", qrels_file)
    if len(sys.argv) < 4:
        summary_file = "summary_lm.tex"
    else:
        summary_file = sys.argv[3]
    if len(sys.argv) < 5:
        append_features = ""
    else:
        append_features = sys.argv[4]
    number_of_folds = 5
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(features_file,True)
    number_of_queries = len(set(queries))
    metrics =["map","ndcg_cut.20","P.10","P.5"]
    evaluator = e.eval(metrics)
    evaluator.create_index_to_doc_name_dict(features_file)
    evaluator.remove_score_file_from_last_run("lm")

    folds = preprocess.create_folds(X, y, queries, number_of_folds)
    fold_number = 1
    trees = [250,500]
    # trees = [250,]
    leaves=[5,10,25,50]
    # leaves=[5,]
    model_handler = mh.model_handler_LambdaMart(trees,leaves)
    validated = set()
    evaluator.empty_validation_files("lm")
    for train,test in folds:
        validated, validation_set, train_set = preprocess.create_validation_set(number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        model_handler.set_queries_to_folds(queries,test,fold_number)
        train_file = preprocess.create_train_file(X[train_set], y[train_set], queries[train_set],fold_number,"lm")
        if append_features:
            print("appending train features")
            run_bash_command("cat "+append_features+" >> "+train_file)

        validation_file = preprocess.create_train_file(X[validation_set], y[validation_set], queries[validation_set], fold_number,"lm",True)
        test_file = preprocess.create_train_file_cv(X[test], y[test], queries[test], fold_number,"lm", True)
        model_handler.fit_model_on_train_set_and_choose_best(train_file,validation_file,validation_set,queries,fold_number,evaluator,qrels_file)
        scores_file=model_handler.run_model_on_test(test_file,fold_number)
        results = model_handler.retrieve_scores(test,scores_file)
        test_trec = evaluator.create_trec_eval_file(test,queries,results,"_".join([str(a) for a in model_handler.chosen_model_per_fold[fold_number]]),"lm",fold_number)
        fold_number += 1
    evaluator.order_trec_file(test_trec)
    evaluator.run_trec_eval_on_test(qrels_file,summary_file,"lm")
    run_bash_command("rm " + test_trec)