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
    number_of_folds = 5
    preprocess = p.preprocess()
    X,y,queries=preprocess.retrieve_data_from_file(features_file,True)
    number_of_queries = len(set(queries))
    evaluator = e.eval()
    evaluator.create_index_to_doc_name_dict(features_file)
    evaluator.remove_score_file_from_last_run("lm")

    folds = preprocess.create_folds(X, y, queries, number_of_folds)
    fold_number = 1
    trees = [250,500]
    leaves=[5,10,25,50]
    model_handler = mh.model_handler_LambdaMart(trees,leaves)
    validated = set()
    evaluator.empty_validation_files("lm")
    for train,test in folds:
        validated, validation_set, train_set = preprocess.create_validation_set(number_of_folds, validated, set(train),
                                                                                number_of_queries, queries)
        validation_set=list(validation_set)
        model_handler.set_queries_to_folds(queries,test,fold_number)
        train_file = preprocess.create_train_file(X[train_set], y[train_set], queries[train_set])
        validation_file = preprocess.create_train_file(X[validation_set], y[validation_set], queries[validation_set], True)
        test_file = preprocess.create_train_file_cv(X[test], y[test], queries[test], fold_number, True)
        model_handler.fit_model_on_train_set_and_choose_best(train_file,validation_file,validation_set,queries,fold_number,evaluator,qrels_file)
        scores_file=model_handler.run_model_on_test(test_file,fold_number)
        results = model_handler.retrieve_scores(test,scores_file)
        test_trec = evaluator.create_trec_eval_file(test,queries,results,"_".join([str(a) for a in model_handler.chosen_model_per_fold[fold_number]],fold_number,"lm"))
        fold_number += 1
    evaluator.run_trec_eval_on_test(qrels_file,summary_file,"lm")
    run_bash_command("rm " + test_trec)