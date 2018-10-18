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
    leaves=[5,10,25,50]
    model_handler = mh.model_handler_LambdaMart(trees,leaves)
    evaluator.empty_validation_files("lm")
    for train,test in folds:
        model_handler.set_queries_to_folds(queries,test,fold_number)
        train_file = preprocess.create_train_file(X[train], y[train], queries[train],fold_number,"lm")
        test_file = preprocess.create_train_file(X[test], y[test], queries[test],fold_number,"lm","True")
        trecs = []
        for tree in trees:
            for leaf in leaves:
                model_file =model_handler.create_model_LambdaMart(number_of_leaves=leaf,number_of_trees=tree,train_file=train_file,fold=fold_number)
                scores_file  = model_handler.run_model(model_path=model_file,test_file=test_file,trees=tree,leaves=leaf)
                results = get_results(scores_file,test)
                trec_file = evaluator.create_trec_eval_file(test,queries,results,model_file,"lm",0)
                trecs.append(trec_file)
                trecs = list(set(trecs))
    for trec_file in trecs:
        print("working on ",trec_file)
        evaluator.run_trec_eval(trec_file,qrels_file)

