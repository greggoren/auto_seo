from QualityImportance import SVM_SGD as svm_sgd
import operator
class models_handler():

    def __init__(self,C_array):
        self.models = []
        for C in C_array:
            self.models.append(svm_sgd.svm_sgd(C))

        self.query_to_fold_index ={}
        self.weights_index = {}
        self.chosen_model_per_fold = {}

    def set_queries_to_folds(self,queries,test_indices,fold):
        set_of_queries = set(queries[test_indices])
        tmp = {a:fold for a in set_of_queries}
        self.query_to_fold_index.update(tmp)


    def fit_model_on_train_set_and_choose_best(self,X,X_i,y_i,validation_indices,fold,queries,evaluator):
        print("fitting models on fold",fold)
        weights = {}
        scores={}
        for svm in self.models:
            svm.fit(X_i,y_i)
            weights[svm.C]=svm.w
            score_file = svm.predict(X, queries, validation_indices,evaluator, True)
            score = evaluator.run_trec_eval(score_file)
            scores[svm.C] = score
        max_C=max(scores.items(), key=operator.itemgetter(1))[0]
        print("on fold",str(fold),"the chosen model is",str(max_C))
        self.weights_index[fold] = weights[max_C]
        self.chosen_model_per_fold[fold]=max_C


    def predict(self,X,queries,test_indices,fold,eval):
        svm = svm_sgd.svm_sgd(C=self.chosen_model_per_fold[fold])
        svm.w = self.weights_index[fold]
        svm.predict(X,queries,test_indices,eval)