import numpy as np


class svm_sgd:

    def __init__(self,C=None):
        self.C=C
        self.w = None


    def check_prediction(self,y_k):

        if np.dot(self.w,y_k.T)>1:
            return True
        return False


    def check_validation(self,validation,tags,X):
        errors=0
        for index in validation:
            y_k=X[index]*tags[index]
            tmp=np.dot(self.w,y_k.T)
            if tmp<1:
                errors+=1
        return float(errors)/len(validation)


    def predict(self,X,queries,test_indices,eval,validation=None):
        results = {}
        for index in test_indices:
            results[index] = np.dot(self.w,X[index].T)
        return eval.create_trec_eval_file(test_indices,queries,results,str(self.C),validation)