"""
Implementation of pairwise ranking using scikit-learn LinearSVC

Reference: "Large Margin Rank Boundaries for Ordinal Regression", R. Herbrich,
    T. Graepel, K. Obermayer.

Authors: Fabian Pedregosa <fabian@fseoane.net>
         Alexandre Gramfort <alexandre.gramfort@inria.fr>
"""

import itertools
import numpy as np
import os
import pickle
from sklearn import svm





class RankSVM(svm.LinearSVC):
    @staticmethod
    def transform_pairwise(X, y):

        X_new = []
        y_new = []
        y = np.asarray(y)
        if y.ndim == 1:
            y = np.c_[y, np.ones(y.shape[0])]
        comb = itertools.combinations(range(X.shape[0]), 2)
        for k, (i, j) in enumerate(comb):
            if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
                # skip if same target or different group
                continue
            X_new.append(X[i] - X[j])
            y_new.append(np.sign(y[i, 0] - y[j, 0]))
            # output balanced classes
            if y_new[-1] != (-1) ** k:
                y_new[-1] = - y_new[-1]
                X_new[-1] = - X_new[-1]
        return np.asarray(X_new), np.asarray(y_new).ravel()


    def __init__(self,C):
        super(RankSVM,self).__init__(C=C,loss="hinge")
    def fit(self, X, y,fold,C):

        models_folder = "svm_rank_own_models/" + str(fold) + "/"
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        model_file = models_folder + "model_" + str(C) + ".pkl"
        X_trans, y_trans = RankSVM.transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        with open(model_file,'wb') as model:
            pickle.dump(model,self)
        return model_file

    def predict(self, X,fold,C,model_file):

        predictions_folder = "svm_rank_own_score/" + str(fold) + "/"
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        predictions_file = predictions_folder + os.path.basename(model_file)

        result = np.dot(X, self.coef_.T)
        np.savetxt(predictions_file,result)
        return predictions_file


    def predict_test(self, X,fold,C,model_file):

        predictions_folder = "svm_rank_own_score/" + str(fold) + "/"
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        predictions_file = predictions_folder + os.path.basename(model_file)

        result = np.dot(X, self.coef_.T)
        np.savetxt(predictions_file,result)
        return predictions_file
