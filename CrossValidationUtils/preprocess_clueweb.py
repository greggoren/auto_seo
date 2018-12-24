import numpy as np
import itertools
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import GroupKFold
import math



class preprocess:
    def __init__(self):
        ""

    def retrieve_data_from_file(self, file, normalized):
        print("loading svmLight file")
        X, y, groups = load_svmlight_file(file, query_id=True)
        print("loading complete")
        X = X.toarray()
        if not normalized:
            X = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        return X, y, groups

    def create_data_set(self, X, y, groups):
        print("creating data set")
        data = []
        labels = []
        k = 0
        unique_groups = set(groups)
        for group in unique_groups:
            relevant_indexes = np.where(groups == group)[0]
            comb = itertools.combinations(relevant_indexes, 2)
            for (i, j) in comb:
                if (y[i] == y[j]):
                    continue
                data.append(X[i] - X[j])
                labels.append(np.sign(y[i] - y[j]))
                if labels[-1] != (-1) ** k:  # to get a balanced data set
                    labels[-1] *= -1
                    data[-1] *= -1
                k += 1
        print("finished data set creation")
        print("number of points", len(data))
        return data, labels

    def create_data_set_opt(self, X, y, groups):
        print("creating data set")
        data = []
        labels = []
        k = 0
        unique_groups = set(groups)
        for group in unique_groups:
            relevant_indexes = np.where(groups == group)[0]
            comb = itertools.combinations(relevant_indexes, 2)
            for (i, j) in comb:
                if (y[i] == y[j]):
                    continue
                data.append(X[i] - X[j])
                labels.append(np.sign(y[i] - y[j]))
                if labels[-1] != (-1) ** k:  # to get a balanced data set
                    labels[-1] *= -1
                    data[-1] *= -1
                k += 1
        print("finished data set creation")
        print("number of points", len(data))
        return data, labels

    def create_folds(self, X, y, groups, number_of_folds):
        kf = GroupKFold(number_of_folds)
        return kf.split(X, y, groups)

    def create_test_train_split_cluweb(self, groups):
        test_indices = []
        train_indices = []
        unique_groups = set(groups)
        for group in unique_groups:
            relevant_indices = np.where(groups == group)[0]
            if group <= 150:
                train_indices.extend(relevant_indices)
            else:
                test_indices.extend(relevant_indices)
        return train_indices, test_indices
    #
    def create_train_file(self, X, y, queries, fold, method, test=False):
        add = ""
        if test:
            add = "_validation"
        train_file = "features_" + method + add + "_" + str(fold)
        with open(train_file, 'w') as feature_file:
            for i, doc in enumerate(X):
                features = " ".join([str(a + 1) + ":" + str(b) for a, b in enumerate(doc)])
                line = str(float(y[i])) + " qid:" + str(queries[i]).zfill(3) + " " + features + "\n"
                feature_file.write(line)
        return train_file

    def create_train_file_cv(self, X, y, queries, fold, method,test=False):
        add = ""
        if test:
            add = "_test"
        train_file = "features_"+method  + add+"_"+ str(fold)
        with open(train_file, 'w') as feature_file:
            for i, doc in enumerate(X):
                features = " ".join([str(a + 1) + ":" + str(b) for a, b in enumerate(doc)])
                line = str(float(y[i])) + " qid:" + str(queries[i]).zfill(3) + " " + features + "\n"
                feature_file.write(line)
        return train_file

    def create_validation_set(self, number_of_folds, already_been_in_validation_indices, train_indices,
                              number_of_queries, queries):
        validation_queries = set()
        number_of_queries_in_set = math.ceil(float(float(number_of_queries) / number_of_folds))
        working_set = train_indices - already_been_in_validation_indices

        validation_set = set()
        working_set=sorted(list(working_set))
        for index in working_set:
            if len(validation_queries)<number_of_queries_in_set:
                validation_queries.add(queries[index])
            query = queries[index]
            if query in validation_queries:
                validation_set.add(index)
        already_been_in_validation_indices = already_been_in_validation_indices.union(validation_set)
        train_set = train_indices - validation_set
        return already_been_in_validation_indices, list(validation_set), list(train_set)