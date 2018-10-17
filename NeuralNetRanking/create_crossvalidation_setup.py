from w2v.train_word2vec import WordToVec
from krovetzstemmer import Stemmer
import numpy as np
from itertools import combinations
from random import shuffle,seed
import torch
import pickle
import os
import math

from shutil import copyfile


def get_sentence_vector(sentence,model):
    stemmer = Stemmer()
    sentence = clean_text(sentence)
    words = sentence.split()
    stemmed = [stemmer.stem(w) for w in words]
    return get_stemmed_document_vector(stemmed,model)


def clean_text(text):
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = text.replace("?", " ")
    text = text.replace("$", " ")
    text = text.replace("%", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("\\", " ")
    text = text.replace("*", " ")
    text = text.replace(";", " ")
    text = text.replace("`", "")
    text = text.replace("'", "")
    text = text.replace("@", " ")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    text = text.replace("/", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    return text


def get_stemmed_document_vector(doc,model):
    vector = np.zeros(300)
    i = 1
    for stem in doc:
        if stem in model.wv:
            vector += model.wv[stem]
        i += 1
    return vector / i


def fill_queries(query_file):
    result = {}
    with open(query_file) as queries:
        for query_data in queries:
            result[query_data.split(":")[0]] = query_data.split(":")[0].rstrip()
    return result





def read_file(data_set_file,queries):
    scores = {}
    raw_data ={}
    with open(data_set_file) as data_set_rows:
        for i, row in enumerate(data_set_rows):
            query, combination_name, doc1, doc2, score = row.split("!@@@!")[0], row.split("!@@@!")[1], \
                                                         row.split("!@@@!")[2], row.split("!@@@!")[3], \
                                                         row.split("!@@@!")[4]
            raw_data[combination_name] = (doc1, doc2, queries[query])
            if query not in scores:
                scores[query] = {}
            scores[query][combination_name] = score
    return scores,raw_data


def create_combinations(scores,raw_data):
    seed(9001)
    index = 0
    label = {}
    query_indexes ={}
    combination_index = {}
    combinations_obj = {}
    for query in scores:
        if query not in query_indexes:
            query_indexes[query] = []
            combination_index[query] =[]
        combination_names = list(scores[query].keys())
        for pair in combinations(combination_names, 2):
            shuffle(list(pair))
            comb1, comb2 = pair
            if comb1 not in combination_index[query]:
                combination_index[query].append(comb1)
            if comb2 not in combination_index[query]:
                combination_index[query].append(comb2)
            if scores[query][comb1] == scores[query][comb2]:
                continue
            combinations_obj[index] = (
                raw_data[comb1][0], raw_data[comb1][1], raw_data[comb1][2], raw_data[comb2][0],
                raw_data[comb2][1])
            if scores[query][comb1] > scores[query][comb2]:
                label[index] = torch.DoubleTensor([1]).cuda()

            else:
                label[index] = torch.DoubleTensor([-1]).cuda()
            query_indexes[query].append(index)
            index += 1
    return combinations_obj, label,query_indexes,combination_index





def create_crossvalidation_folds(query_indexes,number_of_folds):
    queries = list(query_indexes.keys())
    shuffle(queries)
    number_of_queries_per_fold = math.ceil(len(queries)/number_of_folds)
    print("there are ",number_of_queries_per_fold,"queries per fold")
    folds = {i+1:[] for i in range(number_of_folds)}
    i=1
    for query in queries:
        if len(folds[i])>=number_of_queries_per_fold:
            i+=1
        folds[i].append(query)
    return folds


def determine_folds(fold,folds):
    if fold != len(folds):
        validation_set = fold + 1
    else:
        validation_set = 1
    training_set = [i for i in folds if i!=fold and i!=validation_set]
    return training_set,validation_set



def save_single_object(file_name, obj):
    with open(file_name,"wb") as file:
        pickle.dump(obj,file)




def create_inference_folds(output_dir, raw_data, query, combination_index, model, running_index,test_names):
    combinations = combination_index[query]
    for combination in combinations:
        test_names[running_index]=combination
        data = raw_data[combination]
        parts = (data[0],data[1],data[2])
        vectors = [torch.from_numpy(get_sentence_vector(s, model)).cuda() for s in parts]
        extension = [torch.from_numpy(np.zeros(300)).cuda() for i in range(len(vectors)-1)]
        vectors.extend(extension)
        tensors = [torch.cat((vectors[0], vectors[1], vectors[2]), 0),
                   torch.cat((vectors[3], vectors[4], vectors[2]), 0)]
        save_single_object(output_dir + str(running_index),tensors)
        running_index+=1
    return running_index





def create_crossvalidation_folders(folds,query_indexes,input_dir,model,raw_data,combination_index,labels):
    folds_dir = "folds/"
    label_file_prefix = "labels_fold_"
    test_names={"test":{},"val":{}}
    for fold in folds:
        test_names["test"][fold]={}
        test_names["val"][fold]={}
        fold_labels={}
        if not os.path.exists(folds_dir+str(fold)):
            os.makedirs(folds_dir+str(fold))
        training_set, validation_set=determine_folds(fold,folds)
        current_train_folder = folds_dir+str(fold)+"/train/"
        if not os.path.exists(current_train_folder):
            os.makedirs(current_train_folder)
        running_index = 0
        for train_fold in training_set:
            queries = folds[train_fold]
            for query in queries:
                indexes = query_indexes[query]
                for index in indexes:
                    orig = input_dir+str(index)
                    dest = current_train_folder+str(running_index)
                    copyfile(orig,dest)
                    fold_labels[running_index] = labels[index]
                    running_index+=1

        save_single_object(label_file_prefix + str(fold) + ".pkl", fold_labels)
        current_validation_folder = folds_dir+str(fold)+"/validation/"
        if not os.path.exists(current_validation_folder):
            os.makedirs(current_validation_folder)
        validation_queries = folds[int(validation_set)]
        validation_index=0
        for query in validation_queries:
            validation_index=create_inference_folds(current_validation_folder,raw_data,query,combination_index,model,validation_index,test_names["val"][fold])
        current_test_folder = folds_dir + str(fold) + "/test/"
        if not os.path.exists(current_test_folder):
            os.makedirs(current_test_folder)
        test_queries = folds[fold]
        test_index = 0
        for query in test_queries:
           test_index = create_inference_folds(current_test_folder,raw_data,query,combination_index,model,test_index,test_names["test"][fold])

        save_single_object("test_names.pkl",test_names)




if __name__=="__main__":
    print("--== preprocess for NN ==--")
    print("loading w2v model")
    model = WordToVec().load_model()
    data_file = "/home/greg/auto_seo/NeuralNetRanking/new_sentences_add_remove"
    queries_file = "/home/greg/auto_seo/data/queris.txt"
    print("loading query data")
    queries = fill_queries(queries_file)
    print("loading data set file")
    scores, raw_data = read_file(data_file,queries)
    print("creating combinations")
    combinations_obj, labels ,query_indexes,combination_index= create_combinations(scores, raw_data)
    with open("comb_index.pkl","wb") as comb_index_file:
        pickle.dump(combination_index,comb_index_file)
    print("determining folds")
    folds =create_crossvalidation_folds(query_indexes, 5)
    print("creating folders")
    create_crossvalidation_folders(folds, query_indexes, "input_gpu/", model, raw_data, combination_index,labels)
    print("DONE!!")

