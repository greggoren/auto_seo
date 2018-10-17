from w2v.train_word2vec import WordToVec
from krovetzstemmer import Stemmer
import numpy as np
from itertools import combinations
from random import shuffle,seed
import torch
import pickle
import os
from torch.autograd import Variable



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
    combinations_obj = {}
    for query in scores:
        combination_names = list(scores[query].keys())
        for pair in combinations(combination_names, 2):
            shuffle(list(pair))
            comb1, comb2 = pair
            if scores[query][comb1] == scores[query][comb2]:
                continue
            combinations_obj[index] = (
                raw_data[comb1][0], raw_data[comb1][1], raw_data[comb1][2], raw_data[comb2][0],
                raw_data[comb2][1])
            if scores[query][comb1] > scores[query][comb2]:
                label[index] = torch.DoubleTensor([1]).cuda()

            else:
                label[index] = torch.DoubleTensor([-1]).cuda()
            index += 1
    return combinations_obj, label




def save_data(combinations,labels,model):
    labels_dir = "labels/"
    if not os.path.exists(labels_dir):
        os.makedirs(labels_dir)
    with open(labels_dir+"labels.pkl","wb") as labels_data:
        pickle.dump(labels,labels_data)
    data_dir = "input_gpu/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    for idx in combinations:
        combination = combinations[idx]
        vectors = [torch.from_numpy(get_sentence_vector(s,model)).cuda() for s in combination]
        tensors = [torch.cat((vectors[0],vectors[1],vectors[2]),1),torch.cat((vectors[3],vectors[4],vectors[2]),1)]
        with open(data_dir+str(idx),"wb") as input_point:
            pickle.dump(tensors,input_point)



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
    combinations_obj, labels = create_combinations(scores, raw_data)
    print("saving pickle files")
    save_data(combinations_obj, labels, model)
    print("DONE!")

