from torch.utils.data import Dataset
from w2v.train_word2vec import WordToVec
from krovetzstemmer import Stemmer
import numpy as np
from itertools import combinations
from random import shuffle,seed
import torch
class PairWiseDataLoaer(Dataset):
    def __init__(self,data_file,queries_file):
        self.model = WordToVec().load_model()
        self.queries = {}
        print("filling queries")
        self.fill_queries(queries_file)
        self.raw_data = {}
        self.scores = {}
        print("reading data")
        self.read_file(data_file)
        self.combinations = {}
        print("creating combinations")
        self.label = {}
        self.create_combinations()


    def get_sentence_vector(self,sentence):
        stemmer = Stemmer()
        sentence = self.clean_text(sentence)
        words = sentence.split()
        stemmed = [stemmer.stem(w) for w in words]
        return self.get_stemmed_document_vector(stemmed)


    def clean_text(self,text):
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

    def get_stemmed_document_vector(self,doc):
        vector = np.zeros(300)
        i = 1
        for stem in doc:
            if stem in self.model.wv:
                vector += self.model.wv[stem]
            i += 1
        return vector / i


    def fill_queries(self,query_file):
        with open(query_file) as queries:
            for query_data in queries:
                self.queries[query_data.split(":")[0]]=query_data.split(":")[0].rstrip()

    def read_file(self,data_set_file):
        with open(data_set_file) as data_set_rows:
            for i,row in enumerate(data_set_rows):
                query,combination_name,doc1,doc2,score = row.split("!@@@!")[0],row.split("!@@@!")[1],row.split("!@@@!")[2],row.split("!@@@!")[3],row.split("!@@@!")[4]
                self.raw_data[combination_name]=(doc1,doc2,self.queries[query])
                if query not in self.scores:
                    self.scores[query] ={}
                self.scores[query][combination_name]=score



    def create_combinations(self):
        seed(9001)
        index = 0
        for query in self.scores:
            combination_names = list(self.scores[query].keys())
            for pair in combinations(combination_names,2):
                shuffle(list(pair))
                comb1,comb2 = pair
                if self.scores[query][comb1]==self.scores[query][comb2]:
                    continue
                self.combinations[index] = (
                self.raw_data[comb1][0], self.raw_data[comb1][1], self.raw_data[comb1][2], self.raw_data[comb2][0],
                self.raw_data[comb2][1])
                if self.scores[query][comb1]>self.scores[query][comb2]:
                    self.label[index]=1

                else:
                    self.label[index]=-1
                index += 1


    def __len__(self):
        return len(self.combinations)

    def __getitem__(self, idx):
        combination = self.combinations[idx]
        vectors = [torch.from_numpy(self.get_sentence_vector(s),) for s in combination]
        label = self.label[idx]
        return vectors,label