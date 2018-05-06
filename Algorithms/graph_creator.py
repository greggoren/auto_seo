import pyndri
import itertools
def create_graph_from_sentences(sentences,path_to_index):
    index = pyndri.Index(path_to_index)
    token2id, id2token, id2df = index.get_dictionary()
    
    ##print(len(token2id))

