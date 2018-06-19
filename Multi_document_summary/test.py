import pickle as pc
from utils import cosine_similarity
from Preprocess import preprocess
import pyndri
import params

sentences = ["A toilet can be designed for people who prefer to sit (by using a toilet pedestal) or for people who prefer to squat and use a squat toilet.","A toilet is designed for people who prefer to sit (by using a toilet pedestal) or for people who prefer to use a squat toilet."]
vectors=[]
index = pyndri.Index(params.path_to_index)
token2id, id2token, id2df = index.get_dictionary()
del id2token
print(pyndri.escape("#1"))
pyndri.tokenize("1")
