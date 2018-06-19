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
for sentence in sentences:
    vectors.append(preprocess.convert_sentence_to_tfidf_vector(sentence,token2id=token2id,id2df=id2df,index=index))
#f= open("/home/student/Desktop/senteces_1_0.5",'rb')
# texts,vectors = pc.load(f)
# f.close()
print(cosine_similarity(vectors[1],vectors[0]))
