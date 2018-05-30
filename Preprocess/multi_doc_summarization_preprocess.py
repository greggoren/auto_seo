from itertools import combinations
from utils import cosine_similarity

#sentences numbers of sentences - need index: [sentence number+doc_name -> tf_idf]
def create_transition_graph(sentences):
    M={}
    denominators = {}
    keys = list(sentences.keys())
    for i,j in combinations(keys,2):
        if not M.get(i,False):
            M[i]={}
            denominators[i] = 0
        if not M.get(j,False):
            M[j]={}
            denominators[j] = 0
        similarity = cosine_similarity(sentences[i],sentences[j])
        M[i][j]=M[j][i]=similarity
        denominators[i]+=similarity
        denominators[j]+=similarity

    for node in M:
        for neighbor in M[node]:
            M[node][neighbor]/=denominators[node]
    return M




