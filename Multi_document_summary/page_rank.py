from itertools import combinations
from utils import cosine_similarity
import numpy as np



#sentences numbers of sentences - need index: [sentence number+doc_name -> tf_idf]
def create_transition_graph(sentences):
    """
    M[i,j] = cos(i,j)/sum_of_all_edges(i)
    :param sentences:
    :return:Matrix M for transition edges
    """
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
            if denominators[node]!=0:
                M[node][neighbor]/=denominators[node]
            else:
                M[node][neighbor]=1/len(M[node])
    return M


def scores_init(transition_matrix):
    scores={}
    for node in transition_matrix:
        scores[node] = 1
    return scores

def calculate_inner_scores(node,transition_matrix,scores):
    sum = 0
    for inner_node in transition_matrix:
        if inner_node==node:
            continue
        sum+=transition_matrix[inner_node][node]*scores[inner_node]
    return sum


def page_rank(alpha,transition_matrix):
    nodes_number = len(transition_matrix)
    scores = scores_init(transition_matrix)
    stop = False
    while(not stop):
        stop=True
        for node in transition_matrix:
            new_score= alpha*calculate_inner_scores(node,transition_matrix,scores) + (1-alpha)/nodes_number
            if abs(new_score-scores[node])>0.0001:
                stop=False
            scores[node]=new_score
    return scores