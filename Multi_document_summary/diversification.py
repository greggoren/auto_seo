from utils import cosine_similarity

def diversify(scores, transition_matrix, k, query_vector, sentence_vectors, original_doc_vector, gamma):
    new_scores = {}
    for sentence in scores:
        value = scores[sentence]
        new_scores[sentence]= (gamma * cosine_similarity(sentence_vectors[sentence], original_doc_vector) + (1 - gamma) * cosine_similarity(query_vector, sentence_vectors[sentence])) * value
    if k<=0:
        raise Exception("k <= 0")
    open = set(transition_matrix.keys())
    closed=set()
    while len(closed) < k or not open:
        sorted_scores=sorted(list(open),key=lambda x:new_scores[x],reverse=True)
        chosen_sentence = sorted_scores[0]
        closed.add(chosen_sentence)
        open.discard(chosen_sentence)
        for j in open:
            scores[j]=scores[j] -transition_matrix[j][chosen_sentence]*scores[chosen_sentence]
    return closed



