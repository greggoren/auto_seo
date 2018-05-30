
def diversify(scores,transition_matrix,k):
    if k<=0:
        raise Exception("k <= 0")
    open = set(transition_matrix.keys())
    closed=set()
    while len(closed) < k or not open:
        sorted_scores=sorted(list(open),key=lambda x:scores[x],reverse=True)
        chosen_sentence = sorted_scores[0]
        closed.add(chosen_sentence)
        open.discard(chosen_sentence)
        for j in open:
            scores[j]=scores[j] -transition_matrix[j][chosen_sentence]*scores[chosen_sentence]
    return closed




