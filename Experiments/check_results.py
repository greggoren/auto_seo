import pickle
import params
from Experiments.stats import average_rank_addition,create_histogram
from Preprocess.preprocess import retrieve_ranked_lists,load_file


def create_lists(scores):
    lists = {}
    for doc in scores:
        query = doc.split("-")[2]
        if not lists.get(query,False):
            lists[query]={}
        lists[query][doc]=scores[doc]
    results = {}
    for query in lists:
        results[query]=sorted(list(lists[query].keys()),key=lambda x:lists[query][x],reverse=True)
    return results





ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
runs_pagerank = ["1_00","1_01","1_02","1_03","1_04","1_05","1_06","1_07","1_08","1_09","1_10"]
runs_weaving = ["00","01","02","030000000000000004","04","05","06","07"]
reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
for run in runs_pagerank:
    f = open("scores_of_model_"+run, "rb")
    scores = pickle.load(f)
    new_lists = create_lists(scores)
    average_rank_addition_value,meadian_rank_addition_value = average_rank_addition(ranked_lists,new_lists,reference_docs)
    create_histogram([new_lists[q].index(reference_docs[q]) + 1 for q in new_lists], "Rank", "#docs", "hist_"+run)

# print(average_rank_addition_value,meadian_rank_addition_value)




# doc_texts = load_file(params.new_trec_text_file)





