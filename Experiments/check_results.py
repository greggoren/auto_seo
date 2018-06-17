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



f = open("scores_of_model","rb")

scores = pickle.load(f)

ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}

new_lists = create_lists(scores)

average_rank_addition_value,meadian_rank_addition_value = average_rank_addition(ranked_lists,new_lists,reference_docs)

print(average_rank_addition_value,meadian_rank_addition_value)

create_histogram([new_lists[q].index(reference_docs[q])+1 for q in new_lists],"Rank","#docs","hist")

reference_doc_names_for_example =  list(reference_docs.values())[:2]

# doc_texts = load_file(params.new_trec_text_file)

summary_file = open("summaries","rb")
summaries=pickle.load(summary_file)


print(summaries["004"])




