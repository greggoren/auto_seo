import pickle
import numpy as np
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
        results[query]=sorted(sorted(list(lists[query].keys()),key=lambda x:x),key=lambda x:lists[query][x],reverse=True)
    return results

def create_table(res_mean,res_median,experiment):
    f = open(experiment+".csv",'w')
    for key in res_mean:
        new_key = key.replace("0","0.")
        line = ",".join((new_key,str(res_mean[key]),str(res_median[key])))
        f.write(line+"\n")
    f.close()

# f= open("new_texts_03",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
# print("")
#
# f= open("new_texts_07",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
# print("")
# f= open("summaries_1_03",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
# print("")
# f= open("summaries_1_05",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
# print("")
# f= open("summaries_1_08",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
ranked_lists = retrieve_ranked_lists("trec_file")
runs_pagerank = ["1_00","1_01","1_02","1_03","1_04","1_05","1_06","1_07","1_08","1_09","1_10"]
runs_weaving = ["00","01","02","03","04","05","06","07","08","09","10"]
reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
results_mean={}
results_median={}
for run in runs_pagerank:
    f = open("new_scores/scores_of_model_"+run, "rb")
    scores = pickle.load(f)
    new_lists = create_lists(scores)
    average_rank_addition_value,meadian_rank_addition_value = average_rank_addition(ranked_lists,new_lists,reference_docs)
    results_mean[run]=average_rank_addition_value
    results_median[run]=meadian_rank_addition_value
    a = [new_lists[q].index(reference_docs[q]) + 1 for q in new_lists]
    create_histogram(a, "Rank", "#docs", "pagerank_hist_"+run)
create_table(results_mean,results_median,"pagerank")
results_mean={}
results_median={}
for run in runs_weaving:
    f = open("new_scores/scores_of_model_"+run, "rb")
    scores = pickle.load(f)
    new_lists = create_lists(scores)
    average_rank_addition_value,meadian_rank_addition_value = average_rank_addition(ranked_lists,new_lists,reference_docs)
    results_mean[run]=average_rank_addition_value
    results_median[run]=meadian_rank_addition_value
    create_histogram(np.array([new_lists[q].index(reference_docs[q]) + 1 for q in new_lists]), "Rank", "#docs", "weaving_hist_"+run)
create_table(results_mean,results_median,"weaving")

# f= open("summaries_1_05",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
# print("")
# f= open("summaries_1_08",'rb')
# s = pickle.load(f)
# f.close()
# print(s["ROUND-01-195-51"])
#
#
# f = open("scores/scores_of_model_1_05", "rb")
#
# scores = pickle.load(f)
# f.close()
# new_lists = create_lists(scores)
# print(new_lists["195"])
# print(scores["ROUND-01-195-13"])
# print(scores["ROUND-01-195-51"])
# f = open("scores/scores_of_model_1_08", "rb")
# print("")
# scores = pickle.load(f)
# f.close()
# print(scores["ROUND-01-195-13"])
# print(scores["ROUND-01-195-51"])
# new_lists = create_lists(scores)
# print(new_lists["195"])