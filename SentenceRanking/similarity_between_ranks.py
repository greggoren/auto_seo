from scipy.stats import kendalltau
from Preprocess.preprocess import retrieve_ranked_lists
from SentenceRanking.rbo import rbo
import numpy as np
ranks_tf_idf_file = 'scores_TFIDF'
ranks_vec_file = 'scores_vec'

ranked_lists_tfidf = retrieve_ranked_lists(ranks_tf_idf_file)
ranked_lists_vec = retrieve_ranked_lists(ranks_vec_file)


sum_kt=0

rbo_stats={p:0 for p in [0.5,0.6,0.7,0.8,0.9,0.95]}
for query in ranked_lists_tfidf:
    list_tf_idf = ranked_lists_tfidf[query]
    list_vec = ranked_lists_vec[query]
    kt = kendalltau(list_tf_idf,list_vec)
    print(kt)
    if np.isnan(kt[0]):
        print("problem")
    sum_kt +=kt[0]

    for p in rbo_stats:
        rbo_stats[p] += rbo(list_vec,list_tf_idf,p)["ext"]


average_kt = round(sum_kt/len(ranked_lists_tfidf),3)

f = open("similarity_stats.tex","w")
f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
rbo_keys = sorted(list(rbo_stats.keys()))
header = "Comparison & Kendall-$\\tau$ & "+" & ".join([str(k) for k in rbo_keys])+" \\\\ \n"

f.write(header)
f.write("\\hline\n")
f.write("w2v vs. tf-idf no past winners & "+str(average_kt)+" & "+" & ".join([str(round(rbo_stats[p]/len(ranked_lists_tfidf),3)) for p in rbo_keys])+ "\\\\ \n")
f.write("\\hline\n")
f.write("\\end{tabular}")
