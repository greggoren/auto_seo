from scipy.stats import kendalltau,spearmanr,pearsonr
from Preprocess.preprocess import retrieve_ranked_lists

import itertools


def determine_order(pair, ranked_list):
    tmp = list(pair)
    return sorted(tmp, key=lambda x: ranked_list.index(x))


def kendall_distance(ranked1, ranked2):
    discordant = 0
    all_pairs = list(itertools.combinations(ranked1, 2))
    for pair in all_pairs:
        winner1, loser1 = determine_order(pair, ranked1)
        winner2, loser2 = determine_order(pair, ranked2)
        if winner1 != winner2:
            discordant += 1
    return float(discordant) / len(all_pairs)


ranks_tf_idf_file = 'scores_tfidf_past'
ranks_vec_file = 'scores_vec_past'

ranked_lists_tfidf = retrieve_ranked_lists(ranks_tf_idf_file)
ranked_lists_vec = retrieve_ranked_lists(ranks_vec_file)


sum_kt=0
sum_kt_dist=0
sum_spearman=0
for query in ranked_lists_tfidf:
    list_tf_idf = ranked_lists_tfidf[query]
    list_vec = ranked_lists_vec[query]
    kt = kendalltau(list_tf_idf,list_vec)
    sp = spearmanr(list_tf_idf,list_vec)
    kt_dist = kendall_distance(list_tf_idf,list_vec)
    sum_kt +=kt[0]
    sum_kt_dist+=kt_dist
    sum_spearman+=sp[0]


average_kt = round(sum_kt/len(ranked_lists_tfidf),3)
average_sp = round(sum_spearman/len(ranked_lists_tfidf),3)
average_kt_dist = round(sum_kt_dist/len(ranked_lists_tfidf),3)

f = open("similarity_stats.tex","w")
f.write("\\begin{tabular}{|c|c|c|}\n")
f.write("\\hline\n")
header = "Kendall-$\\tau$ & Kendall-distance & Spearman  \\\\ \n"

f.write(header)
f.write("\\hline\n")
f.write("w2v vs. tf-idf no past winners & "+str(average_kt)+" & "+str(average_kt_dist)+" & "+str(average_sp)+ " \\\\ \n")
f.write("\\hline\n")
f.write("\\end{tabular}")
