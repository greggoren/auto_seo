from scipy.stats import kendalltau
from Preprocess.preprocess import retrieve_ranked_lists
from SentenceRanking.rbo import rbo
ranks_tf_idf_file = ''
ranks_vec = ''

ranked_lists_tfidf = retrieve_ranked_lists(ranks_tf_idf_file)
ranked_lists_vec = retrieve_ranked_lists(ranks_tf_idf_file)


sum_kt=0

rbo_stats={p:0 for p in [0.5,0.6,0,7,0.8,0.9,0.95]}
for query in ranked_lists_tfidf:
    list_tf_idf = ranked_lists_tfidf[query]
    list_vec = ranked_lists_vec[query]
    sum_kt +=kendalltau(list_tf_idf,list_vec)
    for p in rbo_stats:
        rbo_stats[p] += rbo(list_vec,list_tf_idf,p)


average_kt = round(sum_kt/len(ranked_lists_tfidf),3)

f = open("similarity_stats.tex","w")


