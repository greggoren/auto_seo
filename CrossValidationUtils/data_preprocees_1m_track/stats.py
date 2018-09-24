from itertools import combinations
from CrossValidationUtils.evaluator import eval
from scipy.stats import ttest_rel
qrels = "../data/qrels"



def analyze_significance(qrels,score_file1,score_file2):
    evaluator = eval()
    score_data1 = evaluator.run_trec_eval_by_query(qrels,score_file1)
    score_data2 = evaluator.run_trec_eval_by_query(qrels,score_file2)
    for metric in score_data1:
        x = score_data1[metric]
        y = score_data2[metric]
        ttest_val = ttest_rel(x,y)
        print("metric =",ttest_val)

def read_qrels_stats(qrels):
    rels = {}
    with open(qrels) as file:
        for line in file:
            query = int(line.split()[0])
            if query<=200:
                continue
            rel = int(line.split()[3])
            if query not in rels:
                rels[query]=[]
            rels[query].append(rel)
    return rels


def analyze(stats):
    total_added_pairs_different_level = 0
    total_added_pairs_different_binary = 0

    for query in stats:
        for i,j in combinations(stats[query],2):
            if i!=j:
                total_added_pairs_different_level+=1
                if i==0 or j==0:
                    total_added_pairs_different_binary+=1
    return total_added_pairs_different_level,total_added_pairs_different_level/len(stats),total_added_pairs_different_binary,total_added_pairs_different_binary/len((stats))



