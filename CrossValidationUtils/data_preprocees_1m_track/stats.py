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




qrels = "../qrels"
prels = "../mq_track_qrels"

stats = read_qrels_stats(prels)
args = analyze(stats)
print("added diff level=",args[0])
print("avg added diff level=",args[1])
print("added bin level=",args[2])
print("avg added bin level=",args[3])

lm_score1 ="../svm_scores_regular"
lm_score2 ="../svm_scores_extended"
svm_score1="../lm_scores"
svm_score2="../lm_scores_extended"

print("significance for svm change")
analyze_significance(qrels,svm_score1,svm_score2)

print("significance for lm change")
analyze_significance(qrels,lm_score1,lm_score2)