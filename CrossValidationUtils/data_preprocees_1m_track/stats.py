from itertools import combinations
qrels = "../data/qrels"


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



