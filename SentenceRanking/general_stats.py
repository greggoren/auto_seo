import numpy as np
def number_of_effective_sentences(stats):
    result = 0
    for query in stats:
        result+=len([x for x in stats[query] if x>0])
    return result

def number_of_queries_with_no_effective_sentence(stats):
    result = 0
    for query in stats:
        if len([x for x in stats[query] if x==0]) == len(stats[query]):
            result+=1
    return result

def number_of_queries_with_full_effective_list(stats):
    result = 0
    for query in stats:
        if len([x for x in stats[query] if x > 0]) == len(stats[query]):
            result += 1
    return result

def average_label(stats):
    result = 0
    for query in stats:
        result += np.mean(stats[query])
    return result/len(stats)

def analayze_labels(labels_file):
    stats = {}
    f = open(labels_file)
    sentences = 0
    for line in f:
        sentences+=1
        splitted = line.split()
        label = splitted[3]
        query = splitted[0]
        if query not in stats:
            stats[query]=[]
        stats[query].append(int(label))
    f.close()

    s = open("analysis_regular.tex","w")
    s.write("\\begin{tabular}{|c|c|c|c|c|}\n")
    s.write("\\hline\n")
    s.write("Number Of sentences & Effective sentences & Q-N-S & Q-F-S & Average label \\\\ \n")
    s.write("\\hline\n")
    res = [str(sentences),str(number_of_effective_sentences(stats)),str(number_of_queries_with_no_effective_sentence(stats)),str(number_of_queries_with_full_effective_list(stats)),str(average_label(stats))]
    s.write(" & ".join(res)+"\\\\ \n")
    s.write("\\hline\n")
    s.write("\\end{tabular}")
    s.close()

analayze_labels("labels_regular")