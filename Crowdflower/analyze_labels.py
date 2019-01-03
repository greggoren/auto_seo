from Preprocess.preprocess import retrieve_ranked_lists
import params
import numpy as np

def read_labels(filename):
    result ={}
    with open(filename) as file:
        for line in file:
            query = line.split()[0]
            beta = line.split()[2]
            if beta not in result:
                result[beta]={}

            index = line.split(" ")[3]
            result[beta][query] = int(index.rstrip())

    return result

def read_labels_demotion(filename):
    result ={}
    with open(filename) as file:
        for line in file:
            query = line.split()[0]
            beta = "-"
            if beta not in result:
                result[beta]={}

            index = line.split(" ")[2]
            result[beta][query] = int(index.rstrip())

    return result



def define_new_label(new_index,old_index,number_of_docs):
    max_rank = number_of_docs-1
    if new_index>=old_index:
        label=0
    else:
        label = max_rank-new_index
    return label

def determine_indexes(doc,ranked_list):
    return min(ranked_list.index(doc),3)



def get_true_labels(new_indexes,lists,reference_docs):
    result = {}
    for beta in new_indexes:
        result[beta]={}
        for query in new_indexes[beta]:
            r_doc = reference_docs[query]
            old_index = lists[query].index(r_doc)
            result[beta][query]= define_new_label(new_indexes[beta][query],old_index,len(lists[query]))
    return result

def write_table(method,results):
    f = open("summary_two_sentences_"+method+".tex","w")
    f.write("\\begin{tabular}{|c|c|\n")
    f.write("\\hline\n")
    f.write("$\\beta$ & Average Addition \\\\ \n")
    f.write("\\hline\n")
    for beta in results:
        average = str(round(np.mean([results[beta][q] for q in results[beta]]),3))
        f.write(beta+" & "+average+" \\\\ \n")
        f.write("\\hline\n")
    f.write("\\end{tabular}\n")

new_ranked_list ="ranked_lists/trec_file04"
ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
ranked_lists_new = retrieve_ranked_lists(new_ranked_list)
reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}

indexes = read_labels_demotion("labels_demotion")
labels = get_true_labels(indexes,ranked_lists_new,reference_docs)
write_table("demotion",labels)

indexes = read_labels("labels_harmonic")
labels = get_true_labels(indexes,ranked_lists_new,reference_docs)
write_table("harmonic",labels)

indexes = read_labels("labels_weighted")
labels = get_true_labels(indexes,ranked_lists_new,reference_docs)
write_table("weighted",labels)