from Preprocess.preprocess import retrieve_ranked_lists
import params


def read_labels(filename):
    result ={}
    with open(filename) as file:
        for line in file:
            query = line.split()[0]
            if query not in result:
                result[query]={}
            doc = line.split(" ")[2]
            index = line.split(" ")[3]
            result[query][doc]=int(index)
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


new_ranked_list ="trec_file04"
ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
ranked_lists_new = retrieve_ranked_lists(new_ranked_list)
reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
new_indexes = read_labels("labels_new")

with open("labels_new_final","w") as labels:
    for query in new_indexes:
        for doc in new_indexes[query]:
            new_index = new_indexes[query][doc]
            old_index = ranked_lists_new[query].index(doc)
            new_label = str(define_new_label(new_index,old_index,len(ranked_lists_new[query])))
            labels.write(query+" 1 "+doc+" "+new_label+"\n")



