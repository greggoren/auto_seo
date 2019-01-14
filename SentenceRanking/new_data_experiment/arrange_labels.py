from Preprocess.preprocess import retrieve_ranked_lists
import params
import sys

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




if __name__=="__main__":
    current_round =sys.argv[1]
    ref_index = sys.argv[2]
    addition = current_round.zfill(2)+"_"+ref_index
    new_ranked_list ="trec_file"+current_round.zfill(2)
    ranked_lists_new = retrieve_ranked_lists(new_ranked_list)
    reference_docs = {q:ranked_lists_new[q][int(ref_index)].replace("EPOCH","ROUND") for q in ranked_lists_new}
    new_indexes = read_labels("labels_new_"+addition)
    if ref_index=="-1":
        query_name_add =current_round+"5"
    else:
        query_name_add = current_round + "2"
    with open("labels_new_final_all_data","a") as labels:
        for query in new_indexes:
            for doc in new_indexes[query]:
                new_index = new_indexes[query][doc]
                ref_doc = reference_docs[query]
                old_index = ranked_lists_new[query].index(ref_doc)
                new_label = str(define_new_label(new_index,old_index,len(ranked_lists_new[query])))
                labels.write(query+query_name_add+" 1 "+doc+" "+new_label+"\n")



