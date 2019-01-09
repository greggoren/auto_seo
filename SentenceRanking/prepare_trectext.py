from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext
from Experiments.experiment_data_processor import merge_indices
from Experiments.experiment_data_processor import create_index
import params
import sys
import pickle


def retrieve_query_names():
    query_mapper = {}
    with open(params.query_description_file,'r') as file:
        for line in file:
            data = line.split(":")
            query_mapper[data[0]]=data[1].rstrip()
    return query_mapper



if __name__=="__main__":
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
    print(reference_docs)
    winner_docs = {q:ranked_lists[q][0] for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04") or doc.__contains__("ROUND-06"):
            doc_texts[doc]=a_doc_texts[doc]
    trec_text_file = create_trectext(doc_texts, [],"trec_text_round_4_6","ws4_6")

    added_index = create_index(trec_text_file,"4_6")
    merged_index = merge_indices(added_index,"","/home/greg/mergedindex")




