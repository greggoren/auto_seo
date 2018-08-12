from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext
from Experiments.experiment_data_processor import delete_doc_from_index
from Experiments.experiment_data_processor import add_docs_to_index
from Experiments.experiment_data_processor import merge_indices
from Experiments.experiment_data_processor import create_index
from Experiments.experiment_data_processor import create_features_file
from Experiments.model_handler import run_model
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
from SentenceRanking.sentence_parse import map_sentences
from SentenceRanking.sentence_parse import create_lists
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

def avoid_docs_for_working_set(reference_doc,reference_docs):
    diffenrece = set(reference_docs).difference(set([reference_doc]))
    return diffenrece


if __name__=="__main__":
    f = open("dic4.pickle", "rb")
    dic = pickle.load(f)
    f.close()
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

    reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
    winner_docs = {q:ranked_lists[q][0] for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc]=a_doc_texts[doc]

    trec_text_file = create_trectext(doc_texts, [], "",reference_docs)
    added_index = create_index(trec_text_file)
    merged_index = merge_indices(added_index,"","/home/greg/baseindex")




