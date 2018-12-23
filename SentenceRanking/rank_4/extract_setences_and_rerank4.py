from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext
from Experiments.experiment_data_processor import create_features_file
from Experiments.model_handler import run_model
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
from SentenceRanking.sentence_parse import map_sentences, map_set_of_sentences
from SentenceRanking.sentence_parse import create_lists
from Preprocess.preprocess import retrieve_sentences
import params
import sys
import time
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
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q:ranked_lists[q][3].replace("EPOCH","ROUND") for q in ranked_lists}
    winner_docs = {q:ranked_lists[q][:3] for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc]=a_doc_texts[doc]
    sentence_map=map_set_of_sentences(doc_texts,winner_docs)
    summaries = {}
    labels_file=open("labels_4", 'w')
    sentence_data_file = open("sentences_add_remove_4", "w")
    index=1
    for query in sentence_map:
        print("in query",index, "out of",len(sentence_map))
        sys.stdout.flush()
        reference_doc = reference_docs[query].replace("EPOCH","ROUND")
        reference_text = doc_texts[reference_doc]
        reference_sentences = retrieve_sentences(reference_text)
        for sentence in sentence_map[query]:
            r_index = 1
            new_sentence = sentence_map[query][sentence].replace("\n", "")
            if not new_sentence:
                continue
            for reference_sentence in reference_sentences:
                run_name = sentence+"_"+str(r_index)
                reference_sentence=reference_sentence.replace("\n", "")
                if not reference_sentence:
                    continue
                modified_doc=reference_doc+"\n"+new_sentence
                summaries[reference_doc]=modified_doc
                add = open("/home/greg/auto_seo/scripts/add_remove_4",'w',encoding="utf8")
                add.write(reference_doc+"@@@"+new_sentence.rstrip()+"@@@"+reference_sentence.rstrip()+"\n")
                sentence_data_file.write(run_name + "@@@" + new_sentence.rstrip() + "@@@" + reference_sentence.rstrip() + "\n")
                add.close()
                time.sleep(1)
                trec_text_file = create_trectext(doc_texts, summaries, "",[])
                features_dir = "Features_4"
                feature_file = "features_4_"+run_name
                create_features_file(features_dir, params.path_to_index, params.queries_xml,feature_file,"")
                index_doc_name = create_index_to_doc_name_dict(feature_file)
                scores_file = run_model(feature_file)
                results = retrieve_scores(index_doc_name, scores_file)
                lists=create_lists(results)
                addition = abs(lists[query].index(reference_doc) - len(lists[query]))
                query = sentence.split("-")[2]
                labels_file.write(query + " 1 " + run_name + " " + str(addition - 1)+" seo" + "\n")
                r_index+=1
        index+=1
    labels_file.close()
    sentence_data_file.close()



