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
    # f = open("dic4.pickle", "rb")
    # dic = pickle.load(f)
    # f.close()
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

    reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
    winner_docs = {q:ranked_lists[q][0].replace("EPOCH","ROUND") for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts={}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc]=a_doc_texts[doc]
    sentence_map=map_sentences(doc_texts,winner_docs)
    summaries = {}
    f=open("labels",'a')
    index=1
    for query in sentence_map:
        print("in query",index, "out of",len(sentence_map))
        sys.stdout.flush()
        reference_doc = reference_docs[query].replace("EPOCH","ROUND")
        reference_text = doc_texts[reference_doc]
        for sentence in sentence_map[query]:
            if sentence=="":
                continue
            run_name = sentence
            new_sentence = sentence_map[query][sentence]
            if new_sentence=="":
                continue
            modified_doc=reference_doc+"\n"+new_sentence
            summaries[reference_doc]=modified_doc
            add = open("/home/greg/auto_seo/scripts/add",'w',encoding="utf8")
            add.write(reference_doc+"@@@"+new_sentence.rstrip()+"\n")
            add.close()
            time.sleep(8)
            # avoid = avoid_docs_for_working_set(reference_doc, list(reference_docs.values()))
            trec_text_file = create_trectext(doc_texts, summaries, "",[])
            # added_index = create_index(trec_text_file,run_name)
            # merged_index=merge_indices(added_index,run_name)
            features_dir = "Features"
            feature_file = "features"+run_name
            create_features_file(features_dir, params.path_to_index, params.queries_xml,feature_file,run_name)
            index_doc_name = create_index_to_doc_name_dict(feature_file)
            scores_file = run_model(feature_file)
            results = retrieve_scores(index_doc_name, scores_file)
            lists=create_lists(results)
            s = open("lists_"+run_name,'wb')
            pickle.dump(lists,s)
            s.close()
            addition = abs(lists[query].index(reference_doc) - len(lists[query]))
            f.write(run_name+"\t"+str(addition)+"\n")
    f.close()



