import csv
import numpy
from docutils.nodes import header

from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from SentenceRanking.sentence_parse import map_sentences, map_set_of_sentences
import params
from w2v.train_word2vec import WordToVec
from SentenceRanking.sentence_features_experiment import get_sentence_vector,get_vectors,cosine_similarity
def get_total_coherence_level():
    stats={}
    with open("comb.csv",encoding="utf-8") as file:
        data = csv.DictReader(file,delimiter=",")
        for row in data:
            if row["_golden"]=="true" or row["_golden"]=="TRUE":
                continue
            id = row["id"]

            if id not in stats:
                stats[id]=[]

            value = 0
            if row["which_document_has_experienced_manipulation"]=="":
                continue
            if "document_" in  row["check_one_gold"]:
                ref = row["check_one_gold"].split("document_")[1]
            else:
                ref = row["check_one_gold"].split("Document")[1]
            if row["which_document_has_experienced_manipulation"].split("_")[1]!=ref:
                value=1
            stats[id].append(value)
    with open("ident.csv",encoding="utf-8") as file:
        data = csv.DictReader(file, delimiter=",")
        for row in data:
            if row["_golden"] == "true" or row["_golden"] == "TRUE":
                continue

            id = row["id"]

            if id not in stats:
                stats[id]=[]
            value = 0
            if row["which_document_has_experienced_manipulation"]!="":
                if row["which_document_has_experienced_manipulation"]!=row["check_one_gold"]:
                    value = 1
            elif row["which_sentence_doesnt_belong_to_original_document"] != \
                    row["check_one_gold"]:
                value = 1
            stats[id].append(value)
    return stats


def create_sentence_similarities(stats):
    rows={}
    model = WordToVec().load_model()
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q: ranked_lists[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists}
    winner_docs = {q: ranked_lists[q][:3] for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts = {}
    index =1
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc] = a_doc_texts[doc]
    sentence_map = map_set_of_sentences(doc_texts, winner_docs)
    for query in sentence_map:
        ref_doc = reference_docs[query]

        text = doc_texts[ref_doc]
        ref_sentences = retrieve_sentences(text)
        for sentence in sentence_map[query]:

            sentence_vec = get_sentence_vector(sentence_map[query][sentence],model=model)
            for i,ref_sentence in enumerate(ref_sentences):
                row = {}
                run_name = sentence+"_"+str(i+1)
                if run_name not in stats:
                    continue
                print("run name in stats")
                window = []
                if i == 0:
                    window.append(numpy.ones(300))
                    window.append(get_sentence_vector(ref_sentences[1],model))

                elif i+1 == len(ref_sentences):
                    window.append(get_sentence_vector(ref_sentences[i-1],model))
                    window.append(numpy.ones(300))
                else:
                    window.append(get_sentence_vector(ref_sentences[i - 1], model))
                    window.append(get_sentence_vector(ref_sentences[i+1],model))


                ref_vector = get_sentence_vector(ref_sentence,model)
                # window_dict = {}
                # window_dict[0]=window
                # window_centroid_dict,_ = get_vectors(window_dict)
                # window_centroid=window_centroid_dict[0]
                # similarity_to_window = cosine_similarity(window_centroid,sentence_vec)
                similarity_to_ref_sentence = cosine_similarity(ref_vector,sentence_vec)
                row["id"]=run_name
                row["similarity_to_prev"]=cosine_similarity(sentence_vec,window[0])
                row["similarity_to_ref_sentence"] = similarity_to_ref_sentence
                row["similarity_to_pred"] = cosine_similarity(sentence_vec,window[1])
                row["similarity_to_prev_ref"] = cosine_similarity(ref_vector,window[0])
                row["similarity_to_pred_ref"] = cosine_similarity(ref_vector,window[1])
                score = 0
                if numpy.mean(stats[run_name])>0.5:
                    score=1
                row["score"]=score
                # row["score"]=numpy.mean(stats[run_name])
                rows[index]=row
                index+=1
    return rows



stats = get_total_coherence_level()
rows = create_sentence_similarities(stats)
fieldnames = ["id","similarity_to_prev","similarity_to_pred","similarity_to_ref_sentence","similarity_to_prev_ref","similarity_to_pred_ref","score"]
with open("coherence.csv","w",newline='') as data_set:
    writer = csv.DictWriter(data_set,fieldnames=fieldnames)
    writer.writeheader()
    for i in rows:
        row = rows[i]
        writer.writerow(row)
