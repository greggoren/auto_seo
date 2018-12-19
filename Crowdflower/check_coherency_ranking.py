import csv
import numpy

from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import params
from w2v.train_word2vec import WordToVec
from SentenceRanking.sentence_features_experiment import get_sentence_vector,cosine_similarity
from Crowdflower import create_full_ds_per_task as mturk_ds_creator
from CrossValidationUtils.rankSVM_crossvalidation import cross_validation
from CrossValidationUtils.random_baseline import run_random
def create_sentence_similarities_ds(stats):
    f = open("coherency_features","w")
    qrels = open("coherency_labels","w")
    rows={}
    model = WordToVec().load_model()
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q: ranked_lists[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists}
    winner_docs = {q: ranked_lists[q][:3] for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts = {}
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
                window = []
                if i == 0:
                    window.append(get_sentence_vector(ref_sentences[1],model))
                    window.append(get_sentence_vector(ref_sentences[1],model))

                elif i+1 == len(ref_sentences):
                    window.append(get_sentence_vector(ref_sentences[i-1],model))
                    window.append(get_sentence_vector(ref_sentences[i-1],model))
                else:
                    window.append(get_sentence_vector(ref_sentences[i - 1], model))
                    window.append(get_sentence_vector(ref_sentences[i+1],model))
                ref_vector = get_sentence_vector(ref_sentence,model)
                similarity_to_ref_sentence = cosine_similarity(ref_vector,sentence_vec)
                query = run_name.split("-")[2]
                line = str(stats[run_name])+" qid:"+query

                # row["similarity_to_prev"]=cosine_similarity(sentence_vec,window[0])
                # row["similarity_to_ref_sentence"] = similarity_to_ref_sentence
                # row["similarity_to_pred"] = cosine_similarity(sentence_vec,window[1])
                # row["similarity_to_prev_ref"] = cosine_similarity(ref_vector,window[0])
                # row["similarity_to_pred_ref"] = cosine_similarity(ref_vector,window[1])
                row[1]=cosine_similarity(sentence_vec,window[0])
                row[2] = similarity_to_ref_sentence
                row[3] = cosine_similarity(sentence_vec,window[1])
                row[4] = cosine_similarity(ref_vector,window[0])
                row[5] = cosine_similarity(ref_vector,window[1])
                for j in range(1,len(row)+1):
                    line+=" "+str(j)+":"+str(row[j])
                line+=" # "+run_name+"\n"
                f.write(line)
                qrels.write(query+" 0 "+run_name+" "+str(stats[run_name])+"\n")
    f.close()
    qrels.close()
    return "coherency_features","coherency_labels"



ident_filename_fe = "figure-eight/ident_current.csv"
ident_filename_mturk = "Mturk/Manipulated_Document_Identification.csv"
ident_fe=mturk_ds_creator.read_ds_fe(ident_filename_fe,True)
ident_mturk = mturk_ds_creator.read_ds_mturk(ident_filename_mturk,True)


ident_results = mturk_ds_creator.combine_results(ident_fe,ident_mturk)

sentence_filename_fe = "figure-eight/sentence_current.csv"
sentence_filename_mturk = "Mturk/Sentence_Identification.csv"
sentence_filename_mturk_new = "Mturk/Sentence_Identification11.csv"
sentence_fe=mturk_ds_creator.read_ds_fe(sentence_filename_fe)
sentence_mturk = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk)
sentence_mturk_new = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk_new)
sentence_mturk=mturk_ds_creator.update_dict(sentence_mturk,sentence_mturk_new)


sentence_results = mturk_ds_creator.combine_results(sentence_fe,sentence_mturk)


for i in range(2,5):
    ident_annotation,ident_ratio = mturk_ds_creator.create_annotations(ident_results,i)
    sentence_annotation,sentence_ratio = mturk_ds_creator.create_annotations(sentence_results,i)


    results = mturk_ds_creator.keepagreement(ident_annotation,sentence_annotation)
    print("We are left with",len(results),"out of",len(sentence_annotation))
    new_features,qrels = create_sentence_similarities_ds(results)
    cross_validation(new_features,qrels,"summary_labels_"+str(i)+".tex","svm_rank",["map","ndcg","P.2","P.5"],"")
    run_random(new_features,qrels,str(i))
