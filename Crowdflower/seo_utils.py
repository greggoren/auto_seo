from utils import cosine_similarity

from SentenceRanking.sentence_features_experiment import get_sentence_vector
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import params
from w2v.train_word2vec import WordToVec
from Experiments.model_handler import retrieve_scores
from CrossValidationUtils.svm_handler import svm_handler



def save_max_mix_stats(stats,row,query):
    features = list(row.keys())
    if query not in stats:
        stats[query]={}
    for feature in features:
        if feature not in stats[query]:
            stats[query][feature]={}
            stats[query][feature]["max"]  = row[feature]
            stats[query][feature]["min"] = row[feature]
        if row[feature]>stats[query][feature]["max"]:
            stats[query][feature]["max"] = row[feature]
        if row[feature]<stats[query][feature]["min"]:
            stats[query][feature]["min"] = row[feature]
    return stats

def determine_indexes(doc,ranked_list):
    return min(ranked_list.index(doc),3)


def save_modified_file(doc_texts, combinations,chosen_comb, ref_doc):
    old_text = doc_texts[ref_doc]
    comb = combinations[chosen_comb]
    sentence_in = comb[0]
    sentence_out = comb[1].rstrip()
    new_text = old_text.replace(sentence_out,sentence_in)
    doc_texts[ref_doc]=new_text
    return doc_texts


def reduce_combinations():
    pass


def pick_best_sentences(score_file,base_sentences={}):
    stats={}
    if base_sentences:
        stats = base_sentences.copy()
    with open(score_file) as scores:
        for line in scores:
            query = line.split()[0]
            comb = line.split()[2]
            if query not in stats:
                stats[query]=[]
            if not stats[query]:
                stats[query].append(comb)
            elif base_sentences and len(stats[query])<2:
                replacement_index = comb.split("_")[-1]
                prefix = "_".join(comb.split("_")[:-1])
                flag = True
                for existing_comb in stats[query]:
                    e_replacement_index = existing_comb.split("_")[-1]
                    e_prefix ="_".join(existing_comb.split("_")[:-1])

                    if e_replacement_index==replacement_index or e_prefix==prefix:
                        flag=False
                if flag:
                    stats[query].append(comb)
    return stats

def create_doc_name_index(features_file):
    doc_name_index = {}
    with open(features_file) as features:
        for index,line in enumerate(features):
            doc_name = line.split(" # ")[1].rstrip()
            doc_name_index[index]=doc_name
    return doc_name_index



def retrieve_scores(score_file):
    with open(score_file) as scores:
        results = {i: score.rstrip() for i, score in enumerate(scores)}
        return results


def create_trec_eval_file(doc_name_index, results,method):

    trec_file = method + "_scores.txt"
    trec_file_access = open(trec_file, 'w')
    for index in results:
        doc_name = doc_name_index[index]
        query = doc_name.split("-")[2]
        trec_file_access.write(query + " Q0 " + doc_name + " " + str(0) + " " + str(results[index]) + " seo\n")
    trec_file_access.close()
    return trec_file




def read_chosen_model_file(filename):
    result={}
    with open(filename) as file:
        for line in file:
            method = line.split()[0]
            filename = line.split()[1].rstrip()
            C= filename.split("_")[3].replace(".txt","")
            result[method]=C
    return result

def run_chosen_model_for_stats(chosen_models, method,  feature_file, doc_name_index, base_features_file):
    chosen_model_parameter = chosen_models[method]
    svm = svm_handler()
    model_file = svm.learn_svm_rank_model(base_features_file, method, chosen_model_parameter)
    evaluator = eval(["map", "ndcg", "P.2", "P.5"])
    scores_file = svm.run_svm_rank_model(feature_file, model_file, method)

    results = retrieve_scores(scores_file)
    trec_file = create_trec_eval_file(doc_name_index, results, method)
    final_trec_file = evaluator.order_trec_file(trec_file)
    return final_trec_file



def create_coherency_features(ref_index=-1,ranked_list_new_file="",doc_text_modified=""):
    rows={}
    max_min_stats={}
    model = WordToVec().load_model()
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    ranked_lists_new = retrieve_ranked_lists(ranked_list_new_file)
    reference_docs = {q: ranked_lists[q][ref_index].replace("EPOCH", "ROUND") for q in ranked_lists}
    winner_docs = {q: ranked_lists_new[q][:determine_indexes(reference_docs[q],ranked_lists_new[q])] for q in ranked_lists_new}
    file_to_load = params.trec_text_file
    if doc_text_modified:
        a_doc_texts = doc_text_modified
    else:
        a_doc_texts = load_file(file_to_load)
    doc_texts = {}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc] = a_doc_texts[doc]
    sentence_map = map_set_of_sentences(doc_texts, winner_docs)
    for query in sentence_map:
        ref_doc = reference_docs[query]

        text = doc_texts[ref_doc]
        ref_sentences = retrieve_sentences(text)
        if len(ref_sentences)<2:
            continue
        for sentence in sentence_map[query]:

            sentence_vec = get_sentence_vector(sentence_map[query][sentence],model=model)
            for i,ref_sentence in enumerate(ref_sentences):
                row = {}
                run_name = sentence+"_"+str(i+1)
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
                query = run_name.split("-")[2]
                row["similarity_to_prev"]=cosine_similarity(sentence_vec,window[0])
                row["similarity_to_ref_sentence"] = cosine_similarity(ref_vector,sentence_vec)
                row["similarity_to_pred"] = cosine_similarity(sentence_vec,window[1])
                row["similarity_to_prev_ref"] = cosine_similarity(ref_vector,window[0])
                row["similarity_to_pred_ref"] = cosine_similarity(ref_vector,window[1])
                max_min_stats=save_max_mix_stats(max_min_stats,row,query)
                rows[run_name]=row
    return rows,max_min_stats