from Crowdflower import create_full_ds_per_task as mturk_ds_creator
from utils import cosine_similarity
from SentenceRanking.sentence_features_experiment import get_sentence_vector
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import params
from w2v.train_word2vec import WordToVec
from CrossValidationUtils.rankSVM_crossvalidation import cross_validation
from CrossValidationUtils.random_baseline import run_random

def read_seo_score(labels):
    scores = {}
    with open(labels) as labels_file:
        for line in labels_file:
            id = line.split()[2]
            score = int(line.split()[3].rstrip())
            scores[id]=score
    return scores


def get_level(score):
    demotion_level = 0
    if score >=0 and score <2:
        demotion_level=2
    elif score >=2 and score <4:
        demotion_level=1
    return demotion_level


def modify_seo_score_by_demotion(seo_scores, coherency_scores):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        demotion_level = get_level(coherency_score)
        new_score = max(current_score-demotion_level,0)
        new_scores[id] = new_score
    return new_scores


def create_harmonic_mean_score(seo_scores,coherency_scores):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        new_coherency_score = coherency_score*(4/5)
        if new_coherency_score!=0 or current_score!=0:
            harmonic_mean = (2*new_coherency_score*current_score)/(new_coherency_score+current_score)
        else:
            harmonic_mean = 0
        new_scores[id]=harmonic_mean
    return new_scores

def create_weighted_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        new_coherency_score = coherency_score * (4 / 5)
        new_score = current_score*beta+new_coherency_score*(1-beta)
        new_scores[id]=new_score
    return new_scores



def create_coherency_features(stats=[]):
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
                # if run_name not in stats:
                #     continue
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
                row["similarity_to_prev"]=cosine_similarity(sentence_vec,window[0])
                row["similarity_to_ref_sentence"] = similarity_to_ref_sentence
                row["similarity_to_pred"] = cosine_similarity(sentence_vec,window[1])
                row["similarity_to_prev_ref"] = cosine_similarity(ref_vector,window[0])
                row["similarity_to_pred_ref"] = cosine_similarity(ref_vector,window[1])
                rows[run_name]=row
    return rows


def rewrite_fetures(new_scores, coherency_features_set, old_features_file, new_features_filename, coherency_features_names,qrels_name):
    f = open(new_features_filename,"w")
    qrels = open(qrels_name,"w")
    with open(old_features_file) as file:
        for line in file:
            qid = line.split()[1]
            query = qid.split(":")[1]
            features = line.split()[2:-3]
            number_of_features = len(features)
            id = line.split(" # ")[1].rstrip()
            coherency_features = [str(i)+":"+str(coherency_features_set[id][feature]) for i,feature in enumerate(coherency_features_names,start=number_of_features+1)]
            new_line = str(new_scores[id]) + " " + qid + " " + " ".join(features) + " " + " ".join(coherency_features) + " # " + id + "\n"
            f.write(new_line)
            qrels.write(query+" 0 "+id+" "+str(new_scores[id])+"\n")
    f.close()
    qrels.close()


if __name__=="__main__":

    ident_filename_fe = "figure-eight/ident_current.csv"
    ident_filename_mturk = "Mturk/Manipulated_Document_Identification.csv"
    ident_fe = mturk_ds_creator.read_ds_fe(ident_filename_fe, True)
    ident_mturk = mturk_ds_creator.read_ds_mturk(ident_filename_mturk, True)

    ident_results = mturk_ds_creator.combine_results(ident_fe, ident_mturk)

    sentence_filename_fe = "figure-eight/sentence_current.csv"
    sentence_filename_mturk = "Mturk/Sentence_Identification.csv"
    sentence_filename_mturk_new = "Mturk/Sentence_Identification11.csv"
    sentence_fe = mturk_ds_creator.read_ds_fe(sentence_filename_fe)
    sentence_mturk = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk)
    sentence_mturk_new = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk_new)
    sentence_mturk = mturk_ds_creator.update_dict(sentence_mturk, sentence_mturk_new)

    sentence_results = mturk_ds_creator.combine_results(sentence_fe, sentence_mturk)

    sentence_tags = mturk_ds_creator.get_tags(sentence_results)
    ident_tags = mturk_ds_creator.get_tags(ident_results)
    aggregated_results = mturk_ds_creator.aggregate_results(sentence_tags,ident_tags)
    coherency_features = ["similarity_to_prev", "similarity_to_ref_sentence", "similarity_to_pred",
                          "similarity_to_prev_ref", "similarity_to_pred_ref"]
    seo_scores_file = "labels_final1"
    seo_scores = read_seo_score(seo_scores_file)
    modified_scores= modify_seo_score_by_demotion(seo_scores,aggregated_results)
    seo_features_file = "new_sentence_features"
    coherency_features_set = create_coherency_features()
    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"
    rewrite_fetures(modified_scores,coherency_features_set,seo_features_file,new_features_with_demotion_file,coherency_features,new_qrels_with_demotion_file)
    cross_validation(new_features_with_demotion_file, new_qrels_with_demotion_file, "summary_labels_demotion.tex", "svm_rank",
                     ["map", "ndcg", "P.2", "P.5"], "")
    run_random(new_features_with_demotion_file,new_qrels_with_demotion_file,"demotion")
    new_features_with_harmonic_file = "all_seo_features_harmonic"
    new_qrels_with_harmonic_file = "seo_harmonic_qrels"
    harmonic_mean_scores = create_harmonic_mean_score(seo_scores,aggregated_results)
    rewrite_fetures(harmonic_mean_scores, coherency_features_set, seo_features_file, new_features_with_harmonic_file,
                    coherency_features, new_qrels_with_harmonic_file)
    cross_validation(new_features_with_harmonic_file, new_qrels_with_harmonic_file, "summary_labels_harmonic.tex",
                     "svm_rank",
                     ["map", "ndcg", "P.2", "P.5"], "")
    run_random(new_features_with_demotion_file, new_qrels_with_demotion_file, "harmonic")
    betas = [i/10 for i in range(0,11)]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_"+str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_"+str(beta)
        weighted_mean_scores = create_weighted_mean_score(seo_scores, aggregated_results,beta)
        rewrite_fetures(weighted_mean_scores, coherency_features_set, seo_features_file, new_features_with_weighted_file,
                        coherency_features, new_qrels_with_weighted_file)
        cross_validation(new_features_with_demotion_file,new_qrels_with_weighted_file, "summary_labels_weighted"+str(beta)+".tex","svm_rank",["map", "ndcg", "P.2", "P.5"], "")
        run_random(new_features_with_weighted_file, new_qrels_with_weighted_file, "weighted_"+str(beta))