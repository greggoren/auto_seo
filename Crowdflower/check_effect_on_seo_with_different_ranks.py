from CrossValidationUtils.rankSVM_model_chooser import choose_model
from Crowdflower.ban_non_coherent_docs import get_scores,sort_files_by_date,retrieve_initial_documents,ban_non_coherent_docs,get_dataset_stas,get_banned_queries
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from Crowdflower import create_full_ds_per_task as mturk_ds_creator
import params
import sys
from Crowdflower.create_unified_experiment import read_seo_score,modify_seo_score_by_demotion,create_coherency_features,rewrite_fetures,create_harmonic_mean_score,create_weighted_mean_score,get_histogram,write_weighted_results
from CrossValidationUtils.svm_handler import svm_handler
from CrossValidationUtils.evaluator import eval
from CrossValidationUtils.rankSVM_crossvalidation import get_average_score_increase
from CrossValidationUtils.random_baseline import run_random

def read_chosen_model_file(filename):
    result={}
    with open(filename) as file:
        for line in file:
            method = line.split()[0]
            filename = line.split()[1]
            C= filename.split("_")[2].replace(".txt","")
            result[method]=C
    return result

def create_features_for_different_ranks(seo_feature_file, ref_index, top_docs_index, seo_qrels_file,coherency_feature_names):
    new_features_file = "seo_features_"+str(ref_index)
    coherency_features_set,max_min_stats = create_coherency_features(ref_index,top_docs_index)
    seo_scores = read_seo_score(seo_qrels_file)
    rewrite_fetures(seo_scores,coherency_features_set,seo_feature_file,new_features_file,coherency_feature_names,"qrels_"+str(ref_index),max_min_stats)
    doc_name_index = create_doc_name_index(new_features_file)
    return new_features_file,doc_name_index,seo_scores

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


def run_chosen_model_for_stats(chosen_models, method, qrels_file, feature_file, doc_name_index, seo_scores,base_features_file,ref_index,beta=""):
    chosen_model_parameter = chosen_models[method]
    svm = svm_handler()
    model_file = svm.learn_svm_rank_model(base_features_file, method, chosen_model_parameter)
    #
    evaluator = eval(["map", "ndcg", "P.2", "P.5"])
    scores_file = svm.run_svm_rank_model(feature_file, model_file, method)

    results = retrieve_scores(scores_file)
    trec_file = create_trec_eval_file(doc_name_index, results, method+"_"+ref_index)
    final_trec_file = evaluator.order_trec_file(trec_file)
    increase_stats = get_average_score_increase(seo_scores, final_trec_file)
    # qrels_file, final_trec_file, method + "_" + ref_index, None, increase_stats
    add=""
    if beta:
        add="_"+str(beta)
    summary_file =method+"_"+str(ref_index)+add+".tex"
    evaluator.run_trec_eval_on_test(qrels_file,summary_file,method+"_"+ref_index,None,increase_stats)
    return summary_file

if __name__=="__main__":
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q: ranked_lists[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists}
    dir = "nimo_annotations"
    sorted_files = sort_files_by_date(dir)

    original_docs = retrieve_initial_documents()
    scores = {}
    for k in range(4):
        needed_file = sorted_files[k]
        scores = get_scores(scores, dir + "/" + needed_file, original_docs)
    banned_queries = get_banned_queries(scores, reference_docs)
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
    tmp_aggregated_results = mturk_ds_creator.aggregate_results(sentence_tags, ident_tags)
    aggregated_results = ban_non_coherent_docs(banned_queries, tmp_aggregated_results)
    coherency_features = ["similarity_to_prev", "similarity_to_ref_sentence", "similarity_to_pred",
                          "similarity_to_prev_ref", "similarity_to_pred_ref"]
    seo_scores_file = "labels_final1"
    tmp_seo_scores = read_seo_score(seo_scores_file)
    seo_scores = ban_non_coherent_docs(banned_queries, tmp_seo_scores)
    modified_scores = modify_seo_score_by_demotion(seo_scores, aggregated_results)
    seo_features_file = "new_sentence_features"
    coherency_features_set, max_min_stats = create_coherency_features()
    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"
    rewrite_fetures(modified_scores, coherency_features_set, seo_features_file, new_features_with_demotion_file,
                    coherency_features, new_qrels_with_demotion_file, max_min_stats)

    original_features_file_2 = "new_sentence_features_2"
    original_features_file_3 = "new_sentence_features_3"
    original_features_file_4 = "new_sentence_features_4"
    original_qrels_file_2 = "labels_2"
    original_qrels_file_3= "labels_3"
    original_qrels_file_4= "labels_4"

    print("creating features rank 2")
    feature_file_rank_2, doc_name_index_2, seo_score_2 = create_features_for_different_ranks(original_features_file_2, 1, 1,original_qrels_file_2,coherency_features)
    print("creating features rank 3")
    feature_file_rank_3, doc_name_index_3, seo_score_3 = create_features_for_different_ranks(original_features_file_3, 2, 2,original_qrels_file_3,coherency_features)
    print("creating features rank 4")
    feature_file_rank_4, doc_name_index_4, seo_score_4 = create_features_for_different_ranks(original_features_file_4, 3, 3,original_qrels_file_4,coherency_features)
    method = "demotion"
    chosen_models_file_name = "chosen_models_demotion"
    chosen_models = read_chosen_model_file(chosen_models_file_name)
    run_chosen_model_for_stats(chosen_models,method,original_qrels_file_2,feature_file_rank_2,doc_name_index_2,seo_score_2,new_features_with_demotion_file,"2")
    run_chosen_model_for_stats(chosen_models,method,original_qrels_file_3,feature_file_rank_3,doc_name_index_3,seo_score_3,new_features_with_demotion_file,"3")
    run_chosen_model_for_stats(chosen_models,method,original_qrels_file_4,feature_file_rank_4,doc_name_index_4,seo_score_4,new_features_with_demotion_file,"4")
    run_random(original_features_file_2,original_qrels_file_2,"2",seo_score_2)
    run_random(original_features_file_3,original_qrels_file_3,"3",seo_score_3)
    run_random(original_features_file_4,original_qrels_file_2,"4",seo_score_4)




    betas=[0,0.5,1,2]
    chosen_models_file_name = "chosen_models_harmonic"
    flag = True
    last = False
    chosen_models = read_chosen_model_file(chosen_models_file_name)
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_" + str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_" + str(beta)
        harmonic_mean_scores = create_harmonic_mean_score(seo_scores, aggregated_results, beta)
        rewrite_fetures(harmonic_mean_scores, coherency_features_set, seo_features_file,
                        new_features_with_harmonic_file,
                        coherency_features, new_qrels_with_harmonic_file, max_min_stats)
        method = "harmonic_"+str(beta)
        if beta==betas[-1]:
            last=True
        summary_file_2=run_chosen_model_for_stats(chosen_models, method, original_qrels_file_2, feature_file_rank_2, doc_name_index_2,
                                   seo_score_2, new_features_with_harmonic_file, "2")

        summary_file_3=run_chosen_model_for_stats(chosen_models, method, original_qrels_file_3, feature_file_rank_3, doc_name_index_3,
                                   seo_score_3, new_features_with_harmonic_file, "3")


        summary_file_4=run_chosen_model_for_stats(chosen_models, method, original_qrels_file_4, feature_file_rank_4, doc_name_index_4,
                                   seo_score_4, new_features_with_harmonic_file, "4")
        write_weighted_results(summary_file_2, "summary_" + method + "_2.tex", beta, "RankSVM", flag, last)
        write_weighted_results(summary_file_3, "summary_" + method + "_3.tex", beta, "RankSVM", flag, last)
        write_weighted_results(summary_file_4, "summary_" + method + "_4.tex", beta, "RankSVM", flag, last)
        flag=False

    betas = [i / 10 for i in range(0, 11)]
    chosen_models_file_name = "chosen_models_weighted"
    chosen_models = read_chosen_model_file(chosen_models_file_name)
    flag=True
    last=False
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_" + str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_" + str(beta)
        weighted_mean_scores = create_weighted_mean_score(seo_scores, aggregated_results, beta)
        rewrite_fetures(weighted_mean_scores, coherency_features_set, seo_features_file,
                        new_features_with_weighted_file,
                        coherency_features, new_qrels_with_weighted_file, max_min_stats)
        method = "harmonic_" + str(beta)
        run_chosen_model_for_stats(chosen_models, method, original_qrels_file_2, feature_file_rank_2, doc_name_index_2,
                                   seo_score_2, new_features_with_weighted_file, "2")
        run_chosen_model_for_stats(chosen_models, method, original_qrels_file_3, feature_file_rank_3, doc_name_index_3,
                                   seo_score_3, new_features_with_weighted_file, "3")
        run_chosen_model_for_stats(chosen_models, method, original_qrels_file_4, feature_file_rank_4, doc_name_index_4,
                                   seo_score_4, new_features_with_weighted_file, "4")

        if beta==betas[-1]:
            last=True
        write_weighted_results(summary_file_2, "summary_" + method + "_2.tex", beta, "RankSVM", flag, last)
        write_weighted_results(summary_file_3, "summary_" + method + "_3.tex", beta, "RankSVM", flag, last)
        write_weighted_results(summary_file_4, "summary_" + method + "_4.tex", beta, "RankSVM", flag, last)
        flag = False
    print("Histograms:")
    print("2",get_histogram(seo_score_2))
    print("3",get_histogram(seo_score_3))
    print("4",get_histogram(seo_score_4))



