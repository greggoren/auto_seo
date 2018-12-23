from CrossValidationUtils.rankSVM_model_chooser import choose_model
from Crowdflower.ban_non_coherent_docs import get_scores,sort_files_by_date,retrieve_initial_documents,ban_non_coherent_docs,get_dataset_stas,get_banned_queries
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from Crowdflower import create_full_ds_per_task as mturk_ds_creator
import params
import sys
from Crowdflower.create_unified_experiment import read_seo_score,modify_seo_score_by_demotion,create_coherency_features,rewrite_fetures,create_harmonic_mean_score,create_weighted_mean_score

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
    choose_model(new_features_with_demotion_file,new_features_with_demotion_file,"demotion")
    betas=[0,0.5,1,2]
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_" + str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_" + str(beta)
        harmonic_mean_scores = create_harmonic_mean_score(seo_scores, aggregated_results, beta)
        rewrite_fetures(harmonic_mean_scores, coherency_features_set, seo_features_file,
                        new_features_with_harmonic_file,
                        coherency_features, new_qrels_with_harmonic_file, max_min_stats)
        choose_model(new_features_with_harmonic_file,new_qrels_with_harmonic_file,"harmonic",str(beta))

    betas = [i/10 for i in range(0,11)]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_" + str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_" + str(beta)
        weighted_mean_scores = create_weighted_mean_score(seo_scores, aggregated_results, beta)
        rewrite_fetures(weighted_mean_scores, coherency_features_set, seo_features_file,
                        new_features_with_weighted_file,
                        coherency_features, new_qrels_with_weighted_file, max_min_stats)
        choose_model(new_features_with_weighted_file,new_qrels_with_weighted_file,"weighted",str(beta))