from CrossValidationUtils.rankSVM_model_chooser import choose_model
from Crowdflower.ban_non_coherent_docs import get_scores,sort_files_by_date,retrieve_initial_documents,ban_non_coherent_docs,get_dataset_stas,get_banned_queries
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from Crowdflower import create_full_ds_per_task as mturk_ds_creator
import params
import sys
from Crowdflower.create_unified_experiment import read_seo_score,modify_seo_score_by_demotion,rewrite_fetures,create_harmonic_mean_score,create_weighted_mean_score
from Crowdflower.seo_utils import create_coherency_features

if __name__=="__main__":

    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"

    choose_model(new_features_with_demotion_file,new_features_with_demotion_file,"demotion")
    betas=[1,]
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_" + str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_" + str(beta)

        choose_model(new_features_with_harmonic_file,new_qrels_with_harmonic_file,"harmonic",str(beta))

    betas = [0.5,]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_" + str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_" + str(beta)

        choose_model(new_features_with_weighted_file,new_qrels_with_weighted_file,"weighted",str(beta))