from Experiments.experiment_data_processor import create_features_file
from Experiments.experiment_data_processor import wait_for_feature_file_to_be_deleted
from Experiments.experiment_data_processor import move_feature_file


wait_for_feature_file_to_be_deleted("features")
create_features_file("Features_testing","/lv_local/home/sgregory/auto_seo/new_merged_index_1_05/","/lv_local/home/sgregory/auto_seo/data/queries.xml","_1_05")
move_feature_file('features',"test")