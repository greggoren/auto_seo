from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Experiments.experiment_data_processor import create_trectext, create_features_file_original, \
    create_trectext_original
from utils import run_command
from utils import run_bash_command
from Experiments.experiment_data_processor import merge_indices
from Experiments.experiment_data_processor import create_index
from Experiments.experiment_data_processor import create_features_file
from Experiments.model_handler import run_model
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
import params

def get_docs(doc_texts,round):
    result = {}
    index = str(round).zfill(2)
    for doc in doc_texts:
        if doc.__contains__("ROUND-"+index):
            result[doc]=doc_texts[doc]
    return result

def create_trec_eval_file(results,run_name):
    trec_file = "/home/greg/auto_seo/data/trec_file"+run_name+".txt"
    trec_file_access = open(trec_file, 'a')
    for doc in results:
        query = doc.split("-")[2]
        trec_file_access.write(query
             + " Q0 " + doc + " " + str(0) + " " + str(
                results[doc]) + " seo\n")
    trec_file_access.close()
    return trec_file

def order_trec_file(trec_file):
    final = trec_file.replace(".txt", "")
    command = "sort -k1,1 -k5nr -k2,1 " + trec_file + " > " + final
    for line in run_command(command):
        print(line)
    return final

ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists[q][:3] for q in ranked_lists}
doc_texts = load_file(params.trec_text_file)
merged_index=""
for index in range(1,4):
    doc_text_for_round = get_docs(doc_texts, round=index)
    trec_text_file = create_trectext_original(doc_text_for_round, [], str(index),[])
    new_index = create_index(trec_text_file,str(index))
    if merged_index:
        run_bash_command("rm -r "+merged_index)
    merged_index = merge_indices(new_index=new_index,run_name=str(index),new_index_name="merged_index")
    feature_file = "features"+ "_" + str(index)
    features_dir = "Features"
    create_features_file_original(features_dir=features_dir, index_path=merged_index, new_features_file=feature_file , run_name=str(index))
    index_doc_name = create_index_to_doc_name_dict(feature_file)
    scores_file = run_model(feature_file)
    results = retrieve_scores(index_doc_name, scores_file)
    trec_file = create_trec_eval_file(results,str(index))
    order_trec_file(trec_file)
    run_bash_command("rm "+trec_file)


