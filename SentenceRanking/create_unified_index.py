from Preprocess.preprocess import load_file
from Experiments.experiment_data_processor import create_trectext,merge_indexes_for_experiments,create_index,create_working_sets_by_round
import params

a_doc_texts = load_file(params.trec_text_file)
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04") or doc.__contains__("ROUND-06"):
        doc_texts[doc]=a_doc_texts[doc]
trec_text_file ="trec_text_sentnece_experiments"
create_trectext(doc_texts,[],trec_text_file,"dummy")
working_set_file_basename = "working_set_sentence_experiments"
create_working_sets_by_round(doc_texts,working_set_file_basename)
current_index = create_index(trec_text_file,"sentence_experiment")
base_index = "/home/greg/cluewebindex"
merge_indexes_for_experiments(merged_index="/home/greg/mergedindex",index1=base_index,index2=current_index)





