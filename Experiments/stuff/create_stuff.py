from Preprocess.preprocess import load_file
from Experiments.experiment_data_processor import create_trectext
import params

a_doc_texts = load_file(params.trec_text_file)
doc_texts = {}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04"):
        doc_texts[doc] = a_doc_texts[doc]

summaries = {}

create_trectext(document_text=doc_texts, avoid=[], summaries=summaries, run_name="")