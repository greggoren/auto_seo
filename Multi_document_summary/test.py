from Multi_document_summary.multi_doc_summarization import create_multi_document_summarization
from Preprocess.preprocess import retrieve_ranked_lists
from Preprocess.preprocess import load_file
from Preprocess.preprocess import retrieve_sentences
import params
import pickle
import re
import pyndri
def retrieve_query_names():
    query_mapper = {}
    with open(params.query_description_file,'r') as file:
        for line in file:
            data = line.split(":")
            query_mapper[data[0]]=data[1].rstrip()
    return query_mapper
#
#
#
#
ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
reference_docs = {q:ranked_lists[q][-1] for q in ranked_lists}
queries = retrieve_query_names()
doc_texts = load_file(params.trec_text_file)


summaries={}
for query in reference_docs:
    if query:
        reference_doc=reference_docs[query]
        for doc in ranked_lists[query]:
            sentences = retrieve_sentences(doc_texts[doc])
            for sentence in sentences:
                sentence = re.sub("’", "", sentence)
                sentence = re.sub("–", " ", sentence)
                sentence = re.sub("—", " ", sentence)
                sentence = re.sub("“", " ", sentence)
                sentence = re.sub("”", " ", sentence)
                sentence = re.sub("…", " ", sentence)
                words = sentence.split()
                tokens = []
                for word in words:
                    try:
                        print(word)
                        modified = pyndri.escape(word)
                        if not modified.isspace():
                            tokens.extend(pyndri.tokenize(modified))
                    except:
                        print(word)
# sentence="fear—there"
# # sentence = re.sub("—","",sentence)
# sentence = re.sub("–","",sentence)

# print(pyndri.tokenize(pyndri.escape(sentence)))
