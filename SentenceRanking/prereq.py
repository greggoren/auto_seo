from Preprocess.preprocess import retrieve_ranked_lists,load_file
from SentenceRanking.sentence_parse import map_sentences, map_set_of_sentences
import params

def determine_indexes(doc,ranked_list):
    return min(ranked_list.index(doc),3)



new_ranked_list ="trec_file04"
ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
ranked_lists_new = retrieve_ranked_lists(new_ranked_list)
reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists_new[q][0].replace("EPOCH","ROUND") for q in ranked_lists_new}
top_docs = {q:ranked_lists_new[q][:determine_indexes(reference_docs[q],ranked_lists_new[q])] for q in ranked_lists_new}
a_doc_texts = load_file(params.trec_text_file)
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04"):
        doc_texts[doc]=a_doc_texts[doc]
sentence_map=map_set_of_sentences(doc_texts,top_docs)
f = open("sentences_top","w")
for query in sentence_map:
    for sentence in sentence_map[query]:
        f.write(sentence+"\t"+sentence_map[query][sentence].replace("\n","")+"\n")
f.close()

f = open("topDocs","w")
for query in top_docs:
    for doc in top_docs[query]:
        f.write(query+"\t"+doc.replace("EPOCH","ROUND")+"\n")
f.close()