from Preprocess.preprocess import retrieve_ranked_lists,load_file
from SentenceRanking.sentence_parse import map_sentences
import params



ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)

reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists[q][0].replace("EPOCH","ROUND") for q in ranked_lists}
top_docs = {q:ranked_lists[q][:3] for q in ranked_lists}
a_doc_texts = load_file(params.trec_text_file)
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04"):
        doc_texts[doc]=a_doc_texts[doc]
sentence_map=map_sentences(doc_texts,winner_docs)
f = open("sentences","w")
for query in sentence_map:
    for sentence in sentence_map[query]:
        f.write(sentence+"\t"+sentence_map[query][sentence].replace("\n","")+"\n")
f.close()

f = open("topDocs","w")
for query in top_docs:
    for doc in top_docs[query]:
        f.write(query+"\t"+doc.replace("EPOCH","ROUND")+"\n")
f.close()