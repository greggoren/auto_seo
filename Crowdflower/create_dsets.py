from Preprocess.preprocess import retrieve_ranked_lists,load_file
from Preprocess.preprocess import retrieve_sentences
import params
from SentenceRanking.sentence_parse import map_sentences, map_set_of_sentences
import csv

ranked_lists = retrieve_ranked_lists("trec_file")
reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists[q][:3] for q in ranked_lists}
a_doc_texts = load_file("documents.trectext")
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04"):
        doc_texts[doc]=a_doc_texts[doc]
sentence_map=map_set_of_sentences(doc_texts,winner_docs)
rows ={}
i=1
for query  in sentence_map:
    reference = reference_docs[query]
    sentences =[s.replace("\"","") for s in retrieve_sentences(doc_texts[reference])]
    for sentence in sentence_map[query]:
        sentence_text = sentence_map[query][sentence].replace("\"","")
        for j in range(len(sentences)):
            row = {}
            if len(sentences)==1:
                new_doc=sentence_text
            else:
                if j==0:
                    new_doc=sentence_text+"\n"+"\n".join(sentences[1:])
                elif j+1==len(sentences):
                    new_doc="\n".join(sentences[:j])+"\n"

                else:
                    new_doc = "\n".join(sentences[:j])+"\n"+sentence_text+"\n"+"\n".join(sentences[j+1:])
            row["id"]=sentence+"_"+str(j+1)

            row["modified_doc"] = new_doc.replace("\"","")
            original_doc = "\n".join(sentences)
            row["original_doc"] = original_doc.replace("\"", "")
            rows[i]=row
            i+=1


fieldnames = list(rows[1].keys())
with open("test.csv","w",encoding="utf8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for t in rows:
        row = rows[t]
        writer.writerow(row)
