from Preprocess.preprocess import retrieve_ranked_lists,load_file,get_queries_data
from Preprocess.preprocess import retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import csv
from copy import deepcopy
from random import uniform


# def convert_sentences_to_sentence_task(sentences):
#     new_text =""
#     for j in range(len(sentences)):
#         new_text+=str(j+1)+") "+sentences[j].replace("\n","")+"\n\n\n\n"
#     return new_text
def convert_to_quality_ds(data,headers):
    new_rows = {}
    for i in data:
        row = data[i]
        new_row = {header:row[header] for header in row if header in headers}
        doc = row["check_one_gold"].lower()
        text = row[doc]
        new_row["document"]=text
        new_rows[i]=new_row
        new_row["check_one_gold"]=""
    return new_rows

def convert_text_to_sentence_task(text):
    sentences = retrieve_sentences(text)
    new_text =""
    for j in range(len(sentences)):
        new_text+=str(j+1)+") "+sentences[j].replace(u"\u009D","").replace("\n","")+" <br><br>\n"
    return new_text

ranked_lists = retrieve_ranked_lists("ranked_lists/trec_file04")
query_data=get_queries_data("topics.full.xml")
reference_docs = {q:ranked_lists[q][1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists[q][:1] for q in ranked_lists}
a_doc_texts = load_file("documents.trectext")
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-04"):
        doc_texts[doc]=a_doc_texts[doc]
sentence_map=map_set_of_sentences(doc_texts,winner_docs)
rows ={}
i=1
sentence_data={}

for query in sentence_map:
    reference = reference_docs[query]
    text = doc_texts[reference][1:].replace(u"\u009D","")
    sentences =[s.replace("\"","") for s in retrieve_sentences(text)]
    for sentence in sentence_map[query]:

        sentence_text = sentence_map[query][sentence].replace("\"","")
        for j in range(len(sentences)):
            row = {}
            sentence_row={}
            copied_text = deepcopy(text).replace(u"\u009D","")

            if j+1!=len(sentences):
                insert = sentence_text.replace("\n","")
                copied_text=copied_text.replace(sentences[j],insert)
            else:
                copied_text = copied_text.replace(sentences[j], sentence_text)
            sentence_line = convert_text_to_sentence_task(copied_text)

            # if len(sentences)==1:
            #     new_doc=sentence_text
            # else:
            #     if j==0:
            #         new_doc=sentence_text+"\n"+"\n".join(sentences[1:])
            #     elif j+1==len(sentences):
            #         new_doc="\n".join(sentences[:j])+"\n"
            #
            #     else:
            #         new_doc = "\n".join(sentences[:j])+"\n"+sentence_text+"\n"+"\n".join(sentences[j+1:])
            row["ID"] = sentence + "_" + str(j + 1)
            sentence_row["ID"]=row["ID"]
            sentence_row["check_one_gold"] = str(j+1)
            sentence_row["text"]=sentence_line
            probablity = uniform(0,1)
            if probablity<0.5:


                row["document1"] = copied_text
                row["document2"] = text
                row["check_one_gold"]="Document1"
            else:
                row["document2"] = copied_text
                row["document1"] = text
                row["check_one_gold"] = "Document2"
            row["query"]=query_data[query]["query"]
            sentence_row["query"]=row["query"]
            row["description"]=query_data[query]["description"]
            sentence_row["description"] = row["description"]
            row["check_one_gold_reason"]=""
            sentence_row["check_one_gold_reason"] = row["check_one_gold_reason"]
            row["_golden"]=""
            sentence_row["_golden"] = row["_golden"]
            sentence_data[i]=sentence_row
            rows[i]=row
            i+=1


fieldnames = ["ID","document1",	"document2","query","description","check_one_gold","check_one_gold_reason","_golden"]
with open("comparison_04_2.csv","w",encoding="utf-8",newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for t in rows:
        row = rows[t]
        writer.writerow(row)
quality_headers = ["ID","query","check_one_gold","check_one_gold_reason","_golden"]
quality_data = convert_to_quality_ds(rows,quality_headers)

# quality_headers.insert(1,"document")
# with open("quality.csv","w",encoding="utf8",newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=quality_headers)
#     writer.writeheader()
#     for t in quality_data:
#         row = quality_data[t]
#         writer.writerow(row)


sentence_task_headers=fieldnames = ["ID","text","query","description","check_one_gold","check_one_gold_reason","_golden"]
with open("sentence_04_2.csv","w",encoding="utf-8",newline='') as f:
    writer = csv.DictWriter(f, fieldnames=sentence_task_headers)
    writer.writeheader()
    for t in sentence_data:
        row = sentence_data[t]
        writer.writerow(row)
