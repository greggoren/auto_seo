import params
from Preprocess.preprocess import load_file
import csv
from random import shuffle
def analyze_topics(qrels_file, queries):
    stats = {}
    with open(qrels_file) as qrels:
        for line in qrels:
            query = line.split()[0]
            if  query not in queries:
                continue

            if line.split()[3].rstrip()=="0":
                continue
            if query not in stats:
                stats[query]={}
            topic = line.split()[1]
            if topic not in stats[query]:
                stats[query][topic]=0
            stats[query][topic]+=1
    return stats


def get_queries(topics_file):
    queries = []
    with open(topics_file) as file:
        for line in file:
            if line.__contains__("<topic "):
                splits = line.split()
                query = splits[1].split("=")[1].replace("\"","").zfill(3)
                queries.append(query)
    return queries

def transform(doc_text):
    result = {}
    i=0
    for doc in doc_text:
        result[i]={}
        result[i]["name"]=doc.split("-")[2]
        result[i]["text"]=doc_text[doc]
        i+=1
    return result

topics_file = "topics.full.xml"
qrels_file = "topic_rel"

queries = get_queries(topics_file)
# a_doc_texts = load_file("documents.trectext")
# doc_texts={}
# for doc in a_doc_texts:
#     if doc.__contains__("ROUND-00"):
#         doc_texts[doc]=a_doc_texts[doc]
# res = transform(doc_texts)
# f = open("original_docs.csv","w",encoding="utf-8",newline='')
# writer = csv.DictWriter(f,fieldnames=["name","text"])
# for i in res:
#
#     writer.writerow(res[i])
# f.close()



stats = analyze_topics(qrels_file,queries)
shuffle(queries)
sorted_q = sorted(queries)

for query in sorted_q:
    print("for query number",query)
    for topic in stats[query]:
        print(topic,":",stats[query][topic])