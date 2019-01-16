import os
import csv
from datetime import datetime
import xml.etree.ElementTree as ET

def read_file_get_max_data(filename):
    max =""
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if not max:
                max = datetime.strptime(row["_created_at"],'%m/%d/%Y %H:%M:%S')
            if datetime.strptime(row["_created_at"],'%m/%d/%Y %H:%M:%S')>max:
                max = datetime.strptime(row["_created_at"],'%m/%d/%Y %H:%M:%S')
    return max

def sort_files_by_date(dir):
    dates = {}
    for file in os.listdir(dir):
        filename = dir+"/"+file
        max = read_file_get_max_data(filename)
        dates[file]=max

    sorted_files = sorted(list(dates.keys()),key=lambda x:dates[x])
    for file in sorted_files:
        print(file,dates[file])

    return sorted_files


def retrieve_initial_documents():
    initial_query_docs={}
    tree = ET.parse('documents.trectext')
    root = tree.getroot()
    for doc in root:
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                name=att.text
            else:
                if name.__contains__("ROUND-04-") or name.__contains__("ROUND-06-"):
                    text = str(att.text).rstrip().replace("\n","").replace(" ","").replace('&','and').replace("'","").replace(".","").lower()
                    initial_query_docs[text]=name
    return initial_query_docs

def get_scores(scores,filename,reverse,index):
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        seen=[]
        for row in reader:
            text = row["post_content"].rstrip().replace("\n","").replace(" ","").replace('&','and').replace("'","").replace(".","").lower()
            if text in reverse:
                doc = reverse[text]
                id = doc.split("-")[3]
                query = doc.split("-")[2]
                current_key1 = "ROUND-04-"+query+"-"+id
                current_key2 = "ROUND-06-"+query+"-"+id
                if index<=4 and current_key1 not in seen:
                    scores[current_key1]=0
                    scores[current_key2]=0
                    seen.append(current_key1)
                    if "this_document_is" in row:
                        if row["this_document_is"].lower()=="valid":
                            scores[current_key1]+=1
                            scores[current_key2]+=1
                    else:
                        if row["check_one"].lower() == "valid":
                            scores[current_key1] += 1
                            scores[current_key2] += 1
                else:
                    if current_key1 not in seen:

                        scores[current_key2] = 0
                        seen.append(current_key1)
                    if "this_document_is" in row:
                        if row["this_document_is"].lower() == "valid":
                            scores[current_key2] += 1
                    else:
                        if row["check_one"].lower() == "valid":
                            scores[current_key2] += 1
    return scores

def get_dataset_stas(dataset):
    stats={}
    for id in dataset:
        doc = id.split("_")[0]
        query = doc.split("-")[2]
        stats[query]=True
    return stats



def get_banned_queries(scores,reference_docs):
    banned = []
    for key in reference_docs:
        for query in reference_docs[key]:
            if scores[reference_docs[key][query]]<3:
                banned.append(query+key)
    return banned

def ban_non_coherent_docs(banned,dataset):
    final_dataset={}

    for key in dataset:
        final_dataset[key]={}
        for id in dataset[key]:
            doc = id.split("_")[0]
            query = doc.split("-")[2]+key
            if query not in banned:
                final_dataset[key][id]=dataset[key][id]
    return final_dataset