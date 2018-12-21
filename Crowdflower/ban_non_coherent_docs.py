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
                if name.__contains__("ROUND-04-"):
                    text = str(att.text).rstrip().replace("\n","").replace(" ","").replace('&','and').lower()
                    initial_query_docs[text]=name
    return initial_query_docs

def get_scores(filename,reverse):
    scores = {}
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["post_content"].replace('&','and').rstrip().replace("\n","").replace(" ","").lower()
            if text in reverse:
                doc = reverse[text]
                if doc not in scores:
                    scores[doc]=0
                if "this_document_is" in row:
                    if row["this_document_is"].lower()=="valid":
                        scores[doc]+=1
                else:
                    if row["check_one"].lower() == "valid":
                        scores[doc] += 1
    return scores

def get_dataset_stas(dataset):
    stats={}
    for id in dataset:
        doc = id.split("_")[0]
        query = doc.split("-")[2]
        stats[query]=True
    return stats

def ban_non_coherent_docs(scores,dataset):
    final_dataset={}
    for id in dataset:
        doc = id.split("_")[0]
        if scores[doc]>=3:
            final_dataset[id]=dataset[id]
    return final_dataset