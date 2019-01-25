from pymongo import MongoClient
from CompetitionBot.analyze_positions import ASR_MONGO_HOST,ASR_MONGO_PORT
import csv
from datetime import datetime
import os
import xml.etree.ElementTree as ET


def set_waterloo_scores_dummies(dummy_waterloo):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[8:]
    for iteration in iterations:
        docs = db.archive.find({"query_id":{"$regex":".*_2"},"username":{"$regex":"dummy_doc.*"},"iteration":iteration})
        for doc in docs:
            if "bot_method" in doc:
                continue
            doc_name = doc["doc_name"]
            if doc_name.split("-")[2]!="013":
                waterloo = dummy_waterloo[doc_name]
            else:
                needed_doc = next(db.archive.find({"query_id":doc["query_id"],"doc_name":doc_name}))
                waterloo = needed_doc["waterloo"]
            doc["waterloo"]=waterloo

            print(doc["query_id"],doc["username"],waterloo)
            # db.archive.save(doc)

def retrieve_initial_documents(epoch):
    initial_query_docs={}
    tree = ET.parse('documents.trectext')
    root = tree.getroot()
    for doc in root:
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                name=att.text
            else:
                if name.__contains__("ROUND-"+epoch+"-"):
                    text = str(att.text).replace('&','and').rstrip().replace("\n","").replace(" ","").lower()
                    initial_query_docs[text]=name
    return initial_query_docs


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
        dates[filename]=max

    sorted_files = sorted(list(dates.keys()),key=lambda x:dates[x])
    for file in sorted_files:
        print(file,dates[file])

    return sorted_files

def get_scores(filename,reverse):
    scores = {}
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            text = row["post_content"].replace('&','and').rstrip().replace("\n","").replace(" ","").lower()
            if text in reverse:
                doc = reverse[text]
                if doc not in scores:
                    scores[doc]=100
                if "this_document_is" in row:
                    if row["this_document_is"].lower()!="valid":
                        scores[doc]-=20
                else:
                    if row["check_one"].lower() != "valid":
                        scores[doc] -= 20
    return scores


def update_dummies_waterloo(dummies_waterloo,current_score,epoch):
    for doc in current_score:
        dummies_waterloo[doc]=current_score[doc]
    if epoch=="01":
        return dummies_waterloo
    tmp_dict = {}
    for doc in dummies_waterloo:

        doc_epoch = doc.split("-")[1]
        query = doc.split("-")[2]
        id = doc.split("-")[3].rstrip()

        if int(doc_epoch)<int(epoch)-1:
            current_doc_epoch = "ROUND-" + epoch + "-" + query + "-" + id
            if current_doc_epoch not in dummies_waterloo:
                tmp_dict[current_doc_epoch]=dummies_waterloo[doc]
    for doc in tmp_dict:
        dummies_waterloo[doc]=tmp_dict[doc]
    return dummies_waterloo

def retrieve_waterloo_for_dummies(annotations_dir):
    sorted_files = sort_files_by_date(annotations_dir)
    dummies_watreloo = {}
    for i,file in enumerate(sorted_files,start=1):
        epoch = str(i).zfill(2)
        doc_texts_reverse = retrieve_initial_documents(epoch)
        waterloo_scores = get_scores(file,doc_texts_reverse)
        dummies_watreloo = update_dummies_waterloo(dummies_watreloo,waterloo_scores,epoch)
    return dummies_watreloo



def read_group_dir(dir):
    files = sorted(list(os.listdir(dir)))
    stats={}
    for i,file in enumerate(files):
        initial_results = read_file(dir+"/"+file)
        stats[i]=initial_results
    return stats

def read_file(filename):
    stats={}
    with open(filename,encoding="utf-8") as file:
        ref = "valid"
        reader = csv.DictReader(file)
        for row in reader:
            query = row["query_id"]
            user = row["username"]
            annotation = row["this_document_is"].lower()
            label = 0
            if annotation!=ref:
                label = 1
            if query not in stats:
                stats[query]={}
            if user not in stats[query]:
                stats[query][user] = []
            stats[query][user].append(label)
        for query in stats:
            for user in stats[query]:
                stats[query][user]= 100-20*sum(stats[query][user])
    return stats

def set_bots_waterloo(stats):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[8:]
    for i,iteration in enumerate(iterations):
        docs = db.archive.find({"username":{"$regex":"dummy_doc.*"},"iteration":iteration})
        for doc in docs:
            if "bot_method" not in doc:
                continue
            query = doc["query_id"]
            username = doc["username"]
            waterloo = stats[i][query][username]
            doc["waterloo"]=waterloo
            print(doc["query_id"], doc["username"],waterloo)
            # db.archive.save(doc)

if __name__=="__main__":
    print("--===DUMMY==--")
    dummy_waterloo=retrieve_waterloo_for_dummies("nimo_annotations")
    set_waterloo_scores_dummies(dummy_waterloo)
    stats = read_group_dir("annotations")
    print("--===BOT==--")
    set_bots_waterloo(stats)



