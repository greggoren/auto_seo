from pymongo import MongoClient
import csv
from random import uniform
from Preprocess.preprocess import retrieve_sentences
ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017

def get_reference_documents():
    reference_docs={}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    docs = db.documents.find({})
    for doc in docs:
        if "bot_method" in doc:
            query = doc["query_id"]
            if query not in reference_docs:
                reference_docs[query]=[]
            reference_docs[query].append(doc["username"])
    return reference_docs

def create_data_set_fe():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(db.archive.distinct("iteration"))[8:]
    f = open("bot_ds_rel.csv", "w", newline='', encoding="utf-8")
    for i,iteration in enumerate(iterations):


        writer = csv.DictWriter(f,fieldnames=["current_document","query_id","query","username","description","iteration"])
        writer.writeheader()
        documents = db.archive.find({"iteration":iteration})
        for document in documents:
            if "bot_method" in document and document["query_id"].__contains__("_2"):
                obj={}
                obj["current_document"]=document["text"]
                obj["query_id"] = document["query_id"]
                obj["query"] = document["query"]
                obj["username"]=document["username"]
                obj["description"]=document["description"]
                obj["iteration"]=document["iteration"]
                writer.writerow(obj)
        f.close()

def convert_text_to_sentence_task(text):
    sentences = retrieve_sentences(text)
    new_text =""
    for j in range(len(sentences)):
        new_text+=str(j+1)+") "+sentences[j].replace(u"\u009D","").replace("\n","")+" <br><br>\n"
    return new_text

def create_data_for_mturk(reference_docs, index):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))
    needed_iterations = iterations[index - 1:index + 1]
    print(needed_iterations)
    f = open("bot_comparison_"+str(index)+".csv","w",newline='',encoding="utf-8")
    f2 = open("bot_sentences_"+str(index)+".csv","w",newline='',encoding="utf-8")
    writer = csv.DictWriter(f,fieldnames=["ID","document1","document2","query_id","query","description","method","check_one_gold"])
    writer2 = csv.DictWriter(f2,fieldnames=["ID","text","query_id","query","description","method","check_one_gold"])
    writer.writeheader()
    writer2.writeheader()
    for query_id in reference_docs:
        for doc in reference_docs[query_id]:
            old_doc = next(db.archive.find({"iteration":needed_iterations[0],"username":doc,"query_id":query_id}))
            new_doc = next(db.archive.find({"iteration":needed_iterations[1],"username":doc,"query_id":query_id}))

            obj = {}
            obj["ID"]=doc
            if uniform(0,1)<0.5:
                obj["document1"]=old_doc["text"]
                obj["document2"]=new_doc["text"]
                obj["check_one_gold"]="2"
            else:
                obj["document2"] = old_doc["text"]
                obj["document1"] = new_doc["text"]
                obj["check_one_gold"] = "1"
            obj["query"]=old_doc["query"]
            obj["query_id"]=query_id
            obj["description"]=old_doc["description"]
            obj["method"]=new_doc["bot_method"]
            writer.writerow(obj)
            sentences_line = convert_text_to_sentence_task(new_doc["text"])
            obj2={}
            obj2["ID"]=doc
            obj2["query"] = old_doc["query"]
            obj2["query_id"] = query_id
            obj2["description"] = old_doc["description"]
            obj2["method"] = new_doc["bot_method"]
            obj2["text"] = sentences_line
            obj2["check_one_gold"]=""
            writer2.writerow(obj2)

    f.close()
    f2.close()

if __name__=="__main__":
    # reference_docs = get_reference_documents()
    # create_data_for_mturk(reference_docs,11)
    create_data_set_fe()