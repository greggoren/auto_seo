from pymongo import MongoClient,ASCENDING
from Preprocess.preprocess import retrieve_sentences
import os

ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017

def create_former_winners_file(current_time):
    client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
    db = client.asr16
    iterations = db.archive.distinct("iteration")
    sorted_iterations = sorted(iterations)
    start_iter_index = 7
    needed_iterations = sorted_iterations[start_iter_index:]
    past_winners_dir = "past_winners/"+current_time+"/"
    if not os.path.exists(past_winners_dir):
        os.makedirs(past_winners_dir)
    past_winners_filename = past_winners_dir+"past_winners_text_"+current_time
    f = open(past_winners_filename,"w")
    for iteration in needed_iterations:
        documents = db.archive.find({"iteration":iteration,"position":1}).sort("query_id",1)
        for document in documents:
            query = document["query_id"]
            if not query.__contains__("_0") and not query.__contains__("_2"):
                continue
            text = document["text"]
            sentences = retrieve_sentences(text)
            f.write(query + "@@@" + " ".join([a.replace("\n", "").replace("\r","") for a in sentences]) + "\n")
    f.close()
    return past_winners_filename




def create_sentence_file(top_docs_file, ref_doc, query,current_time):
    sentence_files_dir = "sentence_files/"+current_time+"/"
    sentences_index={}
    sentences_index[query]={}
    if not os.path.exists(sentence_files_dir):
        os.makedirs(sentence_files_dir)
    sentence_filename = sentence_files_dir+"sentence_file_"+ref_doc+"_"+query+"_"+current_time
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    ref_text=next(db.documents.find({"query_id":query,"username":ref_doc}))["posted_document"]
    ref_sentences = retrieve_sentences(ref_text)
    sentences_index[query][ref_doc]=ref_sentences
    f = open(sentence_filename,"w")
    with open(top_docs_file) as file:
        for line in file:
            query_id = line.split("\t")[0]
            top_doc = line.split("\t")[1].split("-")[1].rstrip()
            top_doc_text = next(db.documents.find({"query_id":query_id,"username":top_doc}))["posted_document"]
            top_doc_sentences = retrieve_sentences(top_doc_text)
            sentences_index[query][top_doc]=top_doc_sentences
            for i,top_doc_sentence in enumerate(top_doc_sentences,start=1):
                for j,ref_sentence in enumerate(ref_sentences,start=1):
                    comb_name = top_doc+"_"+str(i)+"_"+str(j)
                    f.write(comb_name+"@@@"+top_doc_sentence.replace("\n","").replace("\r","").rstrip()+"@@@"+ref_sentence.replace("\n","").replace("\r","").rstrip()+"\n")
    f.close()
    return sentence_filename,sentences_index


def get_label_strategies():
    result = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    documents= db.documents.find({})
    for doc in documents:
        if "bot_method" not in doc:
            continue
        key = doc["query_id"]+"-"+doc["username"]
        result[key] = doc["bot_method"]
    return result


def create_sentence_working_set(ref_doc,current_time,sentence_file,query):
    working_set_dir = "sentence_working_set/"+current_time+"/"
    if not os.path.exists(working_set_dir):
        os.makedirs(working_set_dir)
    working_set_filename = working_set_dir+ref_doc+"_"+query+"_workingset_"+current_time
    with open(working_set_filename,"w") as working_set_file:
        with open(sentence_file) as file:
            for i,line in enumerate(file,start=1):
                comb = line.split("@@@")[0]
                working_set_file.write(query.split("_")[0]+" Q0 "+comb+" "+str(i)+" "+str(-i)+" seo\n")
    return working_set_filename



def create_top_docs_per_ref_doc(current_time,ref_doc,query):
    client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
    db = client.asr16
    ref_doc_data = db.documents.find({"username":ref_doc,"query_id":query})
    ref_position = next(ref_doc_data)["position"]
    first = False
    if ref_position==1:
        first=True
    top_docs = db.documents.find({"query_id":query,"position":{"$lt":ref_position}})
    top_docs_dir = "top_docs/"+current_time+"/"

    if not os.path.exists(top_docs_dir):
        os.makedirs(top_docs_dir)
    top_docs_filename = top_docs_dir+ref_doc+"_"+query+"_top_docs_"+current_time
    f = open(top_docs_filename,"w")
    for doc in top_docs:
        username = doc["username"]
        q = query.split("_")[0]
        working_name = q+"-"+username
        f.write(query+"\t"+working_name+"\n")
    f.close()
    return top_docs_filename,first





