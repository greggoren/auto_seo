from pymongo import MongoClient,ASCENDING
from Preprocess.preprocess import retrieve_sentences


ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017


def assign_single_bot(single_bot_method):
    client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
    db = client.asr16
    documents = db.documents.find({"query_id":{"$regex":".*_2"},"position":{"$ne":1},"username":{"$regex":"dummy_doc.*"}}).sort([["query_id",ASCENDING],["position",ASCENDING]])
    # documents = db.documents.find({"query_id":"180_2","position":{"$ne":1},"username":{"$regex":"dummy_doc.*"}}).sort([["query_id",ASCENDING],["position",ASCENDING]])
    seen=[]
    for doc in documents:
        query = doc["query_id"]
        # print(query,doc["position"])
        if query in seen or doc["waterloo"]<60:
        # if query in seen:
            continue
        doc["bot_method"]=single_bot_method
        print(doc["username"],doc["position"],doc["waterloo"])
        seen.append(query)
        db.documents.save(doc)


def pick_startegy(relative_place,method_counts,query_counts):

    # sorted_strategies = sorted(list(method_counts.keys()),key=lambda x:method_counts[x][relative_place])
    sorted_strategies = sorted(list(method_counts.keys()),key=lambda x:(query_counts[x],sum([method_counts[x][r] for r in method_counts[x]]),method_counts[x][relative_place]))
    return sorted_strategies[0]

def assign_three_bots():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    strategies = ["demotion","harmonic","weighted"]
    method_counts = {i:{1:0,2:0,3:0} for i in strategies}
    query_counts = {}
    documents = db.documents.find({"query_id": {"$regex":".*_0"}, "username": {"$regex":"dummy_doc.*"}}).sort([["query_id",ASCENDING],["position",ASCENDING]])
    relative_places = {}
    for doc in documents:
        query = doc["query_id"]
        if query not in query_counts:
            query_counts[query]={i:0 for i in strategies}
        if query not in relative_places:
            relative_places[query]=1
        relative_place = relative_places[query]
        bot_method=pick_startegy(relative_place,method_counts,query_counts[query])
        query_counts[query][bot_method]+=1
        doc["bot_method"] = bot_method
        print(query,doc["username"],doc["position"],bot_method)
        db.documents.save(doc)
        method_counts[bot_method][relative_place]+=1
        relative_places[query]+=1
    print(method_counts)


def fix():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    docs= db.documents.find({"query_id":"051_0","username":"dummy_doc_051_2"})
    doc = next(docs)
    doc["bot_method"]="demotion"
    docs = db.documents.find({"query_id": "051_0", "username": "dummy_doc_051_1"})
    doc = next(docs)
    doc["bot_method"] = "weighted"
assign_single_bot("harmonic")
assign_three_bots()