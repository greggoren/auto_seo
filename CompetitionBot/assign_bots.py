from pymongo import MongoClient,ASCENDING
from Preprocess.preprocess import retrieve_sentences


ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017


def assign_single_bot(single_bot_method):
    client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
    db = client.asr16
    documents = db.documents.find({"query_id":{"$regex":".*_2"},"position":{"$ne":1},"username":{"$regex":"dummy_doc.*"}}).sort([["query_id",ASCENDING],["position",ASCENDING]])
    seen=[]
    for doc in documents:
        query = doc["query_id"]
        # print(query,doc["position"])
        if query in seen or doc["waterloo"]<60:
            continue
        doc["bot_method"]=single_bot_method
        print(doc["username"],doc["position"],doc["waterloo"])
        seen.append(query)
        db.documents.save(doc)


def pick_startegy(relative_place,method_counts):
    sorted_strategies = sorted(list(method_counts.keys()),key=lambda x:method_counts[x][relative_place])
    return sorted_strategies[0]

def assign_three_bots():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    strategies = ["demotion","harmonic","weighted"]
    method_counts = {i:{1:0,2:0,3:0} for i in strategies}
    documents = db.documents.find({"query_id": {"$regex":".*_0"}, "username": {"$regex":"dummy_doc.*"}}).sort([["query_id",ASCENDING],["position",ASCENDING]])
    relative_places = {}
    for doc in documents:
        query = doc["query_id"]
        if query not in relative_places:
            relative_places[query]=1
        relative_place = relative_places[query]
        bot_method=pick_startegy(relative_place,method_counts)
        doc["bot_method"] = bot_method
        db.documents.save(doc)
        method_counts[bot_method][relative_place]+=1
        relative_places[query]+=1
    print(method_counts)


assign_single_bot("weighted")
assign_three_bots()