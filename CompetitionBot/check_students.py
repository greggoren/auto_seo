from pymongo import MongoClient
from copy import  deepcopy
ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017
def check():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    stats={}
    iteration = sorted(list(db.archive.distinct("iteration")))[7]
    docs =db.archive.find({"iteration":iteration})
    for doc in docs:
        query = doc["query_id"]
        group = query.split("_")[1]
        if group in ["0","2"]:
            continue
        if group not in stats:
            stats[group]=set()
        if doc["username"].__contains__("dummy_doc"):
            continue
        stats[group].add(doc["username"])
    res={}
    groups=["1","3","4"]
    for group1 in groups:
        for group2 in groups:
            if group1==group2:
                continue
            key = tuple(sorted([group1,group2]))
            if key in res:
                continue
            set1 = deepcopy(stats[group1])
            set2 = deepcopy(stats[group2])
            res[key]= len(set1.intersection(set2))
    print(res)



def get_waterloo_13():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    stats = {}
    iterations = sorted(list(db.archive.distinct("iteration")))[:5]
    for iteration in iterations:
        docs = db.archive.find({"iteration":iteration,"query_id":"013_2","doc_name":{"$regex":"ROUND-"}})


check()