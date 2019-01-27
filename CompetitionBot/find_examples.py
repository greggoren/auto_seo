from pymongo import MongoClient
from CompetitionBot.analyze_positions import ASR_MONGO_PORT,ASR_MONGO_HOST,get_reference_documents

def find(reference_docs):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))
    first = iterations[7]
    second = iterations[8]
    docs = db.archive.find({"iteration":first,"query_id":{"$regex":".*_2"}})
    for doc in docs:
        query = doc["query_id"]
        username = doc["username"]
        if username in reference_docs[query]:
            modified = db.archive.find({"iteration":second,"query_id":query,})




