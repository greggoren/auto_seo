from pymongo import MongoClient
ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017


client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
db = client.asr16
docs = db.documents.find({})
for doc in docs:
    if "bot_method" in doc:
        del doc["bot_method"]
        db.documents.save(doc)