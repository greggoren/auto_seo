from pymongo import MongoClient,ASCENDING
from Preprocess.preprocess import retrieve_sentences


ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017

def create_former_winners_file(current_time):
    client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
    db = client.asr16
    iterations = db.archive.distinct("iteration")
    sorted_iterations = sorted(iterations)
    start_iter_index = 5
    needed_iterations = sorted_iterations[start_iter_index:]
    past_winners_filename = "past_winners_text_"+current_time
    f = open(past_winners_filename,"w")
    for iteration in needed_iterations:
        documents = db.archive.find({"iteration":iteration,"position":1}).sort("query_id",1)
        for document in documents:
            query = document["query_id"]
            if not query.__contains__("_0") and not query.__contains__("_2"):
                continue
            text = document["text"]
            sentences = retrieve_sentences(text)
            f.write(query + "@@@" + " ".join([a.replace("\n", "") for a in sentences]) + "\n")
    f.close()



def create_top_docs_per_ref_doc(current_time,ref_doc,query):
    client = MongoClient(ASR_MONGO_HOST,ASR_MONGO_PORT)
    db = client.asr16
    ref_doc_data = db.documents.find({"username":ref_doc,"query_id":query})
    ref_position = next(ref_doc_data)["position"]
    top_docs = db.documents.find({"query_id":query,"position":{"$lt":ref_position}})
    top_docs_filename = ref_doc+"_top_docs_"+current_time
    f = open(top_docs_filename,"w")
    for doc in top_docs:
        username = doc["username"]
        q = query.split("_")[0]
        working_name = q+"-"+username
        f.write(q+"\t"+working_name+"\n")
    f.close()





