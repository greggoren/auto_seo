from CompetitionBot.create_ds_for_annotations import get_reference_documents
from pymongo import MongoClient
import os
ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017


def get_addition_histogram_single_bot(reference_docs):
    hist = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    last = {}
    for iteration in iterations:
        hist[iteration]={}
        for query_id in reference_docs:
            if query_id.__contains__("_0"):
                continue
            for doc in reference_docs[query_id]:
                position = next(db.archive.find({"iteration":iteration,"username":doc,"query_id":query_id}))["position"]

                if query_id+"_"+doc not in last:
                    last[query_id+"_"+doc]=position
                    continue
                else:
                    hist[iteration][last[query_id + "_" + doc]]={}
                    if position not in hist[iteration][last[query_id+"_"+doc]]:
                        hist[iteration][last[query_id + "_" + doc]][position]=0
                    hist[iteration][last[query_id + "_" + doc]][position]+=1
                    last[query_id+"_"+doc]=position
    return hist


def get_addition_histogram_multiple_bots(reference_docs):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    last = {}
    hist={}
    for iteration in iterations:
        hist[iteration]={}
        for query_id in reference_docs:
            if query_id.__contains__("_2"):
                continue
            for doc in reference_docs[query_id]:
                doc = next(db.archive.find({"iteration":iteration,"username":doc,"query_id":query_id}))
                position = doc["position"]
                method =doc["bot_method"]
                if method not in hist:
                    hist[iteration][method]={}
                if not last:
                    last[query_id+"_"+doc]=position
                    continue
                else:
                    hist[iteration][method][last[query_id + "_" + doc]]={}
                    if position not in hist[iteration][method][last[query_id+"_"+doc]]:
                        hist[iteration][method][last[query_id + "_" + doc]][position]=0
                    hist[iteration][method][last[query_id + "_" + doc]][position]+=1
                    last[query_id+"_"+doc]=position
    return hist


def create_table_single_bot(hist,results_dir):
    iterations = sorted(list(hist.keys()))
    for iter_num,iteration in enumerate(iterations,start=1):
        f = open(results_dir+"single_bot_iteration"+str(iter_num)+".tex","w")
        f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
        f.write("\\hline\n")
        f.write("Places & "+" & ".join([str(i) for i in range(1,6)])+"\\\\ \n")
        for i in range(1,6):
            line = str(i)+" & "
            vals = []
            for j in range(1,6):

                if i not in hist[iteration] or j not in hist[iteration][i]:
                    transition_value = "0"
                else:
                    transition_value=str(hist[iteration][i][j])
                vals.append(transition_value)
            line+=" & ".join(vals)+"\\\\ \n"
            f.write(line)
            f.write("\\hline")
        f.write("\\end{tabular}\n")

def create_table_multiple_bots(hist,results_dir):
    iterations = sorted(list(hist.keys()))
    for iter_num,iteration in enumerate(iterations,start=1):
        for method in hist[iteration]:
            f = open(results_dir+method+"_multiple_bots_iteration"+str(iter_num)+".tex","w")
            f.write("\\begin{tabular}{|c|c|c|c|c|c|}\n")
            f.write("\\hline\n")
            f.write("Places & "+" & ".join([str(i) for i in range(1,6)])+"\\\\ \n")
            for i in range(1,6):
                line = str(i)+" & "
                vals = []
                for j in range(1,6):

                    if i not in hist[iteration][method] or j not in hist[iteration][method][i]:
                        transition_value = "0"
                    else:
                        transition_value=str(hist[iteration][method][i][j])
                    vals.append(transition_value)
                line+=" & ".join(vals)+"\\\\ \n"
                f.write(line)
                f.write("\\hline")
            f.write("\\end{tabular}\n")


if __name__=="__main__":
    results_dir = "tex_tables/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    reference_docs = get_reference_documents()
    hist_single = get_addition_histogram_single_bot(reference_docs)
    create_table_single_bot(hist_single)
    hist_multiple = get_addition_histogram_multiple_bots(reference_docs)
    create_table_multiple_bots(hist_multiple)