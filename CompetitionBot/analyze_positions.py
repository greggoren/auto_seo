from CompetitionBot.create_ds_for_annotations import get_reference_documents
from pymongo import MongoClient
import os
import numpy as np
import csv

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
                else:
                    if last[query_id + "_" + doc] not in hist[iteration]:
                        hist[iteration][last[query_id + "_" + doc]] = {}
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
                document = next(db.archive.find({"iteration":iteration,"username":doc,"query_id":query_id}))
                position = document["position"]
                method =document["bot_method"]
                if method not in hist[iteration]:
                    hist[iteration][method]={}
                if query_id+"_"+doc not in last:
                    last[query_id+"_"+doc]=position
                else:
                    if last[query_id + "_" + doc] not in hist[iteration][method]:
                        hist[iteration][method][last[query_id + "_" + doc]] = {}
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
            f.write("\\hline\n")
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
                f.write("\\hline\n")
            f.write("\\end{tabular}\n")


def get_average_bot_ranking(reference_docs,method_index,group):
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        results[iteration]={}
        for query_id in reference_docs:
            query_group = query_id.split("_")[1]
            if query_group!=group:
                continue
            for doc in reference_docs[query_id]:
                document = next(db.archive.find({"iteration": iteration, "username": doc, "query_id": query_id}))
                bot_method = method_index[query_id+"_"+doc]#document["bot_method"]
                position = document["position"]
                if bot_method not in results[iteration]:
                    results[iteration][bot_method]=[]
                results[iteration][bot_method].append(position)
    for iteration in results:
        for bot_method in results[iteration]:
            results[iteration][bot_method]= np.mean(results[iteration][bot_method])
    return results

def get_average_rank_of_active_competitors():
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        results[iteration]={}
        documents = db.archive.find({"iteration":iteration})

        for document in documents:
            query = document["query_id"]
            group = query.split("_")[1]

            if group not in ["0","2"]:
                continue
            username = document["username"]
            if username.__contains__("dummy_doc"):
                continue
            if group not in results[iteration]:
                results[iteration][group]=[]
            results[iteration][group].append(document["position"])
    for iteration in results:
        for group in results[iteration]:
            results[iteration][group]=np.mean(results[iteration][group])
    return results


def write_table_bots_ranking(group,results,results_dir):
    f = open(results_dir+"bots_ranking_"+group+".tex","w")
    cols = "c|" * (len(results) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{"+cols+"} \n")
    f.write("\\hline\n")
    f.write("Method & "+" & ".join([str(i+1) for i in range(len(results))])+" \\\\ \n")
    f.write("\\hline\n")
    bot_methods  = ["demotion","harmonic","weighted"]
    for method in bot_methods:
        if group=="2" and method!="harmonic":
            continue
        f.write(method+" & "+" & ".join([str(round(results[iteration][method],3)) for iteration in sorted(list(results.keys()))])+ "\\\\ \n")
        f.write("\\hline \n")
    f.write("\\end{tabular}\n")
    f.close()


def write_competitors_ranking_table(results,results_dir):
    f = open(results_dir+"competitors_ranking.tex", "w")
    cols = "c|" * (len(results) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{"+cols+"} \n")
    f.write("\\hline\n")
    f.write("Test group & " + " & ".join([str(i + 1) for i in range(len(results))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("Single bot & "+" & ".join([str(round(results[i]["2"],3)) for i in sorted(list(results.keys()))])+" \\\\ \n")
    f.write("Multiple bots & "+" & ".join([str(round(results[i]["0"],3)) for i in sorted(list(results.keys()))])+" \\\\ \n")
    f.write("\\end{tabular}\n")
    f.close()

def read_group_dir(dir,method_index,rel=False):
    files = sorted(list(os.listdir(dir)))
    stats={}
    for i,file in enumerate(files):
        initial_results = read_file(dir+"/"+file,method_index,rel)
        stats[i]=initial_results
    return stats

def read_file(filename,method_index,rel=False):
    stats={}
    final_stats={}
    with open(filename,encoding="utf-8") as file:
        ref = "valid"
        if rel:
            ref = "non-relevant"
        reader = csv.DictReader(file)
        for row in reader:
            query = row["query_id"]
            user = row["username"]
            annotation = row["this_document_is"].lower()
            label = 0
            if annotation==ref:
                label = 1
            if query not in stats:
                stats[query]={}
            if user not in stats[query]:
                stats[query][user] = []
            stats[query][user].append(label)
        for query in stats:
            for user in stats[query]:
                if sum(stats[query][user]) >= 3:
                    stats[query][user]=1
                else:
                    stats[query][user] = 0
        for query in stats:
            group = query.split("_")[1]
            if group not in final_stats:
                final_stats[group]={}
            for user in stats[query]:
                method = method_index[query+"_"+user]
                if method not in final_stats[group]:
                    final_stats[group][method]=[]
                final_stats[group][method].append(stats[query][user])
        print(final_stats)
        for group in final_stats:
            for method in final_stats[group]:
                final_stats[group][method]=np.mean(final_stats[group][method])
    print(final_stats)
    return final_stats

def get_method_index():
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    docs = db.documents.find({})
    index = {}
    for doc in docs:
        if "bot_method" in doc:
            index[doc["query_id"]+"_"+doc["username"]]=doc["bot_method"]
    return index

def write_quality_annotation_table(results,results_dir):
    f = open(results_dir + "bots_quality_by_epoch.tex", "w")
    cols = "c|"*(len(results)+1)
    cols="|"+cols
    f.write("\\begin{tabular}{"+cols+"} \n")
    f.write("\\hline\n")
    f.write("Test group & " + " & ".join([str(i + 2) for i in range(len(results))]) + " \\\\ \n")
    f.write("Single bot - harmonic & "+" & ".join([str(round(results[i]["2"]["harmonic"],3)) for i in sorted(list(results.keys()))])+" \\\\ \n")
    f.write("\\hline\n")
    methods = ["demotion", "harmonic", "weighted"]
    for method in methods:
        f.write("Multiple bots - "+method+" & "+" & ".join([str(round(results[i]["0"][method],3)) for i in sorted(list(results.keys()))])+" \\\\ \n")
        f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.close()

if __name__=="__main__":
    results_dir = "tex_tables/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    reference_docs = get_reference_documents()
    # hist_single = get_addition_histogram_single_bot(reference_docs)
    # create_table_single_bot(hist_single,results_dir)
    # hist_multiple = get_addition_histogram_multiple_bots(reference_docs)
    # create_table_multiple_bots(hist_multiple,results_dir)
    # method_index = get_method_index()
    # average_multiple_bot_rankings = get_average_bot_ranking(reference_docs,method_index,"0")
    # write_table_bots_ranking("0",average_multiple_bot_rankings,results_dir)
    # average_single_bot_ranking = get_average_bot_ranking(reference_docs,method_index,"2")
    # write_table_bots_ranking("2",average_single_bot_ranking,results_dir)
    # average_rank_competitrs = get_average_rank_of_active_competitors()
    # write_competitors_ranking_table(average_rank_competitrs,results_dir)
    ks_stats=read_group_dir("annotations/",method_index,False)
    write_quality_annotation_table(ks_stats,results_dir)