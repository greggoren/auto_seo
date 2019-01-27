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



def get_average_rank_of_dummies_and_ks(reference_docs,ks_stats,rel_stats):
    results = {}
    ks={}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        results[iteration]={}
        ks[iteration]={}
        documents = db.archive.find({"iteration":iteration,"query_id":{"$regex":".*_2"}})
        for document in documents:
            query = document["query_id"]
            group = query.split("_")[1]
            if group not in ["2",]:
                continue
            username = document["username"]
            if not username.__contains__("dummy_doc"):
                continue
            if username in reference_docs[query]:
                continue
            if group not in results[iteration]:
                results[iteration][group]=[]
                ks[iteration][group]=[]
            results[iteration][group].append(document["position"])
            doc_name = document["doc_name"]
            ks_tag = ks_stats[doc_name]
            waterloo = document["waterloo"]
            print(username,waterloo)
            if waterloo < 60:
                ks[iteration][group].append(0)
            else:
                ks[iteration][group].append(1)
            # if ks_tag>0:
            #     ks[iteration][group].append(0)
            # else:
            #     ks[iteration][group].append(1)
    for iteration in results:
        for group in results[iteration]:
            results[iteration][group]=np.mean(results[iteration][group])
            ks[iteration][group]=np.mean(ks[iteration][group])

    return results,ks




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


def get_competitors_quality():
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        results[iteration]={}
        docs = db.archive.find({"iteration":iteration})
        for doc in docs:
            query = doc["query_id"]
            group = query.split("_")[1]
            if group not in ["0","2"]:
                continue
            username = doc["username"]
            if username.__contains__("dummy_doc"):
                continue
            if group not in results[iteration]:
                results[iteration][group]=[]
            waterloo = doc["waterloo"]
            if waterloo>=60:
                results[iteration][group].append(1)
            else:
                results[iteration][group].append(0)
    for iteration in results:
        for group in results[iteration]:
            results[iteration][group]= np.mean(results[iteration][group])
    return results


def write_competitors_quality_table(results):
    f = open(results_dir + "competitors_quality_by_epoch.tex", "w")
    cols = "c|" * (len(results) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{" + cols + "} \n")
    f.write("\\hline\n")
    f.write("Population & " + " & ".join([str(i + 2) for i in range(len(results))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("Active competitors - single bot & "+" & ".join([str(round(results[i]["2"],3)) for i in sorted(list(results.keys()))])+" \\\\ \n")
    f.write("\\hline\n")
    f.write("Active competitors - multiple bots & "+" & ".join([str(round(results[i]["0"],3)) for i in sorted(list(results.keys()))])+" \\\\ \n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.close()

def create_bot_ranking_to_quality_tables(quality_results_bots,quality_results_dummy_docs):
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        pass


def calculate_potential_averages(dummies,active,bots):
    dummies_averages = {}
    active_averages = {}
    bot_averages = {}
    for iteration in dummies:
        dummies_averages[iteration] = []
        active_averages [iteration]= []
        bot_averages [iteration]= []
        for query_id in dummies[iteration]:
            dummy_potentials = [dummies[iteration][query_id][d] for d in dummies[iteration][query_id]]
            active_potentials = [active[iteration][query_id][d] for d in active[iteration][query_id]]
            dummies_averages[iteration].append(np.mean(dummy_potentials))
            active_averages[iteration].append(np.mean(active_potentials))
            if query_id in bots[iteration]:
                bot_potentials = [bots[iteration][query_id] [d] for d in bots[iteration][query_id]]
                bot_averages[iteration].append(np.mean(bot_potentials))

    for iteration in dummies_averages:
        dummies_averages[iteration]=np.mean(dummies_averages[iteration])
        active_averages[iteration]=np.mean(active_averages[iteration])
        bot_averages[iteration]=np.mean(bot_averages[iteration])
    return dummies_averages,active_averages,bot_averages


def populate_correct_dictionary(stats,new_rank,old_rank):
    if new_rank==old_rank:
        stats["same"]+=1
    elif new_rank>old_rank:
        stats["demoted"]+=1
    else:
        stats["promoted"] += 1
    return stats

def calculate_promotion_potential(reference_docs,positions):
    active = {}
    bots = {}
    dummies = {}
    stayed_winner={}
    overall_promotion = {}
    iterations = sorted(list(positions.keys()))
    changes_in_ranking_stats = {}
    for i,iteration in enumerate(iterations):
        if i == 0:
            continue
        bots[iteration]={}
        dummies[iteration]={}
        active[iteration]={}
        stayed_winner[iteration]={}
        overall_promotion[iteration]={"Bot":0,"Active":0,"Dummies":0}
        changes_in_ranking_stats[iteration]={"same":0,"promoted":0,"demoted":0}
        for query_id in positions[iteration]:
            number_of_competitors = len(positions[iteration][query_id])

            for doc in positions[iteration][query_id]:
                old_position = positions[iterations[i-1]][query_id][doc]
                new_position = positions[iteration][query_id][doc]


                if new_position>=old_position:
                    denominator = number_of_competitors-old_position
                else:
                    denominator= old_position-1
                if denominator==0:
                    potential=0
                else:
                    potential = (old_position-new_position)/(denominator)
                if new_position==1 and old_position==1:
                    stayed_winner[iteration][query_id]=doc

                if doc in reference_docs[query_id]:
                    if query_id not in bots[iteration]:
                        bots[iteration][query_id]={}
                    bots[iteration][query_id][doc]=potential
                    overall_promotion[iteration]["Bot"]+=(old_position-new_position)
                    changes_in_ranking_stats[iteration] = populate_correct_dictionary(
                        changes_in_ranking_stats[iteration], new_position, old_position)
                elif doc.__contains__("dummy_doc"):
                    if query_id not in dummies[iteration]:
                        dummies[iteration][query_id]={}
                    dummies[iteration][query_id][doc]=potential
                    overall_promotion[iteration]["Dummies"] += (old_position - new_position)
                else:
                    if query_id not in active[iteration]:
                        active[iteration][query_id]={}
                    active[iteration][query_id][doc]=potential
                    overall_promotion[iteration]["Active"] += (old_position - new_position)
    dummy_averages,active_averages,bot_averages = calculate_potential_averages(dummies,active,bots)
    return dummy_averages,active_averages,bot_averages,stayed_winner,overall_promotion,changes_in_ranking_stats






def create_average_promotion_potential(reference_docs):
    positions = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        positions[iteration] = {}
        docs = db.archive.find({"iteration":iteration})

        for doc in docs:
            query_id = doc["query_id"]
            group = query_id.split("_")[1]

            if group!="2":
                continue

            username = doc["username"]
            position = doc["position"]
            if query_id not in positions[iteration]:
                positions[iteration][query_id]={}
            positions[iteration][query_id][username]=position
    return calculate_promotion_potential(reference_docs,positions)

def get_separate_stats(separate,reference_docs):
    hist = {}
    for iteration in separate:
        hist[iteration]={}
        for query in separate[iteration]:
            doc = separate[iteration][query]
            if doc in reference_docs[query]:
                if "Bots" not in hist[iteration]:
                    hist[iteration]["Bots"]=0
                hist[iteration]["Bots"]+=1
            elif doc.__contains__("dummy_doc"):
                if "Dummies" not in hist[iteration]:
                    hist[iteration]["Dummies"]=0
                hist[iteration]["Dummies"]+=1
            else:
                if "Active" not in hist[iteration]:
                    hist[iteration]["Active"]=0
                hist[iteration]["Active"]+=1
    return hist

def write_separate_table(separate_hist,results_dir):
    f = open(results_dir + "single_bot_separate_analysis_tables.tex", "w")
    cols = "c|" * (len(separate_hist) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{" + cols + "}\n")
    f.write("\\hline \n")
    f.write("Group & " + " & ".join([str(i + 2) for i in range(len(separate_hist))]) + " \\\\ \n")
    f.write("Bots & "+" & ".join([str(separate_hist[i].get("Bots","0")) for i in sorted(list(separate_hist.keys())) ])+" \\\\ \n")
    f.write("\\hline \n")
    f.write("Active & "+" & ".join([str(separate_hist[i].get("Active","0")) for i in sorted(list(separate_hist.keys()))])+" \\\\ \n")
    f.write("\\hline \n")
    f.write("Dummies & "+" & ".join([str(separate_hist[i].get("Dummies","0")) for i in sorted(list(separate_hist.keys()))])+" \\\\ \n")
    f.write("\\hline \n")
    f.write("\\end{tabular}\n")
    f.close()




def write_tables_hist_ranking_changes(changes):
    f = open(results_dir + "single_bot_changes_in_ranking.tex", "w")
    cols = "c|" * (len(changes) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{" + cols + "}\n")
    f.write("\\hline \n")
    f.write("Status & " + " & ".join([str(i + 2) for i in range(len(changes))]) + " \\\\ \n")
    f.write("\\hline \n")
    f.write("Same ranking & "+" & ".join([str(changes[i]["same"]) for i in sorted(list(changes.keys()))])+" \\\\ \n")
    f.write("\\hline \n")
    f.write("Demotion & "+" & ".join([str(changes[i]["demoted"]) for i in sorted(list(changes.keys()))])+" \\\\ \n")
    f.write("\\hline \n")
    f.write("Promotion & "+" & ".join([str(changes[i]["promoted"]) for i in sorted(list(changes.keys()))])+" \\\\ \n")
    f.write("\\hline \n")
    f.write("\\end{tabular}\n")
    f.close()


def write_overall_changes(overall):
    f = open(results_dir + "single_bot_overall_change_tables.tex", "w")
    cols = "c|" * (len(overall) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{" + cols + "}\n")
    f.write("\\hline\n")
    f.write("Group & " + " & ".join([str(i + 2) for i in range(len(overall))]) + " \\\\ \n")
    f.write("\\hline \n")
    f.write("Bots & " + " & ".join(
        [str(overall[i].get("Bot", "0")) for i in sorted(list(overall.keys()))]) + " \\\\ \n")
    f.write("\\hline \n")
    f.write("Active & " + " & ".join(
        [str(overall[i].get("Active", "0")) for i in sorted(list(overall.keys()))]) + " \\\\ \n")
    f.write("\\hline \n")
    f.write("Dummies & " + " & ".join(
        [str(overall[i].get("Dummies", "0")) for i in sorted(list(overall.keys()))]) + " \\\\ \n")
    f.write("\\hline \n")
    f.write("\\end{tabular}\n")
    f.close()

def write_potential_tables(dummies,active,bots,results_dir):
    f = open(results_dir+"single_bot_potential_tables.tex","w")
    cols = "c|"*(len(dummies)+1)
    cols = "|"+cols
    f.write("\\begin{tabular}{"+cols+"}\n")
    f.write("\\hline\n")
    f.write("Group & "+" & ".join([str(i+2) for i in range(len(dummies))])+" \\\\ \n")
    f.write("\\hline\n")
    f.write("Bots & "+" & ".join([str(round(bots[i],3)) for i in sorted(list(bots.keys()))])+"\\\\ \n")
    f.write("\\hline\n")
    f.write("Active & "+" & ".join([str(round(active[i],3)) for i in sorted(list(active.keys()))])+" \\\\ \n")
    f.write("\\hline\n")
    f.write("Dummies & "+" & ".join([str(round(dummies[i],3)) for i in sorted(list(dummies.keys()))])+" \\\\ \n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.close()


def write_query_to_quality_table(query,watreloo_stats,positions,results_dir):
    f = open(results_dir+query+"_rank_quality.tex","w")
    cols = "c|" * (len(watreloo_stats) + 1)
    cols = "|" + cols
    f.write("\\begin{tabular}{" + cols + "}\n")
    f.write("\\hline\n")
    f.write("Rank & " + " & ".join([str(i + 1) for i in range(len(watreloo_stats))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("1 & " + " & ".join([str(positions[i][1].replace("_"," "))+":"+str(watreloo_stats[i][positions[i][1]]) for i in sorted(list(watreloo_stats.keys()))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("2 & " + " & ".join([str(positions[i][2].replace("_"," "))+":"+str(watreloo_stats[i][positions[i][2]]) for i in sorted(list(watreloo_stats.keys()))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("3 & " + " & ".join([str(positions[i][3].replace("_"," "))+":"+str(watreloo_stats[i][positions[i][3]]) for i in sorted(list(watreloo_stats.keys()))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("4 & " + " & ".join([str(positions[i][4].replace("_"," "))+":"+str(watreloo_stats[i][positions[i][4]]) for i in sorted(list(watreloo_stats.keys()))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("5 & " + " & ".join([str(positions[i][5].replace("_"," "))+":"+str(watreloo_stats[i][positions[i][5]]) for i in sorted(list(watreloo_stats.keys()))]) + " \\\\ \n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.close()


def create_query_to_quality_tables(reference_docs,results_dir):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    queries = db.archive.distinct("query_id",{"query_id":{"$regex":".*_2"}})
    for query in queries:
        query_stats={}
        waterloo_stats={}
        for i, iteration in enumerate(iterations):
            query_stats[iteration]={}
            waterloo_stats[iteration]={}
            sorted_docs = db.archive.find({"iteration":iteration,"query_id":query})
            for doc in sorted_docs:
                user = doc["username"]
                position = doc["position"]
                if user in reference_docs[query]:
                    username = "BOT"
                else:
                    username = user
                query_stats[iteration][position]=username
                waterloo_stats[iteration][username] = doc["waterloo"]

        write_query_to_quality_table(query.split("_")[0],waterloo_stats,query_stats,results_dir)


def read_annotations(filename):
    stats={}
    with open(filename) as file:
        for line in file:
            doc = line.split()[2]
            tag = int(line.split()[3].rstrip())
            stats[doc]=tag
    return stats




if __name__=="__main__":
    results_dir = "tex_tables/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    reference_docs = get_reference_documents()
    # hist_single = get_addition_histogram_single_bot(reference_docs)
    # create_table_single_bot(hist_single,results_dir)
    # hist_multiple = get_addition_histogram_multiple_bots(reference_docs)
    # create_table_multiple_bots(hist_multiple,results_dir)
    method_index = get_method_index()
    # average_multiple_bot_rankings = get_average_bot_ranking(reference_docs,method_index,"0")
    # write_table_bots_ranking("0",average_multiple_bot_rankings,results_dir)
    # average_single_bot_ranking = get_average_bot_ranking(reference_docs,method_index,"2")
    # write_table_bots_ranking("2",average_single_bot_ranking,results_dir)
    # average_rank_competitrs = get_average_rank_of_active_competitors()
    # write_competitors_ranking_table(average_rank_competitrs,results_dir)
    # ks_stats=read_group_dir("annotations/",method_index,False)
    # write_quality_annotation_table(ks_stats,results_dir)
    # competitrs_quality = get_competitors_quality()
    # write_competitors_quality_table(competitrs_quality)
    # dummy_averages, active_averages, bot_averages,separate,overall_promotion,changes_in_ranking_stats = create_average_promotion_potential(reference_docs)
    # write_potential_tables(dummy_averages,active_averages,bot_averages,results_dir)
    # sep_hist = get_separate_stats(separate,reference_docs)
    # write_separate_table(sep_hist,results_dir)
    # write_tables_hist_ranking_changes(changes_in_ranking_stats)
    # write_overall_changes(overall_promotion)
    # create_query_to_quality_tables(reference_docs,results_dir)
    # print(reference_docs)
    ks_stats = read_annotations("doc_ks_nimrod")
    print(ks_stats)
    rel_stats = read_annotations("doc_rel_nimrod")
    results,ks=get_average_rank_of_dummies_and_ks(reference_docs,ks_stats,rel_stats)
    #
    print([(i,results[i]["2"]) for i in sorted(list(results.keys()))])
    print([(i,ks[i]["2"]) for i in sorted(list(ks.keys()))])