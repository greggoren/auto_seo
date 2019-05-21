from CompetitionBot.create_ds_for_annotations import get_reference_documents
from pymongo import MongoClient
import os
import numpy as np
import csv
from itertools import product

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


def get_average_bot_ranking(reference_docs,group):
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    first_round ={}
    second_round ={}
    for index,iteration in enumerate(iterations):

        results[iteration]=[]
        for query_id in reference_docs:
            query_group = query_id.split("_")[1]
            if query_group!=group:
                continue
            for doc in reference_docs[query_id]:
                document = next(db.archive.find({"iteration": iteration, "username": doc, "query_id": query_id}))
                position = document["position"]
                results[iteration].append(position)
                if index==0:
                    if query_id not in first_round:
                        first_round[query_id]=[]
                    first_round[query_id].append(position)
                elif index ==1:
                    if query_id not in second_round:
                        second_round[query_id]=[]
                    second_round[query_id].append(position)

    for iteration in results:
        results[iteration]= np.mean(results[iteration])
    return results,first_round,second_round

def get_average_rank_of_active_competitors():
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    first={}
    second={}
    for index,iteration in enumerate(iterations):
        results[iteration]=[]
        documents = db.archive.find({"iteration":iteration})

        for document in documents:
            query = document["query_id"]
            group = query.split("_")[1]

            if group not in ["2"]:
                continue
            username = document["username"]
            if username.__contains__("dummy_doc"):
                continue
            results[iteration].append(document["position"])
            if index == 0:
                if query not in first:
                    first[query]=[]
                first[query].append(document["position"])
            elif index == 1:
                if query not in second:
                    second[query]=[]
                second[query].append(document["position"])
    for iteration in results:
        results[iteration]=np.mean(results[iteration])
    return results,first,second



def get_average_rank_of_dummies_and_ks(reference_docs,rel_stats):
    results = {}
    ks={}
    dummy_rel={}
    initial_bot_rel=[]
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for i,iteration in enumerate(iterations):
        results[iteration]=[]
        ks[iteration]=[]
        dummy_rel[iteration]=[]
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
                if i==0:
                    if rel_stats[document["doc_name"]] > 0:
                        initial_bot_rel.append(1)
                    else:
                        initial_bot_rel.append(0)
                continue
            results[iteration].append(document["position"])
            waterloo = document["waterloo"]
            print(username,waterloo)
            if waterloo < 60:
                ks[iteration].append(0)
            else:
                ks[iteration].append(1)
            if rel_stats[document["doc_name"]]>0:
                dummy_rel[iteration].append(1)
            else:
                dummy_rel[iteration].append(0)
            # if ks_tag>0:
            #     ks[iteration][group].append(0)
            # else:
            #     ks[iteration][group].append(1)
    for iteration in results:
        results[iteration]=np.mean(results[iteration])
        ks[iteration]=np.mean(ks[iteration])
        dummy_rel[iteration]=np.mean(dummy_rel[iteration])
    write_rel_for_static(np.mean(initial_bot_rel))
    return results,ks,dummy_rel


def write_rel_for_static(value):
    f = open("static_rel","w")
    for i in range(1,6):
        f.write(str(i)+" "+str(value)+"\n")
    f.close()





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
        if i==0:
            initial_results,first = read_file(dir+"/"+file,method_index,rel)
        elif i==1:
            initial_results, second = read_file(dir + "/" + file, method_index, rel)
        else:
            initial_results, _ = read_file(dir + "/" + file, method_index, rel)
        stats[i]=initial_results
    return stats,first,second

def convert_stats(stats):
    converted_stats={}
    stats_for_perm={}
    for query in stats:
        group = query.split("_")[1]
        if query not in stats_for_perm:
            stats_for_perm[query]=[]

        if group not in converted_stats:
            converted_stats[group]=[]
        for user in stats[query]:
            converted_stats[group].append(stats[query][user])
            stats_for_perm[query].append(stats[query][user])
    converted_stats=np.mean(converted_stats["2"])
    return converted_stats,stats_for_perm



def read_file_rel(filename,rel=False):
    stats={}
    with open(filename,encoding="utf-8") as file:
        ref = "valid"
        if rel:
            ref = "non-relevant"
        reader = csv.DictReader(file)
        for row in reader:
            query = row["query_id"]
            group = query.split("_")[1]
            if group not in ["2"]:
                continue
            user = row["username"]
            annotation = row["this_document_is"].lower()
            label = 0
            if annotation!=ref:
                label = 1
            if query not in stats:
                stats[query]={}
            if user not in stats[query]:
                stats[query][user] = []
            stats[query][user].append(label)
        for query in stats:
            for user in stats[query]:
                if sum(stats[query][user])>=3:
                    stats[query][user]=1
                else:
                    stats[query][user] = 0
    return stats


def read_group_dir_rel(dir,rel=False):
    files = sorted(list(os.listdir(dir)))
    merged_stats={}
    total_result={}
    total_merge={}
    first,second ={},{}
    for i,file in enumerate(files):
        initial_results = read_file_rel(dir+"/"+file,rel)
        merged_stats = merge_stats(merged_stats,initial_results)
        total_merge[i]=merged_stats
        if i==0:
            results,first = convert_stats(merged_stats)
        elif i==1:
            results,second = convert_stats(merged_stats)
        else:
            results,_ = convert_stats(merged_stats)
        total_result[i]=results

    return total_result,total_merge,first,second


def merge_stats(former_stats,stats):
    merged_stats={}
    if not former_stats:
        return stats
    for query in former_stats:

        if query not in stats:
            merged_stats[query] = former_stats[query]
            continue
        merged_stats[query]={}
        for user in former_stats[query]:
            if user not in stats[query]:
                merged_stats[query][user]=former_stats[query][user]
                continue
            merged_stats[query][user]=stats[query][user]
    return merged_stats

def read_file(filename,method_index,rel=False):
    stats={}
    final_stats={}
    perm_stats={}
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
            if annotation!=ref:
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
            if group!="2":
                continue

            if group not in final_stats:
                final_stats[group]=[]
            for user in stats[query]:
                final_stats[group].append(stats[query][user])

                if query not in perm_stats:
                    perm_stats[query] = []
                perm_stats[query].append(stats[query][user])

        final_stats=np.mean(final_stats["2"])
    return final_stats,perm_stats

def read_bot_rel_file(filename):
    stats={}
    first={}
    second={}
    with open(filename,encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iteration = row["iteration"]
            query = row["query_id"]
            if iteration not in stats:
                stats[iteration]={}
            if query not in  stats[iteration]:
                stats[iteration][query]=[]
            label = 1
            if row["this_document_is"].lower()=="relevant":
                label=0
            stats[iteration][query].append(label)
    for iteration in stats:
        for query in stats[iteration]:
            if sum(stats[iteration][query])>=3:
                stats[iteration][query]=0
            else:
                stats[iteration][query] = 1
    iterations = sorted(list(stats.keys()))
    for index,iteration in enumerate(iterations):
        if index==0:
            for q in stats[iteration]:
                first[q] = stats[iteration][q]
        elif index==1:
            for q in stats[iteration]:
                second[q] = stats[iteration][q]
        stats[iteration]=np.mean([stats[iteration][q] for q in stats[iteration]])

    return stats,first,second



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



def get_quality():
    results = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    first,second = {},{}
    for index,iteration in enumerate(iterations):
        results[iteration]=[]
        docs = db.archive.find({"iteration":iteration})
        for doc in docs:
            query = doc["query_id"]
            group = query.split("_")[1]
            if group not in ["2"]:
                continue
            username = doc["username"]
            if username.__contains__("dummy_doc"):
                continue
            waterloo = doc["waterloo"]
            if waterloo>=60:
                results[iteration].append(1)
                if index == 0:
                    if query not in first:
                        first[query]=[]
                    first[query].append(1)
                if index == 1:
                    if query not in second:
                        second[query]=[]
                    second[query].append(1)
            else:
                results[iteration].append(0)
                if index == 0:
                    if query not in first:
                        first[query]=[]
                    first[query].append(0)
                if index == 1:
                    if query not in second:
                        second[query]=[]
                    second[query].append(0)

    for iteration in results:
        results[iteration]= np.mean(results[iteration])
    return results,first,second


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
    iterations = sorted(list(dummies.keys()))
    second_potential = {"Bot":{},"Active":{}}
    for index,iteration in enumerate(iterations):
        dummies_averages[iteration] = []
        active_averages [iteration]= []
        bot_averages [iteration]= []
        for query_id in dummies[iteration]:
            dummy_potentials = [dummies[iteration][query_id][d] for d in dummies[iteration][query_id]]
            active_potentials = [active[iteration][query_id][d] for d in active[iteration][query_id]]
            dummies_averages[iteration].append(np.mean(dummy_potentials))
            active_averages[iteration].append(np.mean(active_potentials))
            if index==1:
                if query_id not in second_potential["Active"]:
                    second_potential["Active"][query_id]=[]
                second_potential["Active"][query_id].append(np.mean(active_potentials))
            if query_id in bots[iteration]:
                bot_potentials = [bots[iteration][query_id] [d] for d in bots[iteration][query_id]]
                bot_averages[iteration].append(np.mean(bot_potentials))
                if index==1:
                    if query_id not in second_potential["Bot"]:
                        second_potential["Bot"][query_id]=[]
                    second_potential["Bot"][query_id].append(np.mean(bot_potentials))

    for iteration in dummies_averages:
        dummies_averages[iteration]=np.mean(dummies_averages[iteration])
        active_averages[iteration]=np.mean(active_averages[iteration])
        bot_averages[iteration]=np.mean(bot_averages[iteration])
    return dummies_averages,active_averages,bot_averages,second_potential


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
    second_promotion = {"Bot": {}, "Active": {}}

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

                elif doc in reference_docs[query_id]:
                    if query_id not in bots[iteration]:
                        bots[iteration][query_id]={}
                    bots[iteration][query_id][doc]=potential
                    overall_promotion[iteration]["Bot"]+=(old_position-new_position)
                    if i==1:
                        second_promotion["Bot"][query_id]=(old_position-new_position)
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
                    if i==1:
                        if query_id not in second_promotion["Active"]:
                            second_promotion["Active"][query_id]=[]
                        second_promotion["Active"][query_id].append(old_position-new_position)
    dummy_averages,active_averages,bot_averages,second_potential = calculate_potential_averages(dummies,active,bots)
    return dummy_averages,active_averages,bot_averages,stayed_winner,overall_promotion,changes_in_ranking_stats,second_potential,second_promotion






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


def get_postitions():
    positions = {}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    for iteration in iterations:
        positions[iteration] = {}
        docs = db.archive.find({"iteration": iteration})

        for doc in docs:
            query_id = doc["query_id"]
            group = query_id.split("_")[1]

            if group != "2":
                continue

            username = doc["username"]
            position = doc["position"]
            if query_id not in positions[iteration]:
                positions[iteration][query_id] = {}
            positions[iteration][query_id][username] = position
    return positions

def get_top_competitor_data(positions,merged_stats):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]
    firsts = {}
    rel_stats={}
    average_positions_data={}
    average_potential_data={}
    raw_position_data={}
    ks={}
    for i,iteration in enumerate(iterations):
        firsts[iteration]={}
        ks[iteration]=[]
        rel_stats[iteration]=[]
        average_positions_data[iteration]=[]
        if i>0:
            average_potential_data[iteration]=[]
            raw_position_data[iteration] = []
        queries = db.archive.distinct("query_id", {"query_id": {"$regex": ".*_2"}})
        for query_id in queries:
            docs = db.archive.find({"iteration":iteration,"query_id":query_id}).sort("position",1)
            for doc in docs:
                print(doc["query_id"],doc["username"], doc["position"])
                if not doc["username"].__contains__("dummy_doc"):
                    rel = merged_stats[i][doc["query_id"]][doc["username"]]
                    rel_stats[iteration].append(rel)
                    firsts[iteration][query_id] = doc["username"]
                    average_positions_data[iteration].append(doc["position"])
                    if i>0:
                        old_position = positions[iterations[i - 1]][query_id][doc["username"]]
                        new_position = positions[iteration][query_id][doc["username"]]
                        if new_position==1 and old_position==1:
                            break
                        if new_position >= old_position:
                            denominator = 5 - old_position
                        else:
                            denominator = old_position - 1
                        if denominator == 0:
                            potential = 0
                        else:
                            potential = (old_position - new_position) / (denominator)
                        overall_promotion = old_position-new_position
                        average_potential_data[iteration].append(potential)
                        raw_position_data[iteration].append(overall_promotion)

                    waterloo = doc["waterloo"]
                    if waterloo>=60:
                        ks[iteration].append(1)
                    else:
                        ks[iteration].append(0)
                    break
    for i,iteration in enumerate(list(average_positions_data)):
        if i>0:
            raw_position_data[iteration]=sum(raw_position_data[iteration])
            average_potential_data[iteration]=np.mean(average_potential_data[iteration])
        average_positions_data[iteration]=np.mean(average_positions_data[iteration])
        ks[iteration]=np.mean(ks[iteration])
        rel_stats[iteration]=np.mean(rel_stats[iteration])
    return raw_position_data,average_potential_data,average_positions_data,firsts,ks,rel_stats


def write_data_file(stats,filename):
    f = open(filename,"w")
    for iteration in stats:
        f.write(str(iteration)+" "+str(stats[iteration])+"\n")
    f.close()

def write_static_ks():
    f = open("static_ks","w")
    for i in range(1,6):
        f.write(str(i)+" 1\n")
    f.close()


def write_raw_promotion_file(stats,filename,group):
    f = open(filename, "w")
    for iteration in stats:
        f.write(iteration + " " + str(stats[iteration][group]) + "\n")
    f.close()






def analyze_ks():
    dummy_ks={}
    active_ks={}
    dummy_ks={}
    dummy_ks={}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = sorted(list(db.archive.distinct("iteration")))[7:]




def permutation_test(sample_a,sample_b):
    real_diff = abs(np.mean(sample_a)-np.mean(sample_b))
    x = list(product([1, -1], repeat=len(sample_a)))
    n_perm = len(x)
    total = 0
    for row in x:
        a_mean=[]
        b_mean=[]
        for index,val in enumerate(row):
            if val>0:
                a_mean.append(sample_a[index])
                b_mean.append(sample_b[index])
            else:
                a_mean.append(sample_b[index])
                b_mean.append(sample_a[index])
        current_diff = abs(np.mean(a_mean)-np.mean(b_mean))
        if current_diff>=real_diff:
            total+=1
    return total/n_perm





def convert_stats_perm(active,bot):
    bot_sample=[]
    active_sample=[]
    for query in bot:
        bot_sample.append(np.mean(bot[query]))
        active_sample.append(np.mean(active[query]))
    return bot_sample,active_sample

def convert_unified_perm(stats):
    bot_sample = []
    active_sample = []
    for query in stats["Bot"]:
        bot_sample.append(np.mean(stats["Bot"][query]))
        active_sample.append(np.mean(stats["Active"][query]))
    return bot_sample,active_sample

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
    average_single_bot_ranking,first_round_bot,second_round_bot = get_average_bot_ranking(reference_docs,"2")
    write_data_file(average_single_bot_ranking,"bot_average")
    # write_table_bots_ranking("2",average_single_bot_ranking,results_dir)
    average_rank_competitrs,first_active,second_active = get_average_rank_of_active_competitors()
    write_data_file(average_rank_competitrs, "active_average")
    # write_competitors_ranking_table(average_rank_competitrs,results_dir)
    ks_stats,first_bot_ks,second_bot_ks=read_group_dir("annotations/",method_index,False)
    write_data_file(ks_stats,"bot_ks")
    # write_quality_annotation_table(ks_stats,results_dir)
    competitrs_quality,quality_active_first,quality_active_second = get_quality()
    write_data_file(competitrs_quality,"active_ks")
    # write_competitors_quality_table(competitrs_quality)
    dummy_averages, active_averages, bot_averages,separate,overall_promotion,changes_in_ranking_stats,second_potential,second_promotion = create_average_promotion_potential(reference_docs)

    # write_potential_tables(dummy_averages,active_averages,bot_averages,results_dir)
    # sep_hist = get_separate_stats(separate,reference_docs)
    # write_separate_table(sep_hist,results_dir)
    # write_tables_hist_ranking_changes(changes_in_ranking_stats)
    # write_overall_changes(overall_promotion)
    # create_query_to_quality_tables(reference_docs,results_dir)
    # print(reference_docs)
    ks_stats = read_annotations("doc_ks_nimrod")
    # print(ks_stats)
    rel_stats = read_annotations("doc_rel_nimrod")
    results,ks,dummy_rel=get_average_rank_of_dummies_and_ks(reference_docs,rel_stats)
    write_data_file(ks,"dummy_ks")
    write_data_file(results,"dummy_average")
    write_data_file(dummy_rel,"dummy_rel")
    positions = get_postitions()
    active_rel_stats, merged_stats,first_active_rel,second_active_rel = read_group_dir_rel("rel_annotations/", True)
    write_data_file(active_rel_stats, "active_rel")
    raw_position_data, average_potential_data, average_positions_data, firsts,top_ks,top_rel=get_top_competitor_data(positions,merged_stats)
    write_data_file(top_ks,"top_ks")
    write_data_file(top_rel,"top_rel")
    write_data_file(raw_position_data,"top_raw")
    write_data_file(average_potential_data,"top_potential")
    write_data_file(average_positions_data,"top_average")
    write_data_file(dummy_averages,"dummy_potential")
    write_data_file(active_averages,"active_potential")
    write_data_file(bot_averages,"bot_potential")
    write_raw_promotion_file(overall_promotion,"bot_raw","Bot")
    write_raw_promotion_file(overall_promotion,"active_raw","Active")
    write_raw_promotion_file(overall_promotion,"dummy_raw","Dummies")
    write_static_ks()
    rel_bot,rel_bot_first,rel_bot_second = read_bot_rel_file("rel_bot.csv")
    write_data_file(rel_bot,"bot_rel")

    ks_first_bot_sample,ks_first_active_sample=convert_stats_perm(quality_active_first,first_bot_ks)
    print("KS_FIRST_PERM_STATS:",permutation_test(ks_first_bot_sample,ks_first_active_sample))

    ks_second_bot_sample, ks_second_active_sample = convert_stats_perm(quality_active_second, second_bot_ks)
    print("KS_SECOND_PERM_STATS:", permutation_test(ks_second_bot_sample, ks_second_active_sample))

    rel_bot_first_sample,rel_active_first_sample=convert_stats_perm(first_active_rel,rel_bot_first)
    print("REL_FIRST_PERM_STATS:",permutation_test(rel_bot_first_sample,rel_active_first_sample))

    rel_bot_second_sample, rel_active_second_sample = convert_stats_perm(second_active_rel, rel_bot_second)
    print("REL_SECOND_PERM_STATS:", permutation_test(rel_bot_second_sample, rel_active_second_sample))

    average_first_promotion_bot_sample,average_first_promotion_active_sample = convert_stats_perm(first_active,first_round_bot)
    print("FIRST_AVERAGE_PROMOTION_PERM_STATS:", permutation_test(average_first_promotion_bot_sample, average_first_promotion_active_sample))

    average_second_promotion_bot_sample, average_second_promotion_active_sample = convert_stats_perm(second_active, second_round_bot)
    print("SECOND_AVERAGE_PROMOTION_PERM_STATS:",
          permutation_test(average_second_promotion_bot_sample, average_second_promotion_active_sample))

    raw_bot,raw_active=convert_unified_perm(second_promotion)
    print("SECOND_RAW_PROMOTION_PERM_STATS:",
          permutation_test(raw_bot, raw_active))

    potential_bot,potential_active=convert_unified_perm(second_potential)
    print("SECOND_POTENTIAL_PROMOTION_PERM_STATS:",
          permutation_test(potential_bot, potential_active))