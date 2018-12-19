import csv
from copy import deepcopy
import numpy as np
from scipy.stats import pearsonr as p
from scipy.stats import spearmanr as s
def read_ds_fe(filename,ident=False):
    result={}
    if ident:
        col = "which_document_has_experienced_manipulation"
    else:
        col="which_sentence_doesnt_belong_to_original_document"
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["_golden"].lower()=="true":
                continue
            id = row["id"]
            if not id.__contains__("ROUND-"):
                continue
            if id not in result:
                result[id]={}
                if ident:
                    gold = row["check_one_gold"].lower()
                    result[id]["golden"]=gold.split("document")[1]
                else:
                    result[id]["golden"] = row["check_one_gold"]
            row_key = len(result[id])
            if ident:
                result[id][row_key]=row[col].split("_")[1]
            else:
                result[id][row_key] = row[col]
    return result


def read_ds_mturk(filename,ident=False):
    result = {}
    with open(filename,encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            id = row["Input.ID"]
            if id not in result:
                result[id] = {}
                if ident:
                    result[id]["golden"] = row["Input.check_one_gold"].lower().split("document")[1]
                else:
                    result[id]["golden"] = row["Input.check_one_gold"]
            row_key = len(result[id])
            result[id][row_key] = {}
            result[id][row_key] = row["Answer.Tag4"]
    return result

def combine_results(fe_res,mturk_res):
    result = deepcopy(mturk_res)
    for id in fe_res:
        if id in mturk_res:
            continue
        result[id]=fe_res[id]
    return result


def create_annotations(results,num):
    stats={}
    stats_ratio={}
    for id in results:
        stats[id]=[]
        for key in results[id]:
            if key == "golden":
                continue
            if results[id][key]==results[id]["golden"]:
                stats[id].append(0)
            else:
                stats[id].append(1)
    for id in stats:
        stats_ratio[id] = np.mean(stats[id])
        if sum(stats[id])>=num:
            stats[id]=1
        else:
            stats[id]=0

    return stats,stats_ratio


def agreements_rate(ident,sentence):
    sum=0
    for id in sentence:
        if sentence[id]==ident[id]:
           sum+=1
    return sum/len(sentence)

def keepagreement(ident,sentence):
    res = {}
    counts ={}
    for id in sentence:
        if sentence[id]==ident[id]:
            res[id]=sentence[id]
            query = id.split("-")[2]
            if query not in counts:
                counts[query]=[]
            if ident[id]==1:
                counts[query].append(1)
            else:
                counts[query].append(0)
    final_res ={}
    for id in res:
        query = id.split("-")[2]
        if query in counts:
            final_res[id] = res[id]
    return final_res,counts

def get_coherent_ratio(results):
    sum=0
    for id in results:
        sum+=results[id]
    return sum/len(results)

def update_dict(old,new):
    for id in new:
        old[id]=new[id]
    return old

def get_correlation_of_ratios(ident,sentence,type="s"):
    sentence_for_corr = []
    ident_for_corr = []
    sorted_ids = sorted(list(ident.keys()),key=lambda x:ident[x])
    for id in sorted_ids:
        ident_for_corr.append(ident[id])
        sentence_for_corr.append(sentence[id])
    if type =="p":
        return p(ident_for_corr,sentence_for_corr)
    else:
        return s(ident_for_corr, sentence_for_corr)

if __name__=="__main__":
    ident_filename_fe = "figure-eight/ident_current.csv"
    ident_filename_mturk = "Mturk/Manipulated_Document_Identification.csv"
    ident_fe=read_ds_fe(ident_filename_fe,True)
    ident_mturk = read_ds_mturk(ident_filename_mturk,True)


    ident_results = combine_results(ident_fe,ident_mturk)
    ident_annotation,ident_ratio = create_annotations(ident_results)

    sentence_filename_fe = "figure-eight/sentence_current.csv"
    sentence_filename_mturk = "Mturk/Sentence_Identification.csv"
    sentence_filename_mturk_new = "Mturk/Sentence_Identification11.csv"
    sentence_fe=read_ds_fe(sentence_filename_fe)
    sentence_mturk = read_ds_mturk(sentence_filename_mturk)
    sentence_mturk_new = read_ds_mturk(sentence_filename_mturk_new)
    sentence_mturk=update_dict(sentence_mturk,sentence_mturk_new)


    sentence_results = combine_results(sentence_fe,sentence_mturk)
    sentence_annotation,sentence_ratio = create_annotations(sentence_results)

    print("Agreement:",agreements_rate(ident_annotation,sentence_annotation))
    print("coherent ratio ident:",get_coherent_ratio(ident_annotation))
    print("coherent ratio sentence:",get_coherent_ratio(sentence_annotation))

    p_corr = get_correlation_of_ratios(ident_ratio,sentence_ratio,"p")
    s_corr = get_correlation_of_ratios(ident_ratio,sentence_ratio)

    print("pearson",p_corr)
    print("spearman",s_corr)