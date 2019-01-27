from random import shuffle,seed
import csv
import numpy as np
from scipy.stats import pearsonr,spearmanr,kendalltau
seed(9001)



def read_coherency_file(filename):
    stats = {}
    ds_stats = {}
    coherencey_stats= {}
    with open(filename) as file:
        for line in file:
            label = float(line.split()[0])*5/4
            if label < 1:
                bucket = 0
            elif label < 2:
                bucket = 1
            elif label < 3:
                bucket = 2
            elif label < 4:
                bucket = 3
            elif label < 5:
                bucket = 4
            else:
                bucket = 5
            query = line.split()[1].split(":")[1]
            if query not in ds_stats:
                ds_stats[query]=0
            ds_stats[query]+=1
            key = query[3:]
            pair = line.split(" # ")[1].rstrip()
            if bucket not in stats:
                stats[bucket]=[]
            stats[bucket].append(pair+key)
            coherencey_stats[pair+key]=label
    values = [ds_stats[i] for i in ds_stats]
    return stats,coherencey_stats


def sample_pairs_uniformly(pairs):
    sampled = {}
    for bucket in pairs:
        pairs_in_bucket = pairs[bucket]
        shuffle(pairs_in_bucket)
        sampled[bucket]=pairs_in_bucket[:7]
    return sampled


def read_documents():
    rounds = ["04","06"]
    suffix = ["last","2"]
    key_index={"last":"5","2":"2"}
    texts={}
    for r in rounds:
        for s in suffix:
            filename = "comparison_"+r+"_"+s+".csv"
            with open(filename,encoding="utf-8") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    id = row["ID"]
                    key = id+str(int(r))+key_index[s]
                    text = row[row["check_one_gold"].lower()]
                    texts[key]=text
    return texts


def create_ds_for_annotations(texts,sampled):
    with open("quality_offline_bot_ds.csv","w",encoding="utf-8",newline='') as f:
        writer = csv.DictWriter(f,fieldnames=["username","current_document"])
        writer.writeheader()
        for bucket in sampled:
            samples = sampled[bucket]
            for pair in samples:
                row={}
                row["username"]=pair
                row["current_document"] = texts[pair]
                writer.writerow(row)

def read_tags(filename):
    with open(filename) as file:
        stats={}
        for line in file:
            id = line.split()[0]
            tag = int(line.split()[1].rstrip())
            stats[id]=tag
        return stats

def read_ks_file(filename):
    f = open(filename,encoding="utf-8")
    stats={}
    reader = csv.DictReader(f)
    for row in reader:
        doc_id=row["username"]
        if doc_id not in stats:
            stats[doc_id]=0
        if row["this_document_is"].lower()=="valid":

            stats[doc_id]+=1

    return stats

ks_stats=read_ks_file("ks_offline_bot.csv")
# print(ks_stats)

def write_tables(ks,coh,ident,sentence):
    f = open("quality_to_coherency_correlation.tex","w")
    f.write("\\begin{tabular}{|c|c|c|c|}\n")
    f.write("\\hline\n")
    f.write("Data-set & Pearson & Spearman & Kendall-$\\tau$ \\\\ \n")
    f.write("\\hline\n")
    pearson = pearsonr(ks, coh)
    spearman = spearmanr(ks, coh)
    kendall= kendalltau(ks, coh)
    f.write("Combined & $"+str(round(pearson[0],3))+"$ ($"+str(round(pearson[1],3))+"$) & $"+str(round(spearman[0],3))+"$ ($"+str(round(spearman[1],3))+"$) & $"+str(round(kendall[0],3))+"$ ($"+str(round(kendall[1],3))+"$) \\\\ \n")
    f.write("\\hline\n")
    pearson = pearsonr(ks, ident)
    spearman = spearmanr(ks, ident)
    kendall = kendalltau(ks, ident)
    f.write("Document identification & $" + str(round(pearson[0], 3)) + "$ ($" + str(round(pearson[1], 3)) + "$) & $" + str(
        round(spearman[0], 3)) + "$ ($" + str(round(spearman[1], 3)) + "$) & $" + str(
        round(kendall[0], 3)) + "$ ($" + str(round(kendall[1], 3)) + "$) \\\\ \n")
    f.write("\\hline\n")
    pearson = pearsonr(ks, sentence)
    spearman = spearmanr(ks, sentence)
    kendall = kendalltau(ks, sentence)
    f.write(
        "Sentence identification & $" + str(round(pearson[0], 3)) + "$ ($" + str(round(pearson[1], 3)) + "$) & $" + str(
            round(spearman[0], 3)) + "$ ($" + str(round(spearman[1], 3)) + "$) & $" + str(
            round(kendall[0], 3)) + "$ ($" + str(round(kendall[1], 3)) + "$) \\\\ \n")
    f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.close()

coherency_file = "all_seo_features_weighted_0"
coherency_label_sentence_pairs,coherencey_stats = read_coherency_file(coherency_file)
sampled = sample_pairs_uniformly(coherency_label_sentence_pairs)
samples = []
for bucket in sampled:
    samples.extend(sampled[bucket])

# samples=sorted(samples)
ks_vector = [ks_stats[pair] for pair in samples]
coherency_label_vector = [coherencey_stats[pair] for pair in samples]


ident_labels = read_tags("document_identification_tags")
ident_vector =[ident_labels[pair] for pair in samples]
sentence_labels = read_tags("sentence_identification_tags")
sentence_vector =[sentence_labels[pair] for pair in samples]

write_tables(ks_vector,coherency_label_vector,ident_vector,sentence_vector)


# texts = read_documents()
# create_ds_for_annotations(texts, sampled)

