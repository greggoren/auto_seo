import numpy as np
from CompetitionBot.create_ds_for_annotations import get_reference_documents
import matplotlib.pyplot as plt
import os
import pickle


def read_file(reference_docs,file,index,stats):

    with open(file) as f:
        for line in f:
            query = line.split()[0]
            doc = line.split()[1].split("-")[3]
            similarity = float(line.split()[3].rstrip())
            if doc in reference_docs[query]:
                if index not in stats["Bot"]:
                    stats["Bot"][index]={}
                if query not in stats["Bot"][index]:

                    stats["Bot"][index][query]=[]
                stats["Bot"][index][query].append(similarity)
            elif doc.__contains__("dummy_doc"):
                if index not in stats["Dummy"]:
                    stats["Dummy"][index]={}
                if query not in stats["Dummy"][index]:

                    stats["Dummy"][index][query]=[]
                stats["Dummy"][index][query].append(similarity)
            else:
                if index not in stats["Active"]:
                    stats["Active"][index]={}
                if query not in stats["Active"][index]:

                    stats["Active"][index][query]=[]
                stats["Active"][index][query].append(similarity)
        for group in stats:
            for query in stats[group][index]:
                stats[group][index][query] = np.mean(stats[group][index][query])
            stats[group][index]=np.mean([stats[group][index][q] for q in stats[group][index]])
        return stats

def gather_stats(dir):
    stats = {"Bot": {}, "Active": {}, "Dummy": {}}
    f = open("ref_docs","rb")
    ref_docs=pickle.load(f)
    f.close()

    # pickle.dump(ref_docs,open("ref_docs","wb"))
    for file in os.listdir(dir):
        index = file.split("_")[2]
        stats = read_file(ref_docs,dir+"/"+file,index,stats)
    return stats


def create_graph(stats):
    params = {'legend.fontsize': 'x-large',
              'figure.figsize': (10, 7),
              'axes.labelsize': 'x-large',
              'axes.titlesize': 'x-large',
              'xtick.labelsize': 'x-large',
              'ytick.labelsize': 'x-large',
              'font.family': 'serif'}
    plt.rcParams.update(params)
    group_name_dict = {"Bot": "Bot", "Active": "Students", "static": "Static", "top": "S-T","Dummy":"Planted"}
    colors_dict = {"Bot": "b", "Active": "r", "static": "y", "top": "k","Dummy":"mediumslateblue"}
    dot_dict = {"Bot": "-o", "Active": "--^", "static": ":p", "top": "-.x","Dummy":"-.+"}


    plt.figure()

    x = [j + 2 for j in range(len(stats["Bot"]))]
    for group in stats:
        y = [stats[group][i] for i in sorted(list(stats[group].keys()))]

        plt.plot(x, y, dot_dict[group], label=group_name_dict[group], color=colors_dict[group], linewidth=5,
                 markersize=10, mew=1)
    plt.xticks(x, fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylabel("Cosine", fontsize=25)
    plt.xlabel("Rounds", fontsize=25)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.11),
               ncol=5, fontsize=20, frameon=False)
    plt.savefig("similarity_to_winner.pdf", format="pdf")
    plt.clf()

stats = gather_stats("similarity_data/")
create_graph(stats)