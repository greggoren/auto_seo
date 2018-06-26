import numpy as np
import matplotlib.pyplot as plt

def average_rank_addition(old_lists, new_lists, reference_docs):
    additions =[]
    for query in reference_docs:
        doc = reference_docs[query]
        additions.append(abs(old_lists[query].index(doc)-new_lists[query].index(doc)))
    return np.mean(additions),np.median(additions)


def create_histogram(x,x_label,y_label,file_name):
    n, bins, patches = plt.hist(x,50, facecolor='blue', alpha=0.75)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.savefig(file_name)
    plt.clf()



