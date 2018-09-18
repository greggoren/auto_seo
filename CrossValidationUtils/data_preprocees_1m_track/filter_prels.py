def get_needed_queries(prels_stats):
    queries=[]
    for query in prels_stats:
        relevant = [doc for doc in prels_stats[query] if prels_stats[query][doc]>0]
        not_relevant = [doc for doc in prels_stats[query] if prels_stats[query][doc]==0]
        if relevant and not_relevant:
            queries.append(query)
    return queries
def filter(prels_file,doc_names_file):
    docs = get_docs(doc_names_file)
    prels_stats = get_stats(prels_file)
    queries_to_keep = get_needed_queries(prels_stats)
    new_prels = open("updated_prels","w")
    with open(prels_file) as prels:
        for line in prels:
            if line.split()[0] in queries_to_keep and line.split()[1] in docs:
                new_prels.write(line)
        new_prels.close()

def get_docs(doc_names_file):
    f = open(doc_names_file)
    docs = {doc.rstrip():True for doc in f}
    f.close()
    return docs


def get_stats(file):
    prels_stats ={}
    with open(file) as prels:
        for line in prels:
            if line.split()[0] not in prels_stats:
                prels_stats[line.split()[0]]={}
            prels_stats[line.split()[0]][line.split()[1]]=int(line.split()[2])
    return prels_stats


filter("../data/prels.20001-60000","../data/docNames")

