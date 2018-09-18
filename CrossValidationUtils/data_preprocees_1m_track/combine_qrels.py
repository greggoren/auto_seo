def combine(qrels,prels,regular_queries_file,extended_queries_file):
    final_qrels = open("mq_track_qrels","w")
    overllaping_map = get_overlapping_queries(regular_queries_file,extended_queries_file)
    qrels_data = get_qrels_stats(qrels)
    with open(qrels) as file:
        for line in file:
            final_qrels.write(line)
    with open(prels) as file:
        for line in file:
            query = line.split()[0]
            doc = line.split()[1]
            rel = line.split()[2]
            if query in overllaping_map and doc in qrels_data[overllaping_map[query]]:
                continue
            elif query in overllaping_map and doc not in qrels_data[overllaping_map[query]]:
                final_qrels.write(overllaping_map[query]+" 1 "+doc+" "+rel+"\n")
            else:
                final_qrels.write(line)
    final_qrels.close()





def get_overlapping_queries(regular_queries_file,extended_queries_file):
    overlapping = {}
    regular_queries = open(regular_queries_file)
    extended_queries = open(extended_queries_file)
    regular_queries_map ={line.split(":")[1].rstrip():line.split(":")[0] for line in regular_queries}
    extended_queries_map ={line.split(":")[1].rstrip():line.split(":")[0] for line in extended_queries}
    for query_text in regular_queries_map:
        if query_text in extended_queries_map:
            overlapping[extended_queries_map[query_text]]=regular_queries_map[query_text]
    return overlapping

def get_qrels_stats(qrels):
    stats ={}
    with open(qrels) as file:
        for line in file:
            if line.split()[0] not in stats:
                stats[line.split()[0]] =[]
            stats[line.split()[0]].append(line.split()[2])
        return stats

def create_queries_xml():
    xml_file = open("mq_queries.xml","w")
    xml_file.write("<parameters>\n")
