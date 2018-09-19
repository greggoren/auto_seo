def combine(qrels,prels,regular_queries_file,extended_queries_file):
    final_qrels = open("mq_track_qrels","w")
    overllaping_map,regular_map,extended_map = get_overlapping_queries(regular_queries_file,extended_queries_file)
    qrels_data = get_qrels_stats(qrels)
    seen = []
    index = 1
    index_map={}
    with open(qrels) as file:
        for line in file:
            query = line.split()[0]
            if regular_map[query] not in seen:
                seen.append(regular_map[query])
                index+=1
            final_qrels.write(line)
    with open(prels) as file:
        for line in file:
            query = line.split()[0]
            if extended_map[query] in seen:
                continue
            doc = line.split()[1]
            rel = line.split()[2]
            if query not in index_map:
                index+=1
                index_map[query]=str(index).zfill(3)
            if query in overllaping_map:
                if overllaping_map[query] in qrels_data:
                    if doc in qrels_data[overllaping_map[query]]:
                        continue
                    else:
                        final_qrels.write(overllaping_map[query] + " 1 " + doc + " " + rel + "\n")
            else:
                final_qrels.write(index_map[query]+" 1 "+ doc + " " + rel + "\n")
    final_qrels.close()
    return index_map



def get_extended_queries(prels):
    f = open(prels)
    queries = [line.split()[0] for line in f]
    f.close()
    return set(queries)

def get_overlapping_queries(regular_queries_file,extended_queries_file):
    overlapping = {}
    regular_queries = open(regular_queries_file)
    extended_queries = open(extended_queries_file)
    regular_queries_map ={line.split(":")[1].rstrip():line.split(":")[0] for line in regular_queries}
    extended_queries_map = {line.split(":")[1].rstrip(): line.split(":")[0] for line in extended_queries}
    regular_queries.close()
    extended_queries.close()
    regular_queries = open(regular_queries_file)
    extended_queries = open(extended_queries_file)
    regular_queries_map_rev ={line.split(":")[0].rstrip():line.split(":")[1].rstrip() for line in regular_queries}
    extended_queries_map_rev ={line.split(":")[0]:line.split(":")[1].rstrip() for line in extended_queries}
    regular_queries.close()
    extended_queries.close()
    for query_text in regular_queries_map:
        if query_text in extended_queries_map:
            overlapping[extended_queries_map[query_text]]=regular_queries_map[query_text]
    return overlapping,regular_queries_map_rev,extended_queries_map_rev

def get_qrels_stats(qrels):
    stats ={}
    with open(qrels) as file:
        for line in file:
            if line.split()[0] not in stats:
                stats[line.split()[0]] =[]
            stats[line.split()[0]].append(line.split()[2])
        return stats

def create_queries_xml(regular_queries_file,extended_queries_file,extended_queries,index_map):
    overlapping_map,regular_queries_map,extended_queries_map = get_overlapping_queries(regular_queries_file,extended_queries_file)
    seen =[]
    xml_file = open("mq_queries.xml","w")
    xml_file.write("<parameters>\n")
    for query in regular_queries_map:
        xml_file.write("<query><number>"+query+"</number><text>#combine("+regular_queries_map[query]+")</text></query>\n")
        if regular_queries_map[query] not in seen:
            seen.append(regular_queries_map[query])
    for query in extended_queries_map:
        if query in overlapping_map or extended_queries_map[query] in seen:
            continue
        elif query in extended_queries:
            xml_file.write("<query><number>" + index_map[query] + "</number><text>#combine(" + extended_queries_map[
                query] + ")</text></query>\n")
    xml_file.write("</parameters>\n")
    xml_file.close()
    seen = []
    query_text_file = open("mq_queries.txt","w")
    for query in regular_queries_map:
        query_text_file.write(query+":"+regular_queries_map[query]+"\n")
        if regular_queries_map[query] not in seen:
            seen.append(regular_queries_map[query])
    for query in extended_queries_map:
        if query in overlapping_map or extended_queries_map[query] in seen:
            continue
        elif query in extended_queries:
            query_text_file.write(index_map[query] + ":" + extended_queries_map[query] + "\n")
    query_text_file.close()


qrels = "../data/qrels"
prels = "../data/updated_prels"
regular_queries_file = "../data/queries.txt"
extended_queries_file = "../data/mq_queries.txt"
extended_queries = get_extended_queries(prels)
index_map=combine(qrels,prels,regular_queries_file,extended_queries_file)
create_queries_xml(regular_queries_file,extended_queries_file,extended_queries,index_map)