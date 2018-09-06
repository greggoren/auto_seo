import csv


def retrieve_all_data(folders):
    epochs=[int(e.split("epoch")[1]) for e in folders]
    stat={e:{} for e in epochs}
    for folder in folders:
        file = folder+"\\rel_full.csv"
        with open(file) as relevant:
            reader = csv.DictReader(relevant)
            for row in reader:
                if row['username']!='nimo':
                    key = row['orig__id']
                    epoch = int(folder.split("epoch")[1])

                    if row['this_document_is'].lower()=='relevant':
                        stat[epoch][key]=stat[epoch].get(key,0)+1
                    else:
                        stat[epoch][key] = stat[epoch].get(key, 0)
    return stat


def get_mapping(mapping_file):
    mapping = {}
    with open(mapping_file) as mapping_data:
        for data in mapping_data:

            name = data.split(" # ")[1]
            if not name.__contains__("originalDoc"):
                id = name.split("-")[1].rstrip()
                new_id = "ObjectId("+id+")"
                mapping[new_id] = name.split("-")[0]
    return mapping


def write_relevance_file(stat,mapping,e,last_rel):
    file = open("qrel_asr",'a')

    if e<=5:
        i=1
    else:
        i=6
    if e==6:
        last_rel={}
    docs = list(stat[i].keys())
    for doc in docs:
        if doc in stat[e]:
            last_rel[doc] = stat[e][doc]
        doc_name = "ROUND-"+str(e).zfill(2)+"-"+mapping[doc]+"-"+doc
        if last_rel[doc] >= 3:
            line = mapping[doc] + " 0 " + doc_name + " " + str(last_rel[doc] - 2)+"\n"
        else:
            line = mapping[doc]+" 0 "+doc_name+" "+str(0)+"\n"
        file.write(line)
    file.close()
    return last_rel













epochs = ["epoch"+str(i) for i in range(1,11)]
base_dir ="C:\\study\\msc-thesis\\thesis - new\\competition\\"
folders = [base_dir+e for e in epochs]
base_dir_features = "C:\\study\\msc-thesis\\thesis - new\\competition\\features\\"
stat=retrieve_all_data(folders)
last_rel = {}
for e in epochs:
    print("in ",e)
    mapping_file = base_dir_features+e+"\\features_ww"
    mapping=get_mapping(mapping_file)
    last_rel=write_relevance_file(stat,mapping,int(e.split("epoch")[1]),last_rel)






#
# mapping = get_mapping(mapping_file)
# write_relevance_file(stat, mapping)
# new_rel("documents_updated.rel")