import sys


def get_new_working_set(working_set_file,number_of_docs):
    new_working_set = {}
    with open(working_set_file) as working_set:
        for line in working_set:
            query=line.split()[0]
            doc = line.split()[2]
            if query not in new_working_set:
                new_working_set[query]=[]
            if len(new_working_set[query])>=number_of_docs:
                continue
            else:
                new_working_set[query].append(doc)
    return new_working_set


def create_append_file(working_set,feature_file,number_of_docs):
    new_file_name = "append_features_mq_"+str(number_of_docs)
    f = open(new_file_name,"w")
    with open(feature_file) as features:
        for line in features:
            doc = line.split(" # ")[1].rstrip()
            query = line.split()[1].split(":")[1]
            if doc in working_set[query]:
                f.write(line)
    f.close()
    return new_file_name



# def create_stats_table(stats):


def post_analysis(append_file):
    stats = {}
    with open(append_file) as file:
        for line in file:
            query = line.split()[1].split(":")[1]
            rel = int(line.split()[0])
            if query not in stats:
                stats[query]=[]
            stats[query].append(rel)


def create_working_set(working_set,number_of_docs):
    f = open("extended_working_set_"+str(number_of_docs),"w")
    for query in working_set:
        for i,doc in enumerate(working_set[query]):
            f.write(query+" Q0 "+doc+" "+str(i+1)+" "+str(-i-1)+" indri\n")
    f.close()


if __name__=="__main__":
    # feature_file = sys.argv[1]
    number_of_docs = int(sys.argv[2])
    working_set_file = sys.argv[1]
    new_working_set = get_new_working_set(working_set_file,number_of_docs)
    create_working_set(new_working_set,number_of_docs)
    # create_append_file(new_working_set,feature_file,number_of_docs)

