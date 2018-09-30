import sys


def get_new_working_set(working_set_file,number_of_docs):
    new_working_set = {}
    with open(working_set_file) as working_set:
        for line in working_set:
            query=line.split()[0]
            doc = line.split()[1]
            if query not in new_working_set:
                new_working_set[query]=[]
            if len(new_working_set[query])>=number_of_docs:
                continue
            else:
                new_working_set[query].append(doc)
    return new_working_set


def create_append_file(working_set,feature_file,number_of_docs):
    f = open("append_features_mq_"+str(number_of_docs),"w")
    with open(feature_file) as features:
        for line in features:
            doc = line.split(" # ")[1]
            query = line.split()[1].split(":")[1]
            if doc in working_set[query]:
                f.write(line)
    f.close()


if __name__=="__main__":
    feature_file = sys.argv[1]
    number_of_docs = int(sys.argv[2])
    working_set_file = sys.argv[3]
    new_working_set = get_new_working_set(working_set_file,number_of_docs)
    create_append_file(new_working_set,feature_file,number_of_docs)

