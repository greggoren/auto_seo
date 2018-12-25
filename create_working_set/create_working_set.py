def create_set(filename):
    working_set={}
    f = open("workin_set_spam_filtered","w")
    with open(filename) as file:
        for line in file:
            doc = line.split(" # ")[1].rstrip()
            query = line.split()[1].split(":")[1]
            if query not in working_set:
                working_set[query]=1
            index =working_set[query]
            f.write(query+" Q0 "+doc+" "+str(index)+" "+str(-index)+" seo\n")
            working_set[query]+=1
    f.close()

create_set("features")

