def create_set(filename):
    working_set={}
    with open(filename) as file:
        for line in file:
            doc = line.split(" # ")[1].rstrip()
            query = line.split()[1].split(":")[1]
            if query not in working_set:
                working_set[query]=0
