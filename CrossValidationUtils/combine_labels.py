import sys
def add_labeles(label_file_path,old_features,new_features_path):
    labels = {}
    label_file = open(label_file_path)
    # labels = {line.split()[2]:line.split()[3].replace("\n","") for line in label_file}
    for line in labels:
        query = line.split()[0]
        doc = line.split()[2]
        score = line.split()[3].replace("\n","")
        if query not in labels:
            labels[query]={}
        labels[query][doc]=score
    label_file.close()
    new_features = open(new_features_path,"w")
    with open(old_features) as features:
        for line in features:
            splited = line.split()
            query = splited[1].split(":")[1]

            if splited[-1].rstrip() in labels[query]:
                if int(labels[query][splited[-1].rstrip()])>=0:
                    label = labels[splited[-1].rstrip()]
                else:
                    label = "0"
            else:
                label="0"
            new_line = label+" "+" ".join(splited[1:])
            new_features.write(new_line+"\n")
        new_features.close()


if __name__=="__main__":
    if len(sys.argv)<4:
        print("not enough params")
        sys.exit(1)
    label_file_path, old_features, new_features_path = sys.argv[1],sys.argv[2],sys.argv[3]
    add_labeles(label_file_path,old_features,new_features_path)