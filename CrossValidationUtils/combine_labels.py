import sys
def add_labeles(label_file_path,old_features,new_features_path):
    label_file = open(label_file_path)
    labels = {line.split()[2]:line.split()[3].replace("\n","") for line in label_file}
    label_file.close()
    new_features = open(new_features_path,"w")
    with open(old_features) as features:
        for line in features:
            splited = line.split()
            if splited[-1].rstrip() in labels:
                if int(labels[splited[-1].rstrip()])>=0:
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