def add_labeles(label_file_path,old_features,new_features_path):
    label_file = open(label_file_path)
    labels = {line.split()[2]:line.split()[3].replace("\n","") for line in label_file}
    label_file.close()
    new_features = open(new_features_path,"w")
    with open(old_features) as features:
        for line in features:
            splited = line.split()
            label = labels[splited[-1].rstrip()]
            new_line = label+" "+" ".join(splited[1:])
            new_features.write(new_line+"\n")
        new_features.close()
        return new_features_path


add_labeles("qrels","features","features_spam_filtered")
