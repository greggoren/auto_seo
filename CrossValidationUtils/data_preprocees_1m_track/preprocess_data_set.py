def preprocess(data_file):
    f = open("featuresCB_asr_new_label","w")
    with open(data_file) as file:
        for line in file:
            label = int(line.split()[0])
            label -=1
            label = max([label,0])
            new_line = str(label)+line[1:]
            f.write(new_line)
    f.close()

preprocess("../featuresCB_asr")


