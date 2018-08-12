with open("../Preprocess/preprocess.py",encoding="utf8") as fp:
    for i,line in enumerate(fp):
        if "\xe2" in line:
            print (i,repr(line))