import sys
if __name__=="__main__":
    features_file =sys.argv[1]
    stats={}
    with open(features_file) as f_file:
        for line in f_file:
            label = line.split()[0]
            query = line.split()[1].split(":")[1]
            if query not in stats:
                stats[query]=[]
            stats[query].append(int(label))
    average_ratio=0
    for query in stats:
        labels = stats[query]
        number_of_relevant = sum([1 for i in labels if i>0])
        total = len(labels)
        ratio = number_of_relevant/total
        average_ratio+=ratio
        print(query+" & "+ str(round(number_of_relevant,3))+" & "+str(round(total,3))+" & "+str(round(ratio,3))+"\\\\")
        print("\hline")
    average_ratio=average_ratio/len(stats)
    print("average_ratio=",average_ratio)
