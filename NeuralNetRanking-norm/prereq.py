def create_data_file(data_set,features_file):
    stats = get_labels(features_file)
    f = open("new_sentences_add_remove","w")
    with open(data_set) as data:
        for line in data:
            comb=line.split("!@@@!")[0]
            query = comb.split("-")[2]
            new_line = query+"!@@@!"+line.rstrip()+"!@@@!"+stats[comb]+"\n"
            f.write(new_line)
    f.close()

def get_labels(features_file):
    stats={}
    with open(features_file) as features:
        for line in features:
            label = line.split()[0]
            comb = line.split()[-1].rstrip()
            stats[comb] = label
    return stats



data_set = "/home/greg/auto_seo/scripts/senetces_add_remove"
features_file = "/home/greg/auto_seo/SentenceRanking/new_sentence_features"

create_data_file(data_set,features_file)