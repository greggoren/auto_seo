import pickle

def read_feature_names(filename):
    feature_index = {}
    with open(filename) as file:
        for line in file:
            feature = line.split()[0]
            index = int(line.split()[1].rstrip())-1
            feature_index[index]=feature
    return feature_index

def create_features_importance_table(feature_index,weights_object):
    f = open(weights_object,"rb")
    weights = pickle.load(f)
    f.close()
    table = open("feature_importance.tex","w")
    table.write("\\begin{tabular}{|c|c|}\n")
    table.write("\\hline\n")
    table.write("Feature & Weight \\\\ \n")
    table.write("\\hline\n")
    for i,w in enumerate(weights):
        feature = feature_index[i]
        weight = str(round(w,3))
        line = feature+" & $"+weight+"$ \\\\ \n"
        table.write(line)
        table.write("\\hline\n")
    table.write("\\end{tabular}\n")
    table.close()


feature_index = read_feature_names("featureID")
create_features_importance_table(feature_index,"demotion_weights.pkl")