import params
from utils import run_bash_command
import  os
import math

def create_features_from_dir(features_dir,features_file,sentence_working_set):
    command = "perl " + params.sentence_feature_creator + " " + features_dir + " " + sentence_working_set
    run_bash_command(command)
    command = "mv features " + features_file
    run_bash_command(command)


def read_files_and_get_labels(files_dir,feature):
    labels = {}
    for file_name in os.listdir(files_dir):
        if file_name.split("_")[0]!=feature:
            continue
        query = file_name.split("_")[1]
        if query not in labels:
            labels[query]={}
        with open(files_dir+"/"+file_name) as f:
            for line in f:
                doc = line.split()[0]
                val = line.split()[1].rstrip()
                labels[query][doc]=val
    return labels

# def create_labels_file(labels,file_name):
#     with open(file_name,"w") as f:
#
#         for query in labels:
#             for doc in labels[query]:
#                 f.write(query+" 0 "+doc+" "+labels[query][doc]+"\n")


def rewrite_fetures(new_scores, old_features_file, new_features_filename,qrels_name):
    f = open(new_features_filename,"w")
    qrels = open(qrels_name,"w")
    with open(old_features_file) as file:
        for line in file:
            qid = line.split()[1]
            query = qid.split(":")[1]
            features = line.split()[2:-2]
            id = line.split(" # ")[1].rstrip()
            if id not in new_scores[query]:
                continue
            new_line=str(new_scores[query][id])+" qid:"+query+" "+" ".join(features)+" # "+id
            f.write(new_line+"\n")
            qrels.write(query+" 0 "+id+" "+str(new_scores[query][id])+"\n")
    f.close()
    qrels.close()


def normalize(number):
    if abs(math.ceil(number)-number)>=abs(math.floor(number)-number):
        return math.floor(number)
    else:
        return math.ceil(number)

def combine_score(scores_in,scores_out):
    scores = {}
    for query in scores_in:
        if query not in scores:
            scores[query]={}
        for doc in scores_in[query]:
            val1 = float(scores_in[query][doc])
            val2 = float(scores_out[query][doc])
            val=((val1+val2)/2)*3
            scores[query][doc]= str(normalize(val))
    return scores

if __name__=='__main__':
    final_features_dir = "sentence_feature_files_test/"
    features_file = final_features_dir + "new_data_sentence_features_test"
    features_dir = "sentence_feature_values_test/"
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(final_features_dir):
        os.makedirs(final_features_dir)
    total_working_set_file = "total_working_set_file_test"
    create_features_from_dir(features_dir, features_file, total_working_set_file)
    scores_in = read_files_and_get_labels(features_dir,"docCosineToWinnerCentroidInVec")
    scores_out = read_files_and_get_labels(features_dir,"docCosineToWinnerCentroidOutVec")
    scores = combine_score(scores_in,scores_out)
    rewrite_fetures(scores,features_file,"WinnerCentroidInVec","WinnerCentroidInVecQrels")

    scores_in = read_files_and_get_labels(features_dir, "docCosineToWinnerCentroidIn")
    scores_out = read_files_and_get_labels(features_dir, "docCosineToWinnerCentroidOut")
    scores = combine_score(scores_in, scores_out)
    rewrite_fetures(scores, features_file, "WinnerCentroidIn", "WinnerCentroidInQrels")