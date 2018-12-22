import sys
from random import shuffle,seed
import numpy as np
from utils import run_command,run_bash_command
from CrossValidationUtils.rankSVM_crossvalidation import get_average_score_increase
def run_random(features_file, qrels, name, seo_scores=False):
    seed(9004)
    score_data = {}
    averaged_rank_increase_stats = {}
    for i in range(10):
        data = {}
        print("in iteration", i + 1)
        # run_bash_command("rm /home/greg/auto_seo/CrossValidationUtils/random_scores")
        scores = open("random_scores" + name, "w")

        features = open(features_file)
        for line in features:
            query = line.split()[1].split(":")[1]
            sentence = line.split(" # ")[1]
            if query not in data:
                data[query] = []
            data[query].append(sentence)

        for query in data:
            shuffle(data[query])
            index = 1
            for object in data[query]:
                scores.write(query + " Q0 " + object.rstrip() + " 0 " + str(index) + " seo\n")
                index += 1
        scores.close()
        if seo_scores:
            tmp_rank_increase_score = get_average_score_increase(seo_scores,"random_scores" + name)
            if not averaged_rank_increase_stats:
                for key in tmp_rank_increase_score:
                    averaged_rank_increase_stats[key]=[]
            for key in tmp_rank_increase_score:
                averaged_rank_increase_stats[key].append(tmp_rank_increase_score[key])

        for metric in ["map", "ndcg", "P.2", "P.5"]:
            command = "./trec_eval -m " + metric + " " + qrels + " random_scores" + name
            for output_line in run_command(command):
                print(metric, output_line)
                score = output_line.split()[-1].rstrip()
                score = str(score).replace("b'", "")
                score = score.replace("'", "")
                if metric not in score_data:
                    score_data[metric] = []
                score_data[metric].append(float(score))
        features.close()

    if seo_scores:
        for key in averaged_rank_increase_stats:
            averaged_rank_increase_stats[key]=round(np.mean(averaged_rank_increase_stats[key]),4)
    summary_file = open("summary_random" + str(name) + ".tex", 'w')

    if not seo_scores:
        cols = "c|" * 4
    else:
        cols = "c|" * 7
    cols = "|" + cols
    summary_file.write("\\begin{tabular}{" + cols + "}\n")
    if not seo_scores:
        next_line = " & ".join([s for s in ["map", "ndcg", "P.2", "P.5"]]) + "\\\\ \n"
    else:
        next_line = " & ".join([s for s in ["map", "ndcg", "P.2", "P.5"]])+" & " +" & ".join(["Top1","Top2","Top5"])+ "\\\\ \n"
    summary_file.write(next_line)
    if not seo_scores:
        next_line = " & ".join([str(round(np.mean(score_data[s]),4)) for s in ["map", "ndcg", "P.2", "P.5"]]) + "\n"
    else:
        next_line = " & ".join([str(round(np.mean(score_data[s]), 4)) for s in ["map", "ndcg", "P.2", "P.5"]])+" & "+" & ".join([str(averaged_rank_increase_stats[j]) for j in [1,2,5]]) + "\\\\ \n"
    summary_file.write(next_line)
    summary_file.write("\\end{tabular}")
    summary_file.close()


if __name__=="__main__":

    features_file =sys.argv[1]
    qrels =sys.argv[2]
    run_random(features_file,qrels,"")
