import sys
from random import shuffle,seed
import numpy as np
from utils import run_command,run_bash_command


def run_random(features_file,qrels,num):
    seed(9001)
    score_data = {}

    for i in range(10):
        data = {}
        print("in iteration", i + 1)
        # run_bash_command("rm /home/greg/auto_seo/CrossValidationUtils/random_scores")
        scores = open("random_scores"+num, "w")

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

        for metric in ["map", "ndcg", "P.2", "P.5"]:
            command = "./trec_eval -m " + metric + " " + qrels + " random_scores"+num
            for output_line in run_command(command):
                print(metric, output_line)
                score = output_line.split()[-1].rstrip()
                score = str(score).replace("b'", "")
                score = score.replace("'", "")
                if metric not in score_data:
                    score_data[metric] = []
                score_data[metric].append(float(score))
        features.close()
    summary_file = open("summary_random.tex", 'w')
    cols = "c|" * 3
    cols = "|" + cols
    summary_file.write("\\begin{tabular}{" + cols + "}\n")
    next_line = " & ".join([s for s in score_data]) + "\\\\ \n"
    summary_file.write(next_line)
    next_line = " & ".join([str(np.mean(score_data[s])) for s in score_data]) + "\n"
    summary_file.write(next_line)
    summary_file.write("\\end{tabular}")
    summary_file.close()


if __name__=="__main__":

    features_file =sys.argv[1]
    qrels =sys.argv[2]
    run_random(features_file,qrels,"")
