from utils import run_bash_command,run_command
import sys


def run_trec_eval_on_test(qrels, trec_file):
    metrics=["ndcg_cut.1","ndcg_cut.2"]
    score_data = {}
    print("last stats:")
    for metric in metrics:
        command = "./trec_eval -m " + metric + " " + qrels + " " + trec_file
        for output_line in run_command(command):
            print(metric, output_line)
            score = output_line.split()[-1].rstrip()
            score = str(score).replace("b'", "")
            score = score.replace("'", "")
            score_data[metric] = str(score)
    return score_data


def order_trec_file(trec_file):
    final = trec_file.replace(".txt", "")
    command = "sort -k1,1 -k5nr -k2,1 " + trec_file + " > " + final
    for line in run_command(command):
        print(line)
    return final


features_file = sys.argv[1]
relevance_file = sys.argv[2]

stats={}
stats_rel={}
with open(features_file) as file:
    for line in file:
        doc = line.split(" # ")[1]
        score = line.split()[7].split(":")[1]
        query =doc.split("-")[2]
        rnd = doc.split("-")[1]
        if rnd not in stats:
            stats[rnd] = {}
        if query not in stats[rnd]:
            stats[rnd][query]={}
        stats[rnd][query][doc]=score


with open(relevance_file) as file:
    for line in file:
        doc=line.split()[2]
        query = line.split()[0]
        rnd = doc.split("-")[1]
        rel = line.split()[3]
        if rnd not in stats:
            stats[rnd] = []

        stats[rnd].append(int(rel.rstrip()))
f= open("summary.tex","w")
f.write("\\begin{tabular}{|c|c|c|c|c}\n")
f.write("ROUND & Relevant & Total & NDCG@1 & NDCG@2 \\\\")
f.write("\\hline\n")
for rnd in stats:
    trec_tmp = "trec_file"+rnd+".txt"
    f = open(trec_tmp,"w")
    for query in stats[rnd]:
        for doc in stats[rnd][query]:
            f.write(query+" Q0 "+doc+" 1 "+stats[rnd][query][doc]+" request\n")
    trec_file = order_trec_file(trec_tmp)
    run_bash_command("rm "+trec_tmp)
    qrels = "qrels_"+rnd
    run_bash_command("cat "+relevance_file+" | grep ROUND-"+rnd+" > "+qrels)
    score_data = run_trec_eval_on_test(qrels,trec_file)

    f.write(rnd+" & "+str(sum([1 for i in stats_rel[rnd] if i>0]))+" & "+len(stats[rnd])+" & "+score_data["ndcg_cut.1"]+" & "+score_data["ndcg_cut.2"]+"\\\\ \n")
    f.write("\\hline\n")
f.close()