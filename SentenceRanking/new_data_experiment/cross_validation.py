from Crowdflower import create_full_ds_per_task as mturk_ds_creator
from Crowdflower.seo_utils import create_coherency_features
from utils import cosine_similarity
from SentenceRanking.sentence_features_experiment import get_sentence_vector
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import params
from w2v.train_word2vec import WordToVec
from CrossValidationUtils.rankSVM_crossvalidation import cross_validation
from CrossValidationUtils.random_baseline import run_random
from Crowdflower.ban_non_coherent_docs import get_scores,sort_files_by_date,retrieve_initial_documents,ban_non_coherent_docs,get_dataset_stas,get_banned_queries
from pathlib import Path
import numpy as np

def read_seo_score(labels):
    scores = {}
    with open(labels) as labels_file:
        for line in labels_file:
            query = line.split()[0]
            key = query[3:]
            if key not in scores:
                scores[key]={}
            id = line.split()[2]
            score = int(line.split()[3].rstrip())
            scores[key][id]=score
    return scores


def get_histogram(dataset):
    hist ={}
    for id in dataset:
        if dataset[id]<1:
            bucket =0
        elif dataset[id]<2:
            bucket =1
        elif dataset[id]<3:
            bucket =2
        elif dataset[id]<4:
            bucket =3
        elif dataset[id]<5:
            bucket=4
        else:
            bucket=5
        if bucket not in hist:
            hist[bucket]=0
        hist[bucket]+=1
    total_examples = sum([hist[b] for b in hist])
    for bucket in hist:
        hist[bucket]=round(hist[bucket]/total_examples,3)
    return hist

def get_average_score_increase_for_initial_rank(seo_scores, ranked_lists_file):

    lists={}
    ranks_stats ={}
    seen=[]
    with open(ranked_lists_file) as file:
        for line in file:
            query = line.split()[0]
            run_name = line.split()[2]
            key=query[3:]
            if query not in lists:
                lists[query]=[]
            if len(lists[query])>=5:
                if query in seen:
                    continue
            lists[query].append(seo_scores[key][run_name])

            if len(lists[query]) >= 5:
                if query in seen:
                    continue
                if query not in ranks_stats:
                    ranks_stats[query] = {}
                    ranks_stats[query][1]=np.mean(lists[query][:1])
                    ranks_stats[query][2]=np.mean(lists[query][:2])
                    ranks_stats[query][5]=np.mean(lists[query])
                seen.append(query)

    for query in lists:
        if "ge" not in ranks_stats[query]:
            ranks_stats[query]["ge"]=[]
            ranks_stats[query]["eq"]=[]
            ranks_stats[query]["le"]=[]
        if lists[query][0]>lists[query][1]:
            ranks_stats[query]["ge"].append(1)
            ranks_stats[query]["eq"].append(0)
            ranks_stats[query]["le"].append(0)
        elif lists[query][0]==lists[query][1]:
            ranks_stats[query]["ge"].append(0)
            ranks_stats[query]["eq"].append(1)
            ranks_stats[query]["le"].append(0)
        else:
            ranks_stats[query]["ge"].append(0)
            ranks_stats[query]["eq"].append(0)
            ranks_stats[query]["le"].append(1)
    for key in ranks_stats:
        for stat in ["ge","le","eq"]:
            ranks_stats[key][stat]=np.mean(ranks_stats[key][stat])
    return ranks_stats


def write_rank_promotion_stats_per_initial_rank(stats,method):
    f = open("summary_rank_promotion_"+method+".tex","w")
    f.write("\\begin{tabular}{|c|c|c|c|c|c|c|c|}\n")
    f.write("\\hline\n")
    f.write("$\\beta$ & Initial Rank & TOP1 & TOP2 & TOP5 & $>$ & $=$ & $<$ \\\\ \n")
    f.write("\\hline\n")
    for beta in stats:
        ranks = sorted(list(stats[beta].keys()))
        for rank in ranks:
            line = beta+" & "+str(rank)+" & "+" & ".join([str(round(stats[beta][rank][i],3)) for i in [1,2,5,"ge","eq","le"]])+" \\\\ \n"
            f.write(line)
            f.write("\\hline\n")
    f.write("\\end{tabular}\n")
    f.close()

def write_histogram_for_weighted_scores(hist_scores,filename,beta,flag=False,last=False):

    if not flag:
        f = open(filename, "w")
        cols = "c|"*6
        cols = "|"+cols
        f.write("\\begin{tabular}{"+cols+"} \n")
        f.write("\\hline \n")
        f.write("$ \\beta $ & 0 & 1 & 2 & 3 & 4 \\\\ \n")
        f.write("\\hline \n")
    else:
        f = open(filename, "a")
    line = str(beta)+" & "
    for i in range(5):
        add = " & "
        if i==4:
            add ="\\\\ \n"
        if i in hist_scores:
            line+=str(hist_scores[i])+add
        else:
            line += "0"+add
    f.write(line)
    f.write("\\hline\n")
    if last:
        f.write("\\end{tabular}\n")
    f.close()


def write_weighted_results(weighted_results_file,filename,beta,method,flag=False,last=False):
    with open(weighted_results_file) as file_w:
        if not flag:
            f = open(filename, "w")
        else:
            f = open(filename, "a")
        for j,line in enumerate(file_w):
            if not flag:
                if j==0:
                    f.write(line)
                    f.write("\\hline\n")
                if j==1:
                    f.write("METHOD & $ \\beta $ & "+line.upper().rstrip()+"\\\\ \n")
                    f.write("\\hline\n")
                if j==2:
                    f.write(method+" & "+str(beta)+" & "+line)
                    f.write("\\hline\n")
            else:
                if j==2:
                    f.write(method+" & "+str(beta)+" & "+line)
                    f.write("\\hline\n")
        if last:
            f.write("\\end{tabular}\n")
        f.close()




if __name__=="__main__":
    ranked_lists_new = retrieve_ranked_lists("trec_file04")
    reference_docs = {}
    top_docs = {}
    reference_docs["45"] = {q: ranked_lists_new[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists_new}
    top_docs["45"] = {q: ranked_lists_new[q][:3] for q in ranked_lists_new}
    reference_docs["42"] = {q: ranked_lists_new[q][1].replace("EPOCH", "ROUND") for q in ranked_lists_new}
    top_docs["42"] = {q: ranked_lists_new[q][:1] for q in ranked_lists_new}
    ranked_lists_new = retrieve_ranked_lists("trec_file06")
    reference_docs["65"] = {q: ranked_lists_new[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists_new}
    top_docs["65"] = {q: ranked_lists_new[q][:3] for q in ranked_lists_new}
    reference_docs["62"] = {q: ranked_lists_new[q][1].replace("EPOCH", "ROUND") for q in ranked_lists_new}
    top_docs["62"] = {q: ranked_lists_new[q][:1] for q in ranked_lists_new}
    dir = "../../Crowdflower/nimo_annotations"
    sorted_files = sort_files_by_date(dir)
    tmp_doc_texts = load_file(params.trec_text_file)
    doc_texts = {}
    for doc in tmp_doc_texts:
        if doc.__contains__("ROUND-04") or doc.__contains__("ROUND-06"):
            doc_texts[doc] = tmp_doc_texts[doc]
    original_docs = retrieve_initial_documents()
    scores = {}
    for k in range(6):
        needed_file = sorted_files[k]
        scores = get_scores(scores, dir + "/" + needed_file, original_docs, k + 1)
    banned_queries = get_banned_queries(scores, reference_docs)

    seo_scores_file = "labels_new_final_all_data"
    tmp_seo_scores = read_seo_score(seo_scores_file)
    seo_scores = ban_non_coherent_docs(banned_queries, tmp_seo_scores)
    final_features_dir = "sentence_feature_files/"
    features_file = final_features_dir + "new_data_sentence_features"
    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"
    final_trec_file=cross_validation(new_features_with_demotion_file, new_qrels_with_demotion_file, "summary_labels_demotion.tex", "svm_rank",
                     ["map","ndcg_cut.1", "ndcg_cut.5", "P.1"], "",seo_scores)
    run_random(new_features_with_demotion_file,new_qrels_with_demotion_file,"demotion",seo_scores)



    stats_harmonic={}
    betas = [0, 0.5, 1, 2, 1000, 100000, 1000000000]
    # betas = [0,]
    flag =False
    flag1 =False
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_"+str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_"+str(beta)
        final_trec_file=cross_validation(new_features_with_harmonic_file, new_qrels_with_harmonic_file, "summary_labels_harmonic_"+str(beta)+".tex",
                         "svm_rank",
                         ["map","ndcg_cut.1", "ndcg_cut.5", "P.1"], "",seo_scores)
        run_random(new_features_with_harmonic_file, new_qrels_with_harmonic_file, "harmonic_"+str(beta),seo_scores)
        write_weighted_results("summary_labels_harmonic_"+str(beta)+".tex", "summary_labels_harmonic.tex", beta,
                               "RankSVM",flag)

        last = False
        if beta==betas[-1]:
            last=True
        flag = True
        write_weighted_results("summary_randomharmonic_" + str(beta) + ".tex", "summary_labels_harmonic.tex", beta,
                               "RandomBaseline",flag,last)
        flag1=True


    flag=False
    flag1=False
    stats_weighted = {}
    betas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    # betas = [0,]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_"+str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_"+str(beta)
        final_trec_file=cross_validation(new_features_with_weighted_file,new_qrels_with_weighted_file, "summary_labels_weighted"+str(beta)+".tex","svm_rank",["map","ndcg_cut.1", "ndcg_cut.5", "P.1"], "",seo_scores)
        run_random(new_features_with_weighted_file, new_qrels_with_weighted_file, "weighted_"+str(beta),seo_scores)

        write_weighted_results("summary_labels_weighted"+str(beta)+".tex","summary_labels_weighted.tex",beta,"RankSVM",flag)
        flag = True
        last = False
        if beta == betas[-1]:
            last = True
        write_weighted_results("summary_randomweighted_"+str(beta)+".tex","summary_labels_weighted.tex",beta,"RandomBaseline",flag,last)
        flag1=True

