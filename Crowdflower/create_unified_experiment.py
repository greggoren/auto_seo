from Crowdflower import create_full_ds_per_task as mturk_ds_creator
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
from CrossValidationUtils.rankSVM_crossvalidation import get_average_score_increase
from utils import run_bash_command
def read_seo_score(labels):
    scores = {}
    with open(labels) as labels_file:
        for line in labels_file:
            id = line.split()[2]
            score = int(line.split()[3].rstrip())
            scores[id]=score
    return scores


def get_level(score):
    demotion_level = 0
    if score >=0 and score <2:
        demotion_level=2
    elif score >=2 and score <4:
        demotion_level=1
    return demotion_level


def modify_seo_score_by_demotion(seo_scores, coherency_scores):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        demotion_level = get_level(coherency_score)
        new_score = max(current_score-demotion_level,0)
        new_scores[id] = new_score
    return new_scores


def create_harmonic_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        new_coherency_score = coherency_score*(4/5)
        numerator = (1+beta**2)*new_coherency_score*current_score
        denominator = (beta**2)*new_coherency_score+current_score
        if denominator!=0:
            harmonic_mean = numerator/denominator #(2*new_coherency_score*current_score)/(new_coherency_score+current_score)
        else:
            harmonic_mean = 0
        new_scores[id]=harmonic_mean
    return new_scores

def create_weighted_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    for id in seo_scores:
        current_score = seo_scores[id]
        coherency_score = coherency_scores[id]
        new_coherency_score = coherency_score * (4 / 5)
        new_score = current_score*beta+new_coherency_score*(1-beta)
        new_scores[id]=new_score
    return new_scores


def save_max_mix_stats(stats,row,query):
    features = list(row.keys())
    if query not in stats:
        stats[query]={}
    for feature in features:
        if feature not in stats[query]:
            stats[query][feature]={}
            stats[query][feature]["max"]  = row[feature]
            stats[query][feature]["min"] = row[feature]
        if row[feature]>stats[query][feature]["max"]:
            stats[query][feature]["max"] = row[feature]
        if row[feature]<stats[query][feature]["min"]:
            stats[query][feature]["min"] = row[feature]
    return stats

def create_coherency_features():
    rows={}
    max_min_stats={}
    model = WordToVec().load_model()
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q: ranked_lists[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists}
    winner_docs = {q: ranked_lists[q][:3] for q in ranked_lists}
    a_doc_texts = load_file(params.trec_text_file)
    doc_texts = {}
    for doc in a_doc_texts:
        if doc.__contains__("ROUND-04"):
            doc_texts[doc] = a_doc_texts[doc]
    sentence_map = map_set_of_sentences(doc_texts, winner_docs)
    for query in sentence_map:
        ref_doc = reference_docs[query]

        text = doc_texts[ref_doc]
        ref_sentences = retrieve_sentences(text)
        for sentence in sentence_map[query]:

            sentence_vec = get_sentence_vector(sentence_map[query][sentence],model=model)
            for i,ref_sentence in enumerate(ref_sentences):
                row = {}
                run_name = sentence+"_"+str(i+1)
                window = []
                if i == 0:
                    window.append(get_sentence_vector(ref_sentences[1],model))
                    window.append(get_sentence_vector(ref_sentences[1],model))

                elif i+1 == len(ref_sentences):
                    window.append(get_sentence_vector(ref_sentences[i-1],model))
                    window.append(get_sentence_vector(ref_sentences[i-1],model))
                else:
                    window.append(get_sentence_vector(ref_sentences[i - 1], model))
                    window.append(get_sentence_vector(ref_sentences[i+1],model))
                ref_vector = get_sentence_vector(ref_sentence,model)
                query = run_name.split("-")[2]
                row["similarity_to_prev"]=cosine_similarity(sentence_vec,window[0])
                row["similarity_to_ref_sentence"] = cosine_similarity(ref_vector,sentence_vec)
                row["similarity_to_pred"] = cosine_similarity(sentence_vec,window[1])
                row["similarity_to_prev_ref"] = cosine_similarity(ref_vector,window[0])
                row["similarity_to_pred_ref"] = cosine_similarity(ref_vector,window[1])
                max_min_stats=save_max_mix_stats(max_min_stats,row,query)
                rows[run_name]=row
    return rows,max_min_stats


def normalize_feature(feature_value,max_min_stats,query,feature):
    if max_min_stats[query][feature]["max"]==max_min_stats[query][feature]["min"]:
        return 0
    denominator =max_min_stats[query][feature]["max"]-max_min_stats[query][feature]["min"]
    value = (feature_value-max_min_stats[query][feature]["min"])/denominator
    return value

def rewrite_fetures(new_scores, coherency_features_set, old_features_file, new_features_filename, coherency_features_names,qrels_name,max_min_stats):
    f = open(new_features_filename,"w")
    qrels = open(qrels_name,"w")
    with open(old_features_file) as file:
        for line in file:
            qid = line.split()[1]
            query = qid.split(":")[1]
            features = line.split()[2:-2]
            number_of_features = len(features)
            id = line.split(" # ")[1].rstrip()
            if id not in new_scores:
                continue
            coherency_features = [str(i)+":"+str(normalize_feature(coherency_features_set[id][feature],max_min_stats,query,feature)) for i,feature in enumerate(coherency_features_names,start=number_of_features+1)]
            new_line = str(new_scores[id]) + " " + qid + " " + " ".join(features) + " " + " ".join(coherency_features) + " # " + id + "\n"
            f.write(new_line)
            qrels.write(query+" 0 "+id+" "+str(new_scores[id])+"\n")
    f.close()
    qrels.close()


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




def write_histogram_for_weighted_scores(hist_scores,filename,beta,last=False):

    file = Path(filename)
    if not file.is_file():
        f = open(filename, "a")
        cols = "c|"*6
        cols = "|"+cols
        f.write("\\begin{tabular}{"+cols+"} \n")
        f.write("\\hline \n")
        f.write("$\\beta$ & 0 & 1 & 2 & 3 & 4 & 5 \\\\ \n")
        f.write("\\hline \n")
    else:
        f = open(filename, "a")
    line = str(beta)+" & "
    for i in range(5):
        add = " & "
        if i==5:
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
        f = open(filename, "a")
        for j,line in enumerate(file_w):
            if not flag:
                if j==0:
                    f.write(line)
                    f.write("\\hline\n")
                if j==1:
                    f.write("METHOD & $ \\beta $ & "+line.upper())
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
    ranked_lists = retrieve_ranked_lists(params.ranked_lists_file)
    reference_docs = {q: ranked_lists[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists}
    dir = "nimo_annotations"
    sorted_files = sort_files_by_date(dir)

    original_docs = retrieve_initial_documents()
    scores={}
    for k in range(4):
        needed_file = sorted_files[k]
        scores = get_scores(scores,dir + "/" + needed_file,original_docs)
    banned_queries = get_banned_queries(scores,reference_docs)
    ident_filename_fe = "figure-eight/ident_current.csv"
    ident_filename_mturk = "Mturk/Manipulated_Document_Identification.csv"
    ident_fe = mturk_ds_creator.read_ds_fe(ident_filename_fe, True)
    ident_mturk = mturk_ds_creator.read_ds_mturk(ident_filename_mturk, True)

    ident_results = mturk_ds_creator.combine_results(ident_fe, ident_mturk)
    # final_ident_results = ban_non_coherent_docs(scores,ident_results)
    sentence_filename_fe = "figure-eight/sentence_current.csv"
    sentence_filename_mturk = "Mturk/Sentence_Identification.csv"
    sentence_filename_mturk_new = "Mturk/Sentence_Identification11.csv"
    sentence_fe = mturk_ds_creator.read_ds_fe(sentence_filename_fe)
    sentence_mturk = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk)
    sentence_mturk_new = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk_new)
    sentence_mturk = mturk_ds_creator.update_dict(sentence_mturk, sentence_mturk_new)

    sentence_results = mturk_ds_creator.combine_results(sentence_fe, sentence_mturk)
    # final_sentence_results = ban_non_coherent_docs(scores,sentence_results)
    sentence_tags = mturk_ds_creator.get_tags(sentence_results)
    ident_tags = mturk_ds_creator.get_tags(ident_results)
    tmp_aggregated_results = mturk_ds_creator.aggregate_results(sentence_tags,ident_tags)
    aggregated_results = ban_non_coherent_docs(banned_queries,tmp_aggregated_results)

    coherency_features = ["similarity_to_prev", "similarity_to_ref_sentence", "similarity_to_pred",
                          "similarity_to_prev_ref", "similarity_to_pred_ref"]
    seo_scores_file = "labels_final1"
    tmp_seo_scores = read_seo_score(seo_scores_file)
    seo_scores = ban_non_coherent_docs(banned_queries,tmp_seo_scores)
    modified_scores= modify_seo_score_by_demotion(seo_scores,aggregated_results)
    seo_features_file = "new_sentence_features"
    coherency_features_set,max_min_stats = create_coherency_features()
    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"
    rewrite_fetures(modified_scores,coherency_features_set,seo_features_file,new_features_with_demotion_file,coherency_features,new_qrels_with_demotion_file,max_min_stats)
    cross_validation(new_features_with_demotion_file, new_qrels_with_demotion_file, "summary_labels_demotion.tex", "svm_rank",
                     ["map", "ndcg", "P.2", "P.5"], "",seo_scores)
    run_random(new_features_with_demotion_file,new_qrels_with_demotion_file,"demotion",seo_scores)
    betas = [0,0.5,1,2]
    flag =False
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_"+str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_"+str(beta)
        harmonic_mean_scores = create_harmonic_mean_score(seo_scores,aggregated_results,beta)
        rewrite_fetures(harmonic_mean_scores, coherency_features_set, seo_features_file, new_features_with_harmonic_file,
                        coherency_features, new_qrels_with_harmonic_file,max_min_stats)
        cross_validation(new_features_with_harmonic_file, new_qrels_with_harmonic_file, "summary_labels_harmonic_"+str(beta)+".tex",
                         "svm_rank",
                         ["map", "ndcg", "P.2", "P.5"], "",seo_scores)
        run_random(new_features_with_harmonic_file, new_qrels_with_harmonic_file, "harmonic_"+str(beta),seo_scores)
        write_weighted_results("summary_labels_harmonic_"+str(beta)+".tex", "summary_labels_harmonic.tex", beta,
                               "RankSVM",flag)
        flag=True
        last = False
        if beta==betas[-1]:
            last=True
        write_weighted_results("summary_randomharmonic_" + str(beta) + ".tex", "summary_labels_harmonic.tex", beta,
                               "RandomBaseline",flag,last)
        harmonic_hist = get_histogram(harmonic_mean_scores)
        write_histogram_for_weighted_scores(harmonic_hist, "harmonic_histogram.tex", beta,last)
    flag=False
    betas = [i/10 for i in range(0,11)]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_"+str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_"+str(beta)
        weighted_mean_scores = create_weighted_mean_score(seo_scores, aggregated_results,beta)
        rewrite_fetures(weighted_mean_scores, coherency_features_set, seo_features_file, new_features_with_weighted_file,
                        coherency_features, new_qrels_with_weighted_file,max_min_stats)
        final_trec_file=cross_validation(new_features_with_demotion_file,new_qrels_with_weighted_file, "summary_labels_weighted"+str(beta)+".tex","svm_rank",["map", "ndcg", "P.2", "P.5"], "",seo_scores)
        if int(beta)==0:
            increase = get_average_score_increase(seo_scores,final_trec_file,True)
            run_bash_command("cp "+final_trec_file+" trec_debug")
            run_bash_command("cp features_svm_rank_1 features_debug")
        run_random(new_features_with_weighted_file, new_qrels_with_weighted_file, "weighted_"+str(beta),seo_scores)

        write_weighted_results("summary_labels_weighted"+str(beta)+".tex","summary_labels_weighted.tex",beta,"RankSVM",flag)
        flag = True
        last = False
        if beta == betas[-1]:
            last = True
        write_weighted_results("summary_randomweighted_"+str(beta)+".tex","summary_labels_weighted.tex",beta,"RandomBaseline",flag,last)
        weighted_hist = get_histogram(weighted_mean_scores)
        write_histogram_for_weighted_scores(weighted_hist, "weighted_histogram.tex", beta,last)
    print("queries=",len(get_dataset_stas(aggregated_results)))
    print("examples=",len(aggregated_results))
    print("histogram_coherency",get_histogram(aggregated_results))
    print("histogram_demotion",get_histogram(modified_scores))
