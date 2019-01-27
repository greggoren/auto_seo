from Crowdflower import create_full_ds_per_task as mturk_ds_creator
from Crowdflower.seo_utils import create_coherency_features
from Preprocess.preprocess import retrieve_ranked_lists,load_file,retrieve_sentences
from Crowdflower.ban_non_coherent_docs import get_scores,sort_files_by_date,retrieve_initial_documents,ban_non_coherent_docs,get_dataset_stas,get_banned_queries
import os
import numpy as np
import sys
from w2v import train_word2vec as model
from utils import run_bash_command,cosine_similarity
from krovetzstemmer import Stemmer
import params
import math
from Crowdflower.create_unified_experiment import write_histogram_for_weighted_scores

def get_centroid(doc_vectors,decay=False):
    sum_of_vecs = np.zeros(300)
    if decay:
        decay_factors = [0.01*math.exp(-0.01*(len(doc_vectors)-i)) for i in range(len(doc_vectors))]
        denominator = sum(decay_factors)
        for i,doc in enumerate(doc_vectors):

            sum_of_vecs+=(doc*decay_factors[i]/denominator)
        return sum_of_vecs
    for doc in doc_vectors:
        sum_of_vecs+=doc
    return sum_of_vecs/len(doc_vectors)


def get_vectors(top_doc_vectors,decay=False):
    result={}
    winners = {}
    for query in top_doc_vectors:
        vectors = top_doc_vectors[query]
        winners[query]=vectors[0]
        centroid = get_centroid(vectors,decay=decay)
        result[query]=centroid
    return result,winners



def clean_text(text):
    text = text.replace(".", " ")
    text = text.replace("-", " ")
    text = text.replace(",", " ")
    text = text.replace(":", " ")
    text = text.replace("?", " ")
    text = text.replace("$", " ")
    text = text.replace("%", " ")
    text = text.replace("<", " ")
    text = text.replace(">", " ")
    text = text.replace("\\", " ")
    text = text.replace("*", " ")
    text = text.replace(";", " ")
    text = text.replace("`", "")
    text = text.replace("'", "")
    text = text.replace("@", " ")
    text = text.replace("\n", " ")
    text = text.replace("\"", "")
    text = text.replace("/", " ")
    text = text.replace("(", "")
    text = text.replace(")", "")
    return text

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


def get_level(score):
    # demotion_level = 0
    # if score >=0 and score <1:
    #     demotion_level=2
    # elif score >=1 and score <2:
    #     demotion_level=1
    # return demotion_level
    demotion_level = 0
    if score >= 0 and score < 2:
        demotion_level = 2
    elif score >= 2 and score < 4:
        demotion_level = 1
    return demotion_level


def modify_seo_score_by_demotion(seo_scores, coherency_scores):
    new_scores = {}
    for key in seo_scores:
        new_scores[key] = {}
        for id in seo_scores[key]:
            current_score = seo_scores[key][id]
            coherency_score = coherency_scores[key][id]
            demotion_level = get_level(coherency_score)
            new_score = max(current_score-demotion_level,0)
            new_scores[key][id] = new_score
    return new_scores


def create_harmonic_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    for key in seo_scores:
        new_scores[key]={}
        for id in seo_scores[key]:
            epsilon = 0.0001
            current_score = seo_scores[key][id]
            # current_score = seo_scores[key][id]*(3.0/5)
            coherency_score = coherency_scores[key][id]
            new_coherency_score = coherency_score*(4.0/5)
            # new_coherency_score = coherency_score
            numerator = (1+beta**2)*new_coherency_score*current_score
            denominator = (beta**2)*new_coherency_score+current_score
            denominator+=epsilon
            harmonic_mean = float(numerator)/denominator #(2*new_coherency_score*current_score)/(new_coherency_score+current_score)
            new_scores[key][id]=harmonic_mean
    return new_scores

def create_weighted_mean_score(seo_scores,coherency_scores,beta):
    new_scores = {}
    for key in seo_scores:
        new_scores[key] = {}
        for id in seo_scores[key]:
            current_score = seo_scores[key][id]
            # current_score = seo_scores[key][id]*(3.0/5)
            coherency_score = coherency_scores[key][id]
            new_coherency_score = coherency_score * (4.0 / 5)
            # new_coherency_score = coherency_score
            new_score = current_score*beta+new_coherency_score*(1-beta)
            new_scores[key][id]=new_score
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



def get_sentence_vector(sentence,model):
    stemmer = Stemmer()
    sentence = clean_text(sentence)
    words = sentence.split()
    stemmed =[stemmer.stem(w) for w in words]
    return get_stemmed_document_vector(stemmed,model)

def normalize_feature(feature_value,max_min_stats,query,feature):
    if max_min_stats[query][feature]["max"]==max_min_stats[query][feature]["min"]:
        return 0
    denominator =max_min_stats[query][feature]["max"]-max_min_stats[query][feature]["min"]
    value = (feature_value-max_min_stats[query][feature]["min"])/denominator
    return value

def rewrite_fetures(new_scores, old_features_file, new_features_filename,qrels_name):
    f = open(new_features_filename,"w")
    qrels = open(qrels_name,"w")
    with open(old_features_file) as file:
        for line in file:
            qid = line.split()[1]
            query = qid.split(":")[1]
            key=query[3:]
            features = line.split()[2:-2]
            id = line.split(" # ")[1].rstrip()
            if id not in new_scores[key]:
                continue
            new_line=str(new_scores[key][id])+" qid:"+query+" "+" ".join(features)+" # "+id
            f.write(new_line+"\n")
            qrels.write(query+" 0 "+id+" "+str(new_scores[key][id])+"\n")
    f.close()
    qrels.close()







def get_stemmed_document_vector(doc,model):
    vector = np.zeros(300)
    i=1
    for stem in doc:
        if stem in model.wv:
            vector +=model.wv[stem]
        i+=1
    return vector/i

def get_document_vector(doc,model):
    words = doc.split()
    return get_stemmed_document_vector(words,model)

def write_files(values,query,comb):
    for feature in values:
        f = open(feature+"_"+query,'a')
        f.write(comb+" "+str(values[feature])+"\n")
        f.close()


def create_tfidf_features_and_features_file(features_dir,index_path,sentence_file,top_doc_files,input_query,past_winners_file,key):
    query = input_query+key
    command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp indri.jar Main "+index_path+" "+sentence_file+" "+top_doc_files+" "+past_winners_file+" "+query
    print(run_bash_command(command))
    command = "mv doc*_* "+features_dir
    run_bash_command(command)



def create_features_from_dir(features_dir,features_file,sentence_working_set):
    command = "perl " + params.sentence_feature_creator + " " + features_dir + " " + sentence_working_set
    run_bash_command(command)
    command = "mv features " + features_file
    run_bash_command(command)


def feature_values(centroid,s_in,s_out,past_winner_centroid):
    result={}
    result["docCosineToCentroidInVec"]= cosine_similarity(centroid,s_in)
    result["docCosineToCentroidOutVec"]= cosine_similarity(centroid,s_out)
    result["docCosineToWinnerCentroidInVec"]=cosine_similarity(past_winner_centroid,s_in)
    result["docCosineToWinnerCentroidOutVec"]=cosine_similarity(past_winner_centroid,s_out)

    return result


def load_model():
    model_w2v_loader = model.WordToVec()
    return model_w2v_loader.load_model()

def create_coherency_features(sentences_index,ref_doc,input_query,model,key):
    query = input_query+key
    ref_doc_sentences = sentences_index[query][ref_doc]
    for top_doc in sentences_index[query]:
        if top_doc==ref_doc:
            continue
        top_doc_sentences = sentences_index[query][top_doc]
        for i,top_doc_sentence in enumerate(top_doc_sentences,start=1):
            sentence_vec = get_sentence_vector(top_doc_sentence,model)
            for j,ref_sentence in enumerate(ref_doc_sentences):
                row={}
                comb = top_doc+"_"+str(i)+"_"+str(j+1)
                window = []
                if j == 0:
                    # if j+1 == len(ref_doc_sentences):
                    #     window.append(get_sentence_vector(ref_doc_sentences[0], model))
                    #     window.append(get_sentence_vector(ref_doc_sentences[0], model))
                    # else:
                    window.append(get_sentence_vector(ref_doc_sentences[1], model))
                    window.append(get_sentence_vector(ref_doc_sentences[1], model))

                elif j+1 == len(ref_doc_sentences):
                    window.append(get_sentence_vector(ref_doc_sentences[j - 1], model))
                    window.append(get_sentence_vector(ref_doc_sentences[j - 1], model))
                else:
                    window.append(get_sentence_vector(ref_doc_sentences[j - 1], model))
                    window.append(get_sentence_vector(ref_doc_sentences[j + 1], model))
                ref_vector = get_sentence_vector(ref_sentence, model)
                row["docSimilarityToPrev"] = cosine_similarity(sentence_vec, window[0])
                row["docSimilarityToRefSentence"] = cosine_similarity(ref_vector, sentence_vec)
                row["docSimilarityToPred"] = cosine_similarity(sentence_vec, window[1])
                row["docSimilarityToPrevRef"] = cosine_similarity(ref_vector, window[0])
                row["docSimilarityToPredRef"] = cosine_similarity(ref_vector, window[1])
                write_files(row,query,comb)



def read_past_winners_file(winners_file):
    winners_data ={}
    stemmer = Stemmer()
    with open(winners_file) as file:
        for line in file:
            query = line.split("@@@")[0]
            text = line.split("@@@")[1]
            if query not in winners_data:
                winners_data[query]=[]
            text = " ".join([stemmer.stem(word) for word in clean_text(text).split()])
            winners_data[query].append(text)
    return winners_data


def init_doc_ids(doc_ids_file):
    doc_ids={}
    with open(doc_ids_file) as file:
        for line in file:
            doc_ids[line.split("\t")[0]]=line.split("\t")[1]
    return doc_ids



def init_top_doc_vectors(top_docs,doc_ids,model):
    top_docs_vectors={}
    for query in top_docs:
        docs = top_docs[query]
        command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp indri.jar DocStems /home/greg/mergedindex \""+" ".join([doc_ids[d.rstrip()].strip() for d in docs])+"\""
        print(command)
        print(run_bash_command(command))
        top_docs_vectors[query]=[]
        with open("/home/greg/auto_seo/SentenceRanking/docsForVectors") as docs:
            for i,doc in enumerate(docs):
                top_docs_vectors[query].append(get_document_vector(doc,model))
    return top_docs_vectors



def init_past_winners_vectors(winners_data,model):
    winners_vectors = {}
    for query in winners_data:
        winners_vectors[query]=[]
        for i,doc in enumerate(winners_data[query]):
            winners_vectors[query].append(get_document_vector(doc,model))
    return winners_vectors

def get_top_docs_per_query(top_docs_file):
    top_docs ={}
    with open(top_docs_file) as file:
        for line in file:
            query = line.split("\t")[0]
            doc = line.split("\t")[1]
            if query not in top_docs:
                top_docs[query]=[]
            top_docs[query].append(doc)
    return top_docs

def combine_winners(winners,past_winners):
    for query in winners:
        winner_vec = winners[query]
        past_winners[query].append(winner_vec)
    return past_winners


def create_w2v_features(senteces_file,top_docs_file,doc_ids_file,past_winners_file,model,input_query,key):
    query = input_query+key
    top_docs = get_top_docs_per_query(top_docs_file)
    doc_ids = init_doc_ids(doc_ids_file)
    past_winners_data = read_past_winners_file(past_winners_file)
    past_winners_vectors = init_past_winners_vectors(past_winners_data,model)
    top_doc_vectors = init_top_doc_vectors(top_docs,doc_ids,model)
    centroids,winners = get_vectors(top_doc_vectors)
    combine_winners(winners,past_winners_vectors)
    past_winner_centroids,_=get_vectors(past_winners_vectors,True)
    with open(senteces_file) as s_file:
        for line in s_file:
            comb,sentence_in,sentence_out = line.split("@@@")[0],line.split("@@@")[1],line.split("@@@")[2]
            centroid = centroids[query]
            past_winner_centroid = past_winner_centroids[query]
            sentence_vector_in = get_sentence_vector(sentence_in,model)
            sentence_vector_out = get_sentence_vector(sentence_out,model)
            values = feature_values(centroid,sentence_vector_in,sentence_vector_out,past_winner_centroid)
            write_files(values,query,comb)


def create_sentence_file(top_docs_file, ref_doc, query,key,doc_texts):
    sentence_files_dir = "sentence_files/"
    sentences_index={}
    sentences_index[query+key]={}
    if not os.path.exists(sentence_files_dir):
        os.makedirs(sentence_files_dir)
    sentence_filename = sentence_files_dir+"sentence_file_"+ref_doc+"_"+query
    ref_text=doc_texts[ref_doc]
    ref_sentences = retrieve_sentences(ref_text)
    sentences_index[query+key][ref_doc]=ref_sentences
    f = open(sentence_filename,"w")
    with open(top_docs_file) as file:
        for line in file:
            top_doc = line.split("\t")[1].rstrip()
            top_doc_text = doc_texts[top_doc]
            top_doc_sentences = retrieve_sentences(top_doc_text)
            sentences_index[query+key][top_doc]=top_doc_sentences
            for i,top_doc_sentence in enumerate(top_doc_sentences,start=1):
                for j,ref_sentence in enumerate(ref_sentences,start=1):
                    comb_name = top_doc+"_"+str(i)+"_"+str(j)
                    f.write(comb_name+"@@@"+top_doc_sentence.replace("\n","").replace("\r","").rstrip()+"@@@"+ref_sentence.replace("\n","").replace("\r","").rstrip()+"\n")
    f.close()
    return sentence_filename,sentences_index





def create_sentence_working_set(ref_doc,sentence_file,query,key):
    working_set_dir = "sentence_working_set/"
    if not os.path.exists(working_set_dir):
        os.makedirs(working_set_dir)
    working_set_filename = working_set_dir+ref_doc+"_workingset"
    with open(working_set_filename,"w") as working_set_file:
        with open(sentence_file) as file:
            for i,line in enumerate(file,start=1):
                comb = line.split("@@@")[0]
                working_set_file.write(query+key+" Q0 "+comb+" "+str(i)+" "+str(-i)+" seo\n")
    return working_set_filename


def create_top_docs_per_ref_doc(top_docs,key,ref_doc,query):
    top_docs_dir = "top_docs/"
    if not os.path.exists(top_docs_dir):
        os.makedirs(top_docs_dir)
    top_docs_file = top_docs_dir+ref_doc
    f = open(top_docs_file,"w")
    top_docs_names = top_docs[key][query]
    for doc in top_docs_names:
        f.write(query+key+"\t"+doc+"\n")
    f.close()
    return top_docs_file

def get_histogram(dataset):
    hist ={}
    for key in dataset:
        for id in dataset[key]:
            if dataset[key][id]<1:
                bucket =0
            elif dataset[key][id]<2:
                bucket =1
            elif dataset[key][id]<3:
                bucket =2
            elif dataset[key][id]<4:
                bucket =3
            elif dataset[key][id]<5:
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

def create_features(reference_docs,past_winners_file_index,doc_ids_file,index_path,top_docs,doc_text):
    print("loading w2v model")
    model = load_model()
    print("loading done")
    final_features_dir = "sentence_feature_files/"
    features_file = final_features_dir + "new_data_sentence_features"
    features_dir = "sentence_feature_values/"
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
    if not os.path.exists(final_features_dir):
        os.makedirs(final_features_dir)
    total_working_set_file = "total_working_set_file"
    run_bash_command("touch "+total_working_set_file)
    for key in reference_docs:
        past_winners_file=past_winners_file_index[key]
        for query in reference_docs[key]:
            print("working on",query)
            doc = reference_docs[key][query]
            print("working on",doc)
            print("top_doc_file is created")
            top_docs_file = create_top_docs_per_ref_doc(top_docs,key,doc,query)
            sentence_file_name,sentences_index = create_sentence_file(top_docs_file,doc,query,key,doc_text)
            if len(sentences_index[query+key][doc])<2:
                continue
            print("sentence_file is created")
            working_set_file =create_sentence_working_set(doc,sentence_file_name,query,key)
            run_bash_command("cat "+working_set_file+" >> "+total_working_set_file)
            print("sentence working-set is created")
            create_w2v_features(sentence_file_name , top_docs_file,doc_ids_file,past_winners_file,model,query,key)
            print("created seo w2v features")
            create_coherency_features(sentences_index,doc,query,model,key)
            print("created coherency features")
            create_tfidf_features_and_features_file(features_dir,index_path,sentence_file_name,top_docs_file,query,past_winners_file,key)
            print("created tf-idf features")
    print("creating all features")
    create_features_from_dir(features_dir,features_file,total_working_set_file)
    return features_file


def write_tags(tags,filename,key):
    with open(filename,"a") as file:
        for id in tags:
            tag = str(min(5,sum(tags[id])))
            file.write(id+key+" "+tag+"\n")

if __name__=="__main__":
    ranked_lists_new = retrieve_ranked_lists("trec_file04")
    reference_docs={}
    top_docs={}
    reference_docs["45"] = {q: ranked_lists_new[q][-1].replace("EPOCH", "ROUND") for q in ranked_lists_new}
    top_docs["45"]={q: ranked_lists_new[q][:3] for q in ranked_lists_new}
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
    scores={}
    for k in range(6):
        needed_file = sorted_files[k]
        scores = get_scores(scores,dir + "/" + needed_file,original_docs,k+1)
    print(scores)
    banned_queries = get_banned_queries(scores,reference_docs)
    # rounds = ["4","6"]
    rounds = ["6"]
    ranks = ["2","5"]
    past_winners_file_4 ="past_winners_file_new_data04"
    past_winners_file_6 ="past_winners_file_new_data06"
    past_winners_file_index={"65":past_winners_file_6,"62":past_winners_file_6,"45":past_winners_file_4,"42":past_winners_file_4}
    doc_ids_file="docIDs"
    index_path="/home/greg/mergedindex"
    # base_features_file=create_features(reference_docs,past_winners_file_index,doc_ids_file,index_path,top_docs,doc_texts)
    base_features_file = "sentence_feature_files/new_data_sentence_features"
    all_aggregated_results={}
    for r in rounds:
        for rank in ranks:

            ident_filename_mturk = "Mturk/Manipulated_Document_Identification_"+r+"_"+rank+".csv"
            sentence_filename_mturk = "Mturk/Sentence_Identification_"+r+"_"+rank+".csv"
            ident_results = mturk_ds_creator.read_ds_mturk(ident_filename_mturk, True)
            sentence_results = mturk_ds_creator.read_ds_mturk(sentence_filename_mturk)
            sentence_tags = mturk_ds_creator.get_tags(sentence_results)
            ident_tags = mturk_ds_creator.get_tags(ident_results)
            tmp_aggregated_results = mturk_ds_creator.aggregate_results(sentence_tags,ident_tags)
            key = r+rank
            write_tags(ident_tags,"document_identification_tags",key)
            write_tags(sentence_tags,"sentence_identification_tags",key)
            all_aggregated_results[key]=tmp_aggregated_results
    all_aggregated_results = ban_non_coherent_docs(banned_queries,all_aggregated_results)
    seo_scores_file = "labels_new_final_all_data"
    tmp_seo_scores = read_seo_score(seo_scores_file)
    seo_scores = ban_non_coherent_docs(banned_queries,tmp_seo_scores)

    modified_scores= modify_seo_score_by_demotion(seo_scores,all_aggregated_results)
    new_features_with_demotion_file = "all_seo_features_demotion"
    new_qrels_with_demotion_file = "seo_demotion_qrels"

    rewrite_fetures(modified_scores,base_features_file,new_features_with_demotion_file,new_qrels_with_demotion_file)

    stats_harmonic={}
    betas = [0,0.5,1,2,1000,100000,1000000000]
    # betas = [1,]
    flag =False
    flag1 =False
    for beta in betas:
        new_features_with_harmonic_file = "all_seo_features_harmonic_"+str(beta)
        new_qrels_with_harmonic_file = "seo_harmonic_qrels_"+str(beta)
        harmonic_mean_scores={}
        harmonic_mean_scores = create_harmonic_mean_score(seo_scores,all_aggregated_results,beta)
        rewrite_fetures(harmonic_mean_scores, base_features_file, new_features_with_harmonic_file,new_qrels_with_harmonic_file)
        harmonic_hist = get_histogram(harmonic_mean_scores)
        last = False
        if beta == betas[-1]:
            last = True
        flag = True
        write_histogram_for_weighted_scores(harmonic_hist, "harmonic_histogram.tex", beta, flag1, last)
        flag1 = True


    flag=False
    flag1=False
    stats_weighted = {}
    # betas = [0,]
    betas = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    for beta in betas:
        new_features_with_weighted_file = "all_seo_features_weighted_"+str(beta)
        new_qrels_with_weighted_file = "seo_weighted_qrels_"+str(beta)
        weighted_mean_scores={}
        weighted_mean_scores = create_weighted_mean_score(seo_scores, all_aggregated_results,beta)
        rewrite_fetures(weighted_mean_scores,base_features_file, new_features_with_weighted_file,new_qrels_with_weighted_file)
        last = False
        if beta == betas[-1]:
            last = True
        flag = True
        weighted_hist = get_histogram(weighted_mean_scores)
        write_histogram_for_weighted_scores(weighted_hist, "weighted_histogram.tex", beta, flag1, last)
        flag1 = True

    # print("queries=", len(get_dataset_stas(all_aggregated_results)))
    # print("examples=", len(aggregated_results))
    # print("seo_examples=", len(seo_scores))
    # print("histogram_coherency", get_histogram(all_aggregated_results))
    # print("histogram_demotion", get_histogram(modified_scores))
    # print("histogram_scores_lables", get_histogram(seo_scores))