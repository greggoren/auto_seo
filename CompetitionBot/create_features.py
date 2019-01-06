from CompetitionBot.get_data_for_feature_creation import create_top_docs_per_ref_doc,create_former_winners_file,create_sentence_file,create_sentence_working_set
from pymongo import MongoClient
import numpy as np
from krovetzstemmer import Stemmer
from w2v import train_word2vec as model
from utils import run_bash_command
import math
import os
import params
from Experiments.model_handler import retrieve_scores
from Experiments.model_handler import create_index_to_doc_name_dict
from CrossValidationUtils.svm_handler import svm_handler
from CrossValidationUtils.evaluator import eval
import datetime


ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017


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


def load_model():
    model_w2v_loader = model.WordToVec()
    return model_w2v_loader.load_model()

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


def get_sentence_vector(sentence,model):
    stemmer = Stemmer()
    sentence = clean_text(sentence)
    words = sentence.split()
    stemmed =[stemmer.stem(w) for w in words]
    return get_stemmed_document_vector(stemmed,model)


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


def cosine_similarity(v1,v2):
    sumxy = v1.dot(v2.T)
    sumxx = np.linalg.norm(v1)
    sumyy = np.linalg.norm(v2)
    denominator = sumxx*sumyy
    if denominator==0:
        return 0
    return sumxy/denominator





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
        command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp /home/greg/auto_seo/scripts/indri.jar DocStems ~/mergedindex \""+" ".join([doc_ids[d.rstrip()].strip() for d in docs])+"\""
        print(command)
        print(run_bash_command(command))
        top_docs_vectors[query]=[]
        with open("/home/greg/auto_seo/SentenceRanking/docsForVectors") as docs:
            for i,doc in enumerate(docs):
                top_docs_vectors[query].append(get_document_vector(doc,model))
    return top_docs_vectors


def get_vectors(top_doc_vectors,decay=False):
    result={}
    winners = {}
    for query in top_doc_vectors:
        vectors = top_doc_vectors[query]
        winners[query]=vectors[0]
        centroid = get_centroid(vectors,decay=decay)
        result[query]=centroid
    return result,winners

def combine_winners(winners,past_winners):
    for query in winners:
        winner_vec = winners[query]
        past_winners[query].append(winner_vec)
    return past_winners


def create_w2v_features(senteces_file,top_docs_file,doc_ids_file,past_winners_file,model,query):
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




def create_bot_models_index():
    model_index = {"harmonic":"models/harmonic_model_0.1","demotion":"models/demotion_model_0.1","weighted":"models/weighted_model_0.01"}
    return model_index


def write_files(values,query,comb):
    for feature in values:
        f = open(feature+"_"+query,'a')
        f.write(comb+" "+str(values[feature])+"\n")
        f.close()


def create_tfidf_features_and_features_file(sentence_working_set,features_file,features_dir,index_path,sentence_file,top_doc_files,query):
    command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp indri.jar Main "+index_path+" "+sentence_file+" "+top_doc_files+" "+sentence_working_set+" "+query
    print(run_bash_command(command))
    command = "mv doc*_* "+features_dir
    run_bash_command(command)
    command = "perl " + params.sentence_feature_creator + " "+features_dir+" " + sentence_working_set
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


def init_past_winners_vectors(winners_data,model):
    winners_vectors = {}
    for query in winners_data:
        winners_vectors[query]=[]
        for i,doc in enumerate(winners_data[query]):
            winners_vectors[query].append(get_document_vector(doc,model))
    return winners_vectors




def write_reference_doc_file(current_time,reference_docs):
    f = open("reference_docs_"+current_time,"w")
    for q in reference_docs:
        f.write(q+" "+reference_docs[q]+"\n")
    f.close()




def get_reference_documents():
    reference_docs={}
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    docs = db.documents.find({})
    for doc in docs:
        if "bot_method" in doc:
            query = doc["query_id"]
            if query not in reference_docs:
                reference_docs[query]=[]
            reference_docs[query].append(doc["username"])
    return reference_docs



def create_coherency_features(sentences_index,ref_doc,query,model):
    ref_doc_sentences = sentences_index[query][ref_doc]
    for top_doc in sentences_index[query]:
        top_doc_sentences = sentences_index[query][top_doc]
        for i,top_doc_sentence in enumerate(top_doc_sentences,start=1):
            sentence_vec = get_sentence_vector(top_doc_sentence,model)
            for j,ref_sentence in enumerate(ref_doc_sentences,start=1):
                row={}
                comb = top_doc+"_"+str(i)+"_"+str(j)
                window = []
                if j == 0:
                    window.append(get_sentence_vector(ref_doc_sentences[1], model))
                    window.append(get_sentence_vector(ref_doc_sentences[1], model))

                elif j + 1 == len(ref_doc_sentences):
                    window.append(get_sentence_vector(ref_doc_sentences[i - 1], model))
                    window.append(get_sentence_vector(ref_doc_sentences[i - 1], model))
                else:
                    window.append(get_sentence_vector(ref_doc_sentences[i - 1], model))
                    window.append(get_sentence_vector(ref_doc_sentences[i + 1], model))
                ref_vector = get_sentence_vector(ref_sentence, model)
                row["docSimilarityToPrev"] = cosine_similarity(sentence_vec, window[0])
                row["docSimilarityToRefSentence"] = cosine_similarity(ref_vector, sentence_vec)
                row["docSimilarityToPred"] = cosine_similarity(sentence_vec, window[1])
                row["docSimilarityToPrevRef"] = cosine_similarity(ref_vector, window[0])
                row["docSimilarityToPredRef"] = cosine_similarity(ref_vector, window[1])
                write_files(row,query,comb)

def create_trec_eval_file(doc_name_index, results, prefix, current_time):
    trec_dir = "sentence_scores_dir/"+current_time+"/"
    if not os.path.exists(trec_dir):
        os.makedirs(trec_dir)
    trec_file = trec_dir+prefix + "_scores.txt"
    trec_file_access = open(trec_file, 'w')
    for index in results:
        doc_name = doc_name_index[index]
        query = doc_name.split("-")[2]
        trec_file_access.write(query + " Q0 " + doc_name + " " + str(0) + " " + str(results[index]) + " seo\n")
    trec_file_access.close()
    return trec_file

def run_svm_model(feature_file, model_file,doc_name_index,query,ref_doc,current_time):
    svm = svm_handler()
    evaluator = eval(["map", "ndcg", "P.2", "P.5"])
    scores_file = svm.run_svm_rank_model(feature_file, model_file, query+"_"+ref_doc)
    results = retrieve_scores(scores_file)
    trec_file = create_trec_eval_file(doc_name_index, results, query+"_"+ref_doc,current_time)
    final_trec_file = evaluator.order_trec_file(trec_file)
    return final_trec_file


def pick_best_sentence_pair(trec_file):
    with open(trec_file) as file:
        for line in file:
            comb = line.split()[2]
            return comb



def replace_sentences_and_save_doc(ref_doc,query,sentence_in,sentence_out):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    doc = next(db.documents.find({"query_id":query,"username":ref_doc}))
    text = doc["posted_document"]
    new_text = text.replace(sentence_out,sentence_in)
    doc["current_document"]=new_text
    db.documents.save(doc)

def get_sentences_for_replacement(comb,sentences_index,ref_doc,query):
    ref_doc_sentences = sentences_index[query][ref_doc]
    top_doc = comb.split("_")[0]
    top_doc_sentence_index = int(comb.split("_")[1])-1
    replacement_index = int(comb.split("_")[2])-1
    sentence_in  = sentences_index[query][top_doc][top_doc_sentence_index]
    sentence_out = ref_doc_sentences[replacement_index]
    return sentence_in,sentence_out

def create_features_for_doc_and_run_model(reference_docs,current_time,past_winners_file,doc_ids_file,model_index,index_path):
    print("loading w2v model")
    model = load_model()
    print("loading done")
    for query in reference_docs:
        print("working on",query)
        for doc in reference_docs[query]:
            print("working on",doc)
            top_docs_file = create_top_docs_per_ref_doc(current_time,doc,query)
            print("top_doc_file is created")
            sentence_file_name,sentences_index = create_sentence_file(top_docs_file,doc,query,current_time)
            print("sentence_file is created")
            working_set_file =create_sentence_working_set(doc,current_time,sentence_file_name,query)
            print("sentence working-set is created")
            create_w2v_features(sentence_file_name,top_docs_file,doc_ids_file,past_winners_file,model,query)
            print("created seo w2v features")
            create_coherency_features(sentences_index,doc,query,model)
            print("created coherency features")
            final_features_dir = "sentence_feature_files/"+current_time+"/"

            features_file = final_features_dir+query+"_"+doc+"_"+current_time
            features_dir = "sentence_feature_values/"+current_time+"/"+query+"_"+doc+"/"
            if not os.path.exists(features_dir):
                os.makedirs(features_dir)
            if not os.path.exists(final_features_dir):
                os.makedirs(final_features_dir)
            create_tfidf_features_and_features_file(working_set_file,features_file,features_dir,index_path,sentence_file_name,top_docs_file,query)
            print("created tf-idf features")
            model_file = model_index[query+"_"+doc]

            doc_name_index = create_index_to_doc_name_dict(features_file)
            print("created doc name index")
            trec_file = run_svm_model(features_file,model_file,doc_name_index,query,doc,current_time)
            print("ran seo model")
            best_comb = pick_best_sentence_pair(trec_file)
            sentence_in,sentence_out = get_sentences_for_replacement(best_comb,sentences_index,doc,query)
            # replace_sentences_and_save_doc(doc,query,sentence_in,sentence_out)
            #print("replaced sentences")
if __name__=="__main__":
    current_time = str(datetime.datetime.now()).replace(":", "-").replace(" ", "-").replace(".", "-")
    doc_ids = "docIDs"
    model_index= create_bot_models_index()
    reference_docs = get_reference_documents()
    past_winners_file = create_former_winners_file(current_time)
    index_path = "/home/greg/ASR18/Collections/mergedindex"
    create_features_for_doc_and_run_model(reference_docs,current_time,past_winners_file,doc_ids,model_index,index_path)