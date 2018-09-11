import numpy as np
from krovetzstemmer import Stemmer
from w2v import train_word2vec as model
from utils import run_bash_command
import params
from CrossValidationUtils.cross_validation import cross_validation
import sys



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

def get_sentence_vector(sentence,model):
    stemmer = Stemmer()
    words = sentence.split()
    stemmed =[stemmer.stem(w) for w in words]
    return get_stemmed_document_vector(stemmed,model)


def get_centroid(doc_vectors):
    sum = np.zeros(300)
    for doc in doc_vectors:
        sum+=doc
    return sum/len(doc_vectors)


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
            for doc in docs:
                top_docs_vectors[query].append(get_document_vector(doc,model))
    return top_docs_vectors


def get_vectors(top_doc_vectors):
    result={}
    winners = {}
    for query in top_doc_vectors:
        vectors = top_doc_vectors[query]
        winners[query]=vectors[0]
        centroid = get_centroid(vectors)
        result[query]=centroid
    return result,winners

def create_features(senteces_file,top_docs_file,doc_ids_file,past_winners_file,model):
    top_docs = get_top_docs_per_query(top_docs_file)
    doc_ids = init_doc_ids(doc_ids_file)
    past_winners_data = read_past_winners_file(past_winners_file)
    past_winners_vectors = init_past_winners_vectors(past_winners_data,model)
    top_doc_vectors = init_top_doc_vectors(top_docs,doc_ids,model)
    centroids,winners = get_vectors(top_doc_vectors)
    past_winner_centroids,_ =get_vectors(past_winners_vectors)
    with open(senteces_file) as s_file:
        for line in s_file:
            comb,sentence_in,sentence_out = line.split("\t")[0],line.split("\t")[1],line.split("\t")[2]
            query = comb.split("-")[2]
            centroid = centroids[query]
            winner =winners[query]
            sentence_vector_in = get_sentence_vector(sentence_in,model)
            sentence_vector_out = get_sentence_vector(sentence_out,model)
            values = feature_values(centroid,sentence_vector_in,sentence_vector_out,winner)
            write_files(values,query,comb)





def write_files(values,query,comb):
    for feature in values:
        f = open(feature+"_"+query,'a')
        f.write(comb+" "+str(values[feature])+"\n")
        f.close()


def feature_values(centroid,s_in,s_out,winner,past_winner_centroid):
    result={}
    result["docCosineToCentroidInVec"]= cosine_similarity(centroid,s_in)
    result["docCosineToCentroidOutVec"]= cosine_similarity(centroid,s_out)
    result["docCosineToWinnerInVec"]=cosine_similarity(winner,s_in)
    result["docCosineToWinnerOutVec"]=cosine_similarity(winner,s_out)
    result["docCosineToWinnerCentroidInVec"]=cosine_similarity(past_winner_centroid,s_in)
    result["docCosineToWinnerCentroidOutVec"]=cosine_similarity(past_winner_centroid,s_out)

    return result


def read_past_winners_file(winners_file):
    winners_data ={}
    stemmer = Stemmer()
    with open(winners_file) as file:
        for line in file:
            query = line.split("\t")[0]
            text = list.split("\t")[1]
            if query not in winners_data:
                winners_data[query]=[]
            text = " ".join([stemmer.stem(word) for word in text.split()])
            winners_data[query].append(text)
    return winners_data


def init_past_winners_vectors(winners_data,model):
    winners_vectors = {}
    for query in winners_data:
        winners_vectors[query]=[]
        for doc in winners_data[query]:
            winners_vectors[query].append(get_document_vector(doc,model))
    return winners_vectors


def add_labeles(label_file_path,old_features,new_features_path):
    label_file = open(label_file_path)
    labels = {line.split()[2]:line.split()[3].replace("\n","") for line in label_file}
    label_file.close()
    new_features = open(new_features_path,"w")
    with open(old_features) as features:
        for line in features:
            splited = line.split()
            label = labels[splited[-1].rstrip()]
            new_line = label+" "+" ".join(splited[1:])
            new_features.write(new_line+"\n")
        new_features.close()
        return new_features_path







if __name__=="__main__":
    qrels =sys.argv[1]
    sentences_file = "/home/greg/auto_seo/scripts/senetces_add_remove"
    top_docs_file= "/home/greg/auto_seo/scripts/topDocs"
    doc_ids_file = "/home/greg/auto_seo/scripts/docIDs"
    model_w2v =load_model()
    features_path = "sentence_features"
    create_features(sentences_file,top_docs_file,doc_ids_file,model_w2v)
    command = "~/jdk1.8.0_181/bin/java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp /home/greg/auto_seo/scripts/indri.jar Main"
    print(run_bash_command(command))
    command= "mkdir vectorFeatures"
    run_bash_command(command)
    command ="mv doc*_* vectorFeatures"
    run_bash_command(command)
    command = "perl "+params.sentence_feature_creator+" vectorFeatures "+params.sentence_working_set
    run_bash_command(command)
    command = "mv features "+features_path
    run_bash_command(command)
    new_features = add_labeles(qrels,features_path,"new_sentence_features")
    cross_validation(new_features,qrels,"summary_all_features.tex")
