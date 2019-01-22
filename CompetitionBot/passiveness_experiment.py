from CompetitionBot.create_ds_for_annotations import get_reference_documents
from pymongo import MongoClient
from Experiments.experiment_data_processor import create_index,merge_indices
import os
from utils import run_bash_command



ASR_MONGO_HOST = "asr2.iem.technion.ac.il"
ASR_MONGO_PORT = 27017

def get_original_documents(reference_docs):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    text_index = {}
    iterations = db.archive.distinct("iteration")
    iterations = sorted(iterations)
    first_iter = iterations[7]
    for query_id in reference_docs:
        for doc in reference_docs[query_id]:
            doc = next(db.archive.find({"username":doc,"query_id":query_id,"iteration":first_iter}))
            text = doc["text"]
            text_index[query_id+"_"+doc]=text
    return text_index


def create_trec_text(text_index):
    client = MongoClient(ASR_MONGO_HOST, ASR_MONGO_PORT)
    db = client.asr16
    iterations = db.archive.distinct("iteration")
    iterations = sorted(iterations)
    needed_iter = iterations[8]
    docs = db.archive.find({"iteration":needed_iter})
    trec_file = "trec_text_bots_passive"
    f = open(trec_file,"w")
    working_set_index = {}
    for doc in docs:
        query = doc["query_id"]
        if query not in working_set_index:
            working_set_index[query]=[]
        username = doc["username"]
        working_set_index[query].append(username)
        key = query+"_"+username
        group = query.split("_")[1]
        if group not in ["0" , "2"]:
            continue
        if key in text_index:
            text = text_index[key]
        else:
            text = doc["text"]
        f.write('<DOC>\n')
        f.write('<DOCNO>' + query+"-"+username + '</DOCNO>\n')
        f.write('<TEXT>\n')
        f.write(text.rstrip())
        f.write('\n</TEXT>\n')
        f.write('</DOC>\n')
    f.close()
    working_set_file = "working_set_passive_bots"
    f = open(working_set_file,"w")
    for query in working_set_index:
        for i,doc in enumerate(working_set_index[query],start=1):
            f.write(query.zfill(3) + ' Q0 ' + doc + ' ' + str(i) + ' -' + str(i) + ' indri\n')
    f.close()
    return trec_file,working_set_file


def create_index(trec_text_file):
    """
    Parse the trectext file given, and create an index.
    """
    path_to_folder = '/lv_local/home/sgregory/Bots/'
    indri_build_index = '/lv_local/home/sgregory/indri/bin/IndriBuildIndex'
    corpus_path = trec_text_file
    corpus_class = 'trectext'
    memory = '1G'
    index = path_to_folder+"/index/new_index"
    stemmer =  'krovetz'
    os.popen('mkdir -p ' + path_to_folder)
    if not os.path.isdir(path_to_folder+"/index/"):
        os.makedirs(path_to_folder+"/index/")
    command = indri_build_index + ' -corpus.path=' + corpus_path + ' -corpus.class=' + corpus_class + ' -index=' + index + ' -memory=' + memory + ' -stemmer.name=' + stemmer
    print(command)
    out=run_bash_command(command)
    print(out)
    return index


def merge_indices(new_index,new_index_name):
    path_to_folder = '/lv_local/home/sgregory/Bots/'
    command = '/lv_local/home/sgregory/indri/bin/dumpindex '+new_index_name+' merge '+new_index+' '+'/lv_local/home/sgregory/cluewebindex'
    print("merging command:",command)
    out=run_bash_command(command)
    print("merging out command:",out)
    return new_index_name



if __name__=="__main__":
    ref_docs = get_reference_documents()
    original_bot_text = get_original_documents(ref_docs)
    trectext_file,working_set = create_trec_text(original_bot_text)
    new_index =create_index(trectext_file)
    merge_indices(new_index,'/lv_local/home/sgregory/Bots/mergedindices')

