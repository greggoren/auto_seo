import params
import os
from utils import run_bash_command
import sys
import time

def create_features_file(features_dir,index_path,queries_file,new_features_file,run_name=""):
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # command= params.ltr_features_script+" "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile='+ params.working_set_file+run_name + ' -workingSetFormat=trec'
    command = " java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp /home/greg/auto_seo/scripts/indri.jar LTRFeaturesCreator"
    print(command)
    out = run_bash_command(command)
    print(out)
    command=params.cent_script+' ' + queries_file + ' -index=' + index_path + ' -useWorkingSet=true -workingSetFile='+ params.working_set_file+run_name + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl "+params.features_generator_script_path+" "+features_dir+" "+params.working_set_file+run_name
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)

def create_trectext(document_text, summaries, run_name="", avoid=[], write_doc=""):
    trec_text_file = params.new_trec_text_file+run_name
    f= open(params.new_trec_text_file+run_name,"w",encoding="utf-8")
    query_to_docs = {}
    for document in document_text:
        if document in avoid:
            continue
        if document in summaries:
            text = summaries[document]
        else:
            text = document_text[document]
        query = document.split("-")[2]
        if not query_to_docs.get(query,False):
            query_to_docs[query]=[]
        query_to_docs[query].append(document)
        if write_doc==document or write_doc=="":
            f.write('<DOC>\n')
            f.write('<DOCNO>' + document + '</DOCNO>\n')
            f.write('<TEXT>\n')
            f.write(text.rstrip())
            f.write('\n</TEXT>\n')
            f.write('</DOC>\n')
    f.close()
    workingSetFilename = params.working_set_file+run_name
    f = open(workingSetFilename, 'w')
    for query, docnos in query_to_docs.items():
        i = 1
        for docid in docnos:
            if docid not in avoid:
                f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
                i += 1

    f.close()
    return trec_text_file

def create_index(trec_text_file,run_name=""):
    """
    Parse the trectext file given, and create an index.
    """
    path_to_folder = '/home/greg/auto_seo'
    indri_build_index = '/home/greg/indri_test/bin/IndriBuildIndex'
    corpus_path = trec_text_file
    corpus_class = 'trectext'
    memory = '1G'
    index = path_to_folder+"/index/new_index"+run_name
    # if not os.path.exists(path_to_folder+"/index/"):
    #     os.makedirs(path_to_folder+"/index/")
    stemmer =  'krovetz'
    os.popen('mkdir -p ' + path_to_folder)
    if not os.path.exists(path_to_folder+"/index/"):
        os.makedirs(path_to_folder+"/index/")
    command = indri_build_index + ' -corpus.path=' + corpus_path + ' -corpus.class=' + corpus_class + ' -index=' + index + ' -memory=' + memory + ' -stemmer.name=' + stemmer
    print(command)
    out=run_bash_command(command)
    print(out)
    return index

def add_docs_to_index(index,run_name=""):
    """
    Parse the trectext file given, and create an index.
    """
    path_to_folder = '/lv_local/home/sgregory/auto_seo'
    indri_build_index = '/lv_local/home/sgregory/indri_test/bin/IndriBuildIndex'
    corpus_path = params.new_trec_text_file+run_name
    corpus_class = 'trectext'
    memory = '1G'
    stemmer =  'krovetz'
    os.popen('mkdir -p ' + path_to_folder)
    if not os.path.exists(path_to_folder+"/index/"):
        os.makedirs(path_to_folder+"/index/")
    command = indri_build_index + ' -corpus.path=' + corpus_path + ' -corpus.class=' + corpus_class + ' -index=' + index + ' -memory=' + memory + ' -stemmer.name=' + stemmer
    print(command)
    out=run_bash_command(command)
    print(out)
    return index



def merge_indices(new_index,run_name="",new_index_name=""):
    path_to_folder = '/home/greg/auto_seo'
    if new_index_name=="":
        new_index_name = path_to_folder+'/new_merged_index'+run_name
    print("deleting old merged index repository")
    command = "rm -r "+path_to_folder+'/new_merged_index*'
    print("delete command = ",command)
    run_bash_command(command)
    print("delete finished")
    command = '/home/greg/indri_test/bin/dumpindex '+new_index_name+' merge '+new_index+' '+params.corpus_path_56
    print("merging command:",command)
    sys.stdout.flush()
    out=run_bash_command(command)
    print("merging out command:",out)
    # run_command(command)
    return new_index_name



def delete_doc_from_index(index,doc,dic,run_name=""):
    did=dic[doc]
    command = '/lv_local/home/sgregory/indri_test/bin/dumpindex '+index+' delete '+did
    print("deleting command:",command)
    sys.stdout.flush()
    out=run_bash_command(command)
    print("deleting out command:",out)




def wait_for_feature_file_to_be_deleted(feature_file):
    while os.path.isfile(feature_file):
        time.sleep(10)
        print("waiting for other procceses to finish")


def move_feature_file(feature_file,run_name):
    command = 'mv '+feature_file+' '+feature_file+run_name
    run_bash_command(command)
    print("feature file moved")




