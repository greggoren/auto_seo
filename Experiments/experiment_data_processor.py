import params
import os
from utils import run_bash_command
import sys
import time

def create_features_file(features_dir,index_path,queries_file,new_features_file,add_remove_file,run_name,working_set):
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    # command= params.ltr_features_script+" "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile='+ params.working_set_file+run_name + ' -workingSetFormat=trec'
    command = " java -Djava.library.path=/home/greg/indri-5.6/swig/obj/java/ -cp /home/greg/auto_seo/scripts/indri.jar LTRFeaturesCreator "+add_remove_file+" "+working_set
    print(command)
    out = run_bash_command(command)
    print(out)
    # command=params.cent_script+' ' + queries_file + ' -index=' + index_path + ' -useWorkingSet=true -workingSetFile='+ params.working_set_file+run_name + ' -workingSetFormat=trec'
    # print(command)
    # out = run_bash_command(command)
    # print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl "+params.features_generator_script_path+" "+features_dir+" "+working_set+" "+run_name
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)



def create_features_file_sentence_exp(features_dir,index_path,queries_file,new_features_file,working_set):
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    command= params.ltr_features_script+" "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile='+ working_set + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    command=params.cent_script+' ' + queries_file + ' -index=' + index_path + ' -useWorkingSet=true -workingSetFile='+ working_set + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl "+params.features_generator_script_path+" "+features_dir+" "+working_set
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)




def create_features_file_original(features_dir,index_path,queries_file,new_features_file,run_name=""):
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    command= "/home/greg/auto_seo/past_winners/LTRFeatures "+ queries_file + ' -stream=doc -index=' + index_path + ' -repository='+ index_path +' -useWorkingSet=true -workingSetFile=/home/greg/auto_seo/SentenceRanking/working_set'+run_name + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    # command='/home/greg/auto_seo/past_winners/Cent ' + queries_file + ' -index=' + index_path + ' -useWorkingSet=true -workingSetFile=/home/greg/auto_seo/SentenceRanking/working_set'+run_name + ' -workingSetFormat=trec'
    # print(command)
    # out = run_bash_command(command)
    # print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl /home/greg/auto_seo/past_winners/generate.pl "+features_dir+" /home/greg/auto_seo/SentenceRanking/working_set"+run_name
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features "+new_features_file
    print(command)
    out = run_bash_command(command)
    print(out)



def create_trectext_original(document_text, summaries, run_name="", avoid=[], write_doc=""):
    trec_text_file = "/home/greg/auto_seo/SentenceRanking/trectext"+run_name
    f= open(trec_text_file,"w",encoding="utf-8")
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
    workingSetFilename = "/home/greg/auto_seo/SentenceRanking/working_set"+run_name
    f = open(workingSetFilename, 'w')
    for query, docnos in query_to_docs.items():
        i = 1
        for docid in docnos:
            if docid not in avoid:
                f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
                i += 1

    f.close()
    return trec_text_file


def create_trectext(document_text, summaries,trec_text_name,working_set_name):
    f= open(trec_text_name,"w",encoding="utf-8")
    query_to_docs = {}
    for document in document_text:

        if document in summaries:
            text = summaries[document]
        else:
            text = document_text[document]
        query = document.split("-")[2]
        if not query_to_docs.get(query,False):
            query_to_docs[query]=[]
        query_to_docs[query].append(document)

        f.write('<DOC>\n')
        f.write('<DOCNO>' + document + '</DOCNO>\n')
        f.write('<TEXT>\n')
        f.write(text.rstrip())
        f.write('\n</TEXT>\n')
        f.write('</DOC>\n')
    f.close()
    f = open(working_set_name, 'w')
    for query, docnos in query_to_docs.items():
        i = 1
        for docid in docnos:
            f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
            i += 1

    f.close()
    return trec_text_name

def create_working_sets_by_round(doc_text,working_set_base_file_name):
    doc_index_per_round ={}
    working_sets = []
    for doc in doc_text:
        doc_round = doc.split("-")[1]
        query = doc.split("-")[2]

        if doc_round not in doc_index_per_round:
            doc_index_per_round[doc_round]={}
        if query not in doc_index_per_round[doc_round]:
            doc_index_per_round[doc_round][query]=[]
        doc_index_per_round[doc_round][query].append(doc)
    for doc_round in doc_index_per_round:
        filename = working_set_base_file_name+"_"+doc_round
        working_sets.append(filename)
        f = open(filename,"w")
        for query in doc_index_per_round[doc_round]:
            for i,doc in enumerate(doc_index_per_round[doc_round][query],start=1):
                f.write(query.zfill(3)+ ' Q0 ' + doc + ' ' + str(i) + ' -' + str(i) + ' indri\n')
        f.close()
    return working_sets


# def create_trectext(document_text, summaries, run_name="", avoid=[], write_doc=""):
#     trec_text_file = params.new_trec_text_file+run_name
#     f= open(params.new_trec_text_file+run_name,"w",encoding="utf-8")
#     query_to_docs = {}
#     for document in document_text:
#         if document in avoid:
#             continue
#         if document in summaries:
#             text = summaries[document]
#         else:
#             text = document_text[document]
#         query = document.split("-")[2]
#         if not query_to_docs.get(query,False):
#             query_to_docs[query]=[]
#         query_to_docs[query].append(document)
#         if write_doc==document or write_doc=="":
#             f.write('<DOC>\n')
#             f.write('<DOCNO>' + document + '</DOCNO>\n')
#             f.write('<TEXT>\n')
#             f.write(text.rstrip())
#             f.write('\n</TEXT>\n')
#             f.write('</DOC>\n')
#     f.close()
#     workingSetFilename = params.working_set_file+run_name
#     f = open(workingSetFilename, 'w')
#     for query, docnos in query_to_docs.items():
#         i = 1
#         for docid in docnos:
#             if docid not in avoid:
#                 f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
#                 i += 1
#
#     f.close()
#     return trec_text_file

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
    stemmer =  'krovetz'
    os.popen('mkdir -p ' + path_to_folder)
    if not os.path.isdir(path_to_folder+"/index/"):
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


def merge_indexes_for_experiments(index1, index2, merged_index):
    if os.path.isdir(merged_index):
        print("merged index exists, deleting the index")
        run_bash_command("rm -r "+merged_index)
        print("deletion of old merged index is done")
    command = '/home/greg/indri_test/bin/dumpindex ' + merged_index + ' merge ' + index1 + ' ' + index2
    print("merging command:",command)
    sys.stdout.flush()
    out=run_bash_command(command)
    print("merging out command:",out)
    return merged_index



def merge_indices(new_index,run_name="",new_index_name=""):
    path_to_folder = '/home/greg/auto_seo'
    if new_index_name=="":
        new_index_name = path_to_folder+'/new_merged_index'+run_name
    # print("deleting old merged index repository")
    # command = "rm -r "+path_to_folder+'/new_merged_index*'
    # print("delete command = ",command)
    # run_bash_command(command)
    # print("delete finished")
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




