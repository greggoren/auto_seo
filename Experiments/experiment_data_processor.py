import params
import os
from utils import run_bash_command


def create_features_file():
    command = "perl "+params.features_generator_script_path+" "+params.features_dir+" "+params.working_set_file
    run_bash_command(command)


def create_trectext(document_text,reference_docs,summaries):
    f= open(params.new_trec_text_file,"w")
    query_to_docs = {}
    for document in document_text:
        if document in reference_docs:
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

    workingSetFilename = params.working_set_file
    f = open(workingSetFilename, 'w')
    for query, docnos in query_to_docs.items():
        i = 1
        for docid in docnos:
            f.write(query.zfill(3) + ' Q0 ' + docid + ' ' + str(i) + ' -' + str(i) + ' indri\n')
            i += 1

    f.close()

def create_index():
    """
    Parse the trectext file given, and create an index.
    """
    path_to_folder = '/lv_local/home/sgregory/auto_seo'
    indri_build_index = '/lv_local/home/sgregory/indri11/bin/IndriBuildIndex'
    corpus_path = params.new_trec_text_file
    corpus_class = 'trectext'
    memory = '1G'
    index = path_to_folder+"/index/new_index"
    stemmer =  'krovetz'
    os.popen('mkdir -p ' + path_to_folder)
    os.popen(indri_build_index + ' -corpus.path='+corpus_path + ' -corpus.class='+corpus_class + ' -index='+index + ' -memory='+memory + ' -stemmer.name=' + stemmer).readlines()
    return index

def merge_indices(new_index):
    path_to_folder = '/lv_local/home/sgregory/auto_seo'
    command = '/lv_local/home/sgregory/indri11/bin/dumpindex '+path_to_folder+'/new_merged_index merge '+new_index+" "+params.corpus_path
    run_bash_command(command)
