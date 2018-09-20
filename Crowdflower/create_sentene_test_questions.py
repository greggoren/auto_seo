from Preprocess.preprocess import retrieve_ranked_lists,load_file,get_queries_data
from Preprocess.preprocess import retrieve_sentences
from SentenceRanking.sentence_parse import  map_set_of_sentences
import csv
import random

stupid_sentence_pull = ["Endeavor bachelor but add eat pleasure doubtful sociable"," Shutters ye marriage to throwing we as. Effect in if agreed he wished wanted admire expect","In friendship diminution instrument so","Betrayed cheerful declared end and","Give lady of they such they sure it","Me contained explained my education","Started earnest brother believe an exposed so","Continued at up to zealously necessary breakfast","Separate entrance welcomed sensible laughing why one moderate shy. We seeing piqued garden he","Enjoyed minutes related as at on on",
                        "Her and effects affixed pretend account ten natural","In raptures building an bringing be",
                        "Up unpacked friendly ecstatic so possible humoured do","Expenses as material breeding insisted building to in",
                        "His followed carriage proposal entrance directly had elegance","Drawings me opinions returned absolute in",
                        "Boy desirous families prepared gay reserved add ecstatic say","Looking started he up perhaps against",
                        "Families blessing he in to no daughter","She evil face fine calm have now","An do on frankness so cordially immediate recommend contained"
                        ,"Was drawing natural fat respect husband","Two before narrow not relied how except moment myself","Dejection assurance mrs led certainly","Want name any wise are able park when",
                        "How daughters not promotion few knowledge contented","Am immediate unwilling of attempted admitting disposing it","In as of whole as match asked",
                        "Built purse maids cease her ham new seven among and","Windows talking painted pasture yet its express parties use","Cause dried no solid no an small so still widen"]

def convert_text_to_sentence_task(text):
    sentences = retrieve_sentences(text)
    new_text =""
    for j in range(len(sentences)):
        new_text+=str(j+1)+") "+sentences[j].replace("\n","")+"\n"
    return new_text


def convert_sentences_to_sentence_task(sentences):
    new_text =""
    for j in range(len(sentences)):
        new_text+=str(j+1)+") "+sentences[j].replace("\n","")+"\n"
    return new_text


ranked_lists = retrieve_ranked_lists("trec_file")
query_data=get_queries_data("topics.full.xml")
reference_docs = {q:ranked_lists[q][-1].replace("EPOCH","ROUND") for q in ranked_lists}
winner_docs = {q:ranked_lists[q][:3] for q in ranked_lists}
a_doc_texts = load_file("documents.trectext")
doc_texts={}
for doc in a_doc_texts:
    if doc.__contains__("ROUND-00"):
        doc_texts[doc]=a_doc_texts[doc]

rows ={}
i=1

sentence_task_headers=fieldnames = ["ID","text","query","description","check_one_gold","check_one_gold_reason","_golden"]
random.seed(9001)
sentences_rows = {}
stupid_sentence_index = 0
for doc in doc_texts:
    sentence_data = {}
    text = doc_texts[doc]
    sentences = retrieve_sentences(text)
    length= len(sentences)
    index = random.randint(0,length-1)
    stupid_sentence = stupid_sentence_pull[stupid_sentence_index]
    sentences[index] = stupid_sentence
    converted = convert_sentences_to_sentence_task(sentences)
    sentence_data["ID"] = "t"+str(stupid_sentence_index)
    sentence_data["text"]=converted

    sentence_data["check_one_gold"] = str(index+1)
    sentence_data["check_one_gold_reason"] ="Sentence number "+ str(index+1)+" is out of context"
    sentence_data["_golden"]="TRUE"
    sentences_rows[stupid_sentence_index]=sentence_data
    stupid_sentence_index+=1

with open("sentence_test1.csv","w",encoding="utf8",newline='') as f:
    writer = csv.DictWriter(f, fieldnames=sentence_task_headers)
    writer.writeheader()
    for t in sentences_rows:
        row = sentences_rows[t]
        writer.writerow(row)

