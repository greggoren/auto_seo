import xml.etree.ElementTree as ET
import nltk.data

def load_file(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    docs={}
    for doc in root:
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                name=att.text
            else:
                docs[name]=att.text
    return docs



def retrieve_sentences(doc):
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = tokenizer.tokenize(doc)
    return sentences

