import xml.etree.ElementTree as ET
import params
def load_file(filename):
    target = open("round1.trectext",'w')
    tree = ET.parse(filename)
    root = tree.getroot()
    docs={}
    for doc in root:
        write=False
        name =""
        for att in doc:
            if att.tag == "DOCNO":
                name=att.text
                if name.__contains__("ROUND-01"):
                    write=True
                    target.write("<DOC>\n")
                    target.write("<DOCNO>"+name+"</DOCNO>\n")
            else:
                if write:
                    target.write("<TEXT>")
                    target.write(att.text)
                    target.write("</TEXT>\n")
                    target.write("</DOC>\n")
                    write=False



load_file(params.trec_text_file)