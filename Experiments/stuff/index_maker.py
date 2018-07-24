import pyndri
import params
import pickle
import sys
from Preprocess.preprocess import retrieve_ranked_lists


print("uploading index")


index = pyndri.Index(params.path_to_index)
dic={}
for document_id in range(index.document_base(), index.maximum_document()):
    if document_id%1000000==0:
        print("in document",document_id)
        sys.stdout.flush()
    if index.document(document_id)[0].__contains__("ROUND-04"):
        dic[index.document(document_id)[0]] = document_id
print("loading index finished")
f = open("dic4.pickle","wb")
pickle.dump(dic,f)
f.close()
if not dic:
    print("empty dictionary")