import pyndri
import params
import pickle
import sys

print("uploading index")
index = pyndri.Index(params.path_to_index)
dic={}
for document_id in range(index.document_base(), index.maximum_document()):
    if document_id%1000000==0:
        print("in document",document_id)
        sys.stdout.flush()
    dic[index.document(document_id)[0]] = document_id
print("loading index finished")
f = open("dic4.pickle","wb")
pickle.dump(dic,f)
f.close()