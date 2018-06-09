import pyndri
import params
import pickle
import sys

print("uploading index")
index = pyndri.Index(params.path_to_index)
token2id, id2token, id2df = index.get_dictionary()
args = (index,token2id,id2df)
f = open("index.pickle","wb")
pickle.dump(args,f)
f.close()
# del id2df
# del id2token
# dic={}
# for document_id in range(index.document_base(), index.maximum_document()):
#     if document_id%1000000==0:
#         print("in document",document_id)
#         sys.stdout.flush()
#     dic[index.document(document_id)[0]] = document_id
# print("loading index finished")
# f = open("dic.pickle","wb")
# pickle.dump(dic,f)
# f.close()