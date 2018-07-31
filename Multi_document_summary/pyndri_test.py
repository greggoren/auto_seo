import pyndri
import params
from  utils import run_bash_command
from time import time
# command="~/.local/bin/PyndriQuery --loglevel warning \
# 	--queries queries.txt \
# 	--index "+params.path_to_index+" \
# 	--smoothing_method dirichlet --smoothing_param auto --prf \
# 	test.run"
begin = time()


index = pyndri.Index(params.path_to_index)
dictionary = pyndri.extract_dictionary(index)
_, int_doc_id = index.document_ids(['clueweb09-en0039-05-00000'])
print([dictionary[token_id]
for token_id in index.document(int_doc_id)[1]])

# Queries the index with 'hello world' and returns the first 1000 results.
# results = index.query('family tree')
# print("it took ",time()-begin)
# for int_document_id, score in results:
#     ext_document_id, _ = index.document(int_document_id)
#     print(ext_document_id, score)
# run_bash_command(command)


