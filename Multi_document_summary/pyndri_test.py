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

index= pyndri.Index(params.path_to_index)
query_env = pyndri.QueryEnvironment(index,rules=('method:linear,collectionLambda:0.4,documentLambda:0.2',))
query_expander = pyndri.QueryExpander(query_env)
results = query_expander.expand("family tree")
# results = index.query('')
print("it took ",time()-begin)
for int_document_id, score in results:
    ext_document_id, _ = index.document(int_document_id)
    print(ext_document_id, score)
run_bash_command(command)


