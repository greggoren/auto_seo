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
query_env = pyndri.PRFQueryEnvironment(query_env,fb_terms=50,fb_docs=10)
# query_expander = pyndri.QueryExpander(query_env)
results = query_env.query("family tree")
print(results)
# results=query_env.query(results)
# print(results)
# results = index.query('')
print("it took ",time()-begin)




