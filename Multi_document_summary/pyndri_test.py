import pyndri
import params
from  utils import run_bash_command
from time import time
command="./.local/bin/PyndriQuery --loglevel warning \
	--queries commoncore2017_queries.txt \
	--index "+params.path_to_index+" \
	--smoothing_method dirichlet --smoothing_param auto --prf \
	test.run"
begin = time()
run_bash_command(command)
print("it took ",time()-begin)

