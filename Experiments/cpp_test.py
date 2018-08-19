import pyndri
import params
from  CrossValidationUtils import run_bash_command
from time import time
command="~/indri11/bin/IndriRunQuery parameters"
begin = time()
run_bash_command(command)
print("it took ",time()-begin)

