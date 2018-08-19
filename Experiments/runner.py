from utils import run_bash_command,run_bash_command_no_wait
import os
import time
def run():
    for i in [j/10 for j in range(11)]:
        run_name1 = str(i)
        command = "rm -r /lv_local/home/sgregory/auto_seo/new_merged_index*"
        run_bash_command(command)
        command1="nohup python weaving_experiment_platform.py "+run_name1+" &"
        run_bash_command_no_wait(command1)
        while True:
            if os.path.isfile("stop.stop_"+run_name1.replace(".","")):
                break



run()