from utils import run_bash_command
import os

def run():
    for i in [j/10 for j in range(11)]:
        run_name1 = str(i)
        command = "rm -r /lv_local/home/sgregory/auto_seo/new_merged_index*"
        run_bash_command(command)
        command="nohup python pagerank_experiment_platform.py 1 "+run_name1+" &"
        run_bash_command(command)
        while True:
            if os.path.isfile("stop.stop_1_"+run_name1.replace(".","")):
                break


run()