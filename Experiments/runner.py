from utils import run_bash_command
import os

def run():
    for i in [j/10 for j in range(11) if j%2==0]:
        run_name1 = str(i)
        run_name2 = str(i+0.1)
        command = "rm -r /lv_local/home/sgregory/auto_seo/new_merged_index*"
        run_bash_command(command)
        command1="nohup python weaving_experiment_platform.py "+run_name1+" &"
        command2="nohup python weaving_experiment_platform.py "+run_name2+" &"
        run_bash_command(command1)
        run_bash_command(command2)
        while True:
            if os.path.isfile("stop.stop_"+run_name1) and os.path.isfile("stop.stop_"+run_name2):
                break

run()