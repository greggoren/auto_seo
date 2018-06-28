from utils import run_bash_command,run_bash_command_no_wait
import os
import time
def run():
    for i in [j/10 for j in range(8,11)]:
        run_name1 = str(i)
        command = "rm -r /lv_local/home/sgregory/auto_seo/new_merged_index*"
        run_bash_command(command)
        command1="nohup python weaving_experiment_platform.py "+run_name1+" &"
        run_bash_command_no_wait(command1)
        while True:
            if os.path.isfile("stop.stop_"+run_name1.replace(".","")):
                break

    for i in [j/10 for j in range(8) if j%2==0]:
        run_name1 = str(i)
        run_name2= str(round(i+0.1,1))
        command = "rm -r /lv_local/home/sgregory/auto_seo/new_merged_index*"
        run_bash_command(command)
        command1="nohup python weaving_experiment_platform.py "+run_name1+" &"
        command2="nohup python weaving_experiment_platform.py "+run_name2+" &"
        run_bash_command_no_wait(command1)
        time.sleep(600)
        run_bash_command_no_wait(command2)
        while True:
            if os.path.isfile("stop.stop_"+run_name1.replace(".","")) and os.path.isfile("stop.stop_"+run_name2.replace(".","")):
                break

run()