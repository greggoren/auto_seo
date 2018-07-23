from utils import run_bash_command


def run():
    for i in [j/10 for j in range(11)]:
        run_name1 = str(i)
        command = "rm -r /lv_local/home/sgregory/auto_seo/new_merged_index*"
        run_bash_command(command)
        command1="nohup python pagerank_experiment_platform.py 1 "+run_name1+" > 1_"+run_name1+" &"
        run_bash_command(command1)

run()