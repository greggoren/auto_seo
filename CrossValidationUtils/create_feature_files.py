from utils import run_bash_command
import os




def create_features_file(features_dir,queries_file,run_name=""):
    run_bash_command("rm -r "+features_dir)
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    command= "/home/greg/auto_seo/scripts/LTRFeatures"+" "+ queries_file + ' -stream=doc -index=' + "/home/greg/cluewebindex" + ' -repository='+ "/home/greg/cluewebindex" +' -useWorkingSet=true -workingSetFile=extended_working_set'+run_name  + ' -workingSetFormat=trec'
    print(command)
    out = run_bash_command(command)
    print(out)
    run_bash_command("mv doc*_* "+features_dir)
    command = "perl "+"/home/greg/auto_seo/scripts/generate.pl"+" "+features_dir+" "+'extended_working_set'+run_name
    print(command)
    out=run_bash_command(command)
    print(out)
    command = "mv features append_features"+run_name
    print(command)
    out = run_bash_command(command)
    print(out)



runs = [25,50,100]
for run in runs:
    run_name = str(run)
    create_features_file("mq_Features_"+run_name,"../scripts/mq_queries.xml",run_name)