import os
from utils import run_command

def learn_svm(train_file, C):
    if not os.path.exists("./models_light/"):
        os.makedirs("./models_light/")
    model_file = "./models_light/svm_model" + str(C)
    learning_command = "./svm_rank_learn -c " + str(C) + " " + train_file + " " + model_file
    for output_line in run_command(learning_command):
        print(output_line)
    return model_file

def classify(model_file,fold,test_file,C):
    score_path = "scores/" + str(fold)
    if not os.path.exists(score_path):
        try:
            os.makedirs(score_path)
        except:
            print("collition")
    rank_command = "./svm_rank_classify " + test_file + " " + model_file + " " + score_path + "/" + str(C)
    for output_line in run_command(rank_command):
        print(output_line)
    return score_path + "/" + str(C)



learn_svm("Quality_Features",0.01)