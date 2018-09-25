import numpy as np
import os
from utils import run_bash_command

class svm_handler:

    def __init__(self):
        pass
        # self.C=C
        # self.w = None


    # def check_prediction(self,y_k):
    #
    #     if np.dot(self.w,y_k.T)>1:
    #         return True
    #     return False
    #
    #
    # def check_validation(self,validation,tags,X):
    #     errors=0
    #     for index in validation:
    #         y_k=X[index]*tags[index]
    #         tmp=np.dot(self.w,y_k.T)
    #         if tmp<1:
    #             errors+=1
    #     return float(errors)/len(validation)
    #
    #
    # def predict(self,X,queries,test_indices,eval,fold,validation=None):
    #     results = {}
    #     for index in test_indices:
    #         results[index] = np.dot(self.w,X[index].T)
    #     return eval.create_trec_eval_file(test_indices,queries,results,str(self.C),"svm",fold,validation)


    def run_svm_rank_model(self,test_file,model_file,fold):
        predictions_folder = "svm_rank_score/"+str(fold)+"/"
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        predictions_file = predictions_folder+os.path.basename(model_file)
        command = "./svm_rank_classify "+test_file +" "+model_file+" "+  predictions_file
        out = run_bash_command(command)
        print(out)
        return predictions_file


    def run_svm_light_model(self,test_file,model_file,fold):
        predictions_folder = "svm_light_score/"+str(fold)+"/"
        if not os.path.exists(predictions_folder):
            os.makedirs(predictions_folder)
        predictions_file = predictions_folder+os.path.basename(model_file)
        command = "./svm_classify "+test_file +" "+model_file+" "+  predictions_file
        out = run_bash_command(command)
        print(out)
        return predictions_file

    def learn_svm_light_model(self,train_file,fold,C_value,number_of_queries):
        C = C_value/number_of_queries
        models_folder = "svm_light_models/" + str(fold) + "/"
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        model_file = models_folder + "model_"+str(C)+".txt"
        command = "./svm_learn -z p -c "+str(C)+" "+ train_file + " " + model_file
        out = run_bash_command(command)
        print(out)
        return model_file


    def learn_svm_rank_model(self,train_file,fold,C):
        models_folder = "svm_light_models/" + str(fold) + "/"
        if not os.path.exists(models_folder):
            os.makedirs(models_folder)
        model_file = models_folder + "model_"+str(C)+".txt"
        command = "./svm_rank_learn -c "+str(C)+" "+ train_file + " " + model_file
        out = run_bash_command(command)
        print(out)
        return model_file

    def retrieve_scores(self,test_indices, score_file):
        with open(score_file) as scores:
            results = {test_indices[i]: score.rstrip() for i, score in enumerate(scores)}
            return results
